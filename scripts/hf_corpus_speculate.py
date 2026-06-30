#!/usr/bin/env python3
"""Local corpus speculative decoding spike for Hugging Face causal LMs.

This script is intentionally standalone. It tests the MachBoost mechanism:
draft likely continuations from local context, then verify those candidate
tokens with the target model.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {".git", ".cache", "build", "dist", "node_modules", "target", "vendor"}


@dataclass
class Candidate:
    tokens: list[int]
    matched_suffix_tokens: int
    source_start: int


@dataclass
class RunStats:
    mode: str
    generated_tokens: int
    model_forwards: int
    accepted_draft_tokens: int
    accepted_draft_spans: int
    normal_tokens: int
    prefill_ms: int
    decode_ms: int
    elapsed_ms: int
    tokens_per_second: float
    decode_tokens_per_second: float
    output: str


@dataclass
class CompareStats:
    schema_version: str
    model: str
    prompt_tokens: int
    context_tokens: int
    source_tokens: int
    max_new_tokens: int
    ngram: int
    max_draft_tokens: int
    draft_policy: str
    source_mode: str
    verify_mode: str
    anchor_tokens: int
    min_verify_margin: float
    warmup_tokens: int
    baseline: RunStats
    speculative: RunStats
    output_match: bool
    wall_clock_speedup: float
    forward_reduction_percent: float
    verdict: str
    note: str


@dataclass
class AutoDraftStats:
    schema_version: str
    model: str
    prompt_tokens: int
    context_tokens: int
    source_tokens: int
    max_new_tokens: int
    ngram: int
    draft_policy: str
    source_mode: str
    verify_mode: str
    anchor_tokens: int
    min_verify_margin: float
    warmup_tokens: int
    baseline: RunStats
    runs: list[CompareStats]
    best_max_draft_tokens: int
    best_wall_clock_speedup: float
    best_forward_reduction_percent: float
    verdict: str


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def iter_context_files(path: str) -> Iterable[Path]:
    root = Path(path)
    if root.is_file():
        yield root
        return
    for item in root.rglob("*"):
        if any(part in SKIP_DIRS for part in item.parts):
            continue
        if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES:
            yield item


def read_context(path: str | None, max_chars: int) -> str:
    if not path:
        return ""
    chunks: list[str] = []
    remaining = max_chars
    for item in iter_context_files(path):
        if remaining <= 0:
            break
        text = item.read_text(encoding="utf-8", errors="ignore")
        if not text:
            continue
        excerpt = text[:remaining]
        chunks.append(f"\n\n# file: {item}\n{excerpt}")
        remaining -= len(excerpt)
    return "".join(chunks)


def build_index(source_tokens: list[int], ngram: int) -> dict[tuple[int, ...], list[int]]:
    index: dict[tuple[int, ...], list[int]] = {}
    for pos in range(0, max(0, len(source_tokens) - ngram + 1)):
        key = tuple(source_tokens[pos : pos + ngram])
        index.setdefault(key, []).append(pos)
    return index


def find_candidate(
    generated_tokens: list[int],
    source_tokens: list[int],
    index: dict[tuple[int, ...], list[int]],
    ngram: int,
    max_draft_tokens: int,
    max_suffix_tokens: int = 32,
) -> Candidate | None:
    if len(generated_tokens) < ngram:
        return None
    longest_suffix = min(max_suffix_tokens, len(generated_tokens))
    best: Candidate | None = None
    for suffix_len in range(longest_suffix, ngram - 1, -1):
        suffix = generated_tokens[-suffix_len:]
        positions = index.get(tuple(suffix[-ngram:]), [])
        for pos in positions:
            start = pos - (suffix_len - ngram)
            if start < 0 or source_tokens[start : start + suffix_len] != suffix:
                continue
            draft_start = start + suffix_len
            draft_end = min(draft_start + max_draft_tokens, len(source_tokens))
            if draft_start >= draft_end:
                continue
            candidate = Candidate(
                tokens=source_tokens[draft_start:draft_end],
                matched_suffix_tokens=suffix_len,
                source_start=draft_start,
            )
            if best is None or candidate.matched_suffix_tokens > best.matched_suffix_tokens:
                best = candidate
        if best is not None:
            return best
    return None


def find_candidates(
    generated_tokens: list[int],
    source_tokens: list[int],
    index: dict[tuple[int, ...], list[int]],
    ngram: int,
    max_draft_tokens: int,
    limit: int,
    max_suffix_tokens: int = 32,
) -> list[Candidate]:
    if len(generated_tokens) < ngram:
        return []
    longest_suffix = min(max_suffix_tokens, len(generated_tokens))
    candidates: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()
    for suffix_len in range(longest_suffix, ngram - 1, -1):
        suffix = generated_tokens[-suffix_len:]
        positions = index.get(tuple(suffix[-ngram:]), [])
        for pos in positions:
            start = pos - (suffix_len - ngram)
            if start < 0 or source_tokens[start : start + suffix_len] != suffix:
                continue
            draft_start = start + suffix_len
            draft_end = min(draft_start + max_draft_tokens, len(source_tokens))
            if draft_start >= draft_end:
                continue
            tokens = source_tokens[draft_start:draft_end]
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(
                    tokens=tokens,
                    matched_suffix_tokens=suffix_len,
                    source_start=draft_start,
                )
            )
        if len(candidates) >= limit:
            return candidates[:limit]
    return candidates[:limit]


def pick_device(requested: str):
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def next_greedy_from_logits(logits) -> int:
    import torch

    return int(torch.argmax(logits, dim=-1).item())


def greedy_token_and_margin(logits) -> tuple[int, float]:
    import torch

    flat = logits.reshape(-1)
    if flat.numel() < 2:
        return int(torch.argmax(flat).item()), float("inf")
    values, indices = torch.topk(flat, k=2)
    margin = float((values[0] - values[1]).detach().cpu().item())
    return int(indices[0].item()), margin


def token_passes_margin(logits, expected_token: int, min_margin: float) -> bool:
    token, margin = greedy_token_and_margin(logits)
    return token == expected_token and margin >= min_margin


def prefill(model, token_ids: list[int], device: str):
    import torch

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids, use_cache=True)
    return output.logits[:, -1, :], output.past_key_values


def warmup_model(model, prompt_ids: list[int], device: str) -> None:
    import torch

    if not prompt_ids:
        return
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(input_ids, use_cache=True).logits[:, -1, :]


def advance_cache(model, past_key_values, token: int, device: str):
    import torch

    input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return output.logits[:, -1, :], output.past_key_values


def clone_cache(past_key_values):
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(item.clone() if hasattr(item, "clone") else item for item in layer)
            for layer in past_key_values
        )
    if hasattr(past_key_values, "layers"):
        cloned = copy.copy(past_key_values)
        cloned.layers = []
        for layer in past_key_values.layers:
            layer_clone = copy.copy(layer)
            if hasattr(layer, "keys"):
                layer_clone.keys = layer.keys.clone()
            if hasattr(layer, "values"):
                layer_clone.values = layer.values.clone()
            cloned.layers.append(layer_clone)
        return cloned
    return copy.deepcopy(past_key_values)


def cache_length(past_key_values) -> int | None:
    if isinstance(past_key_values, tuple):
        for layer in past_key_values:
            for item in layer:
                if hasattr(item, "shape") and len(item.shape) >= 3:
                    return int(item.shape[-2])
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
                return int(keys.shape[-2])
    return None


def crop_cache(past_key_values, max_length: int):
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return past_key_values
    if isinstance(past_key_values, tuple):
        cropped_layers = []
        for layer in past_key_values:
            cropped_items = []
            for item in layer:
                if hasattr(item, "shape") and len(item.shape) >= 3:
                    cropped_items.append(item[..., :max_length, :])
                else:
                    cropped_items.append(item)
            cropped_layers.append(tuple(cropped_items))
        return tuple(cropped_layers)
    return past_key_values


def verify_candidate_block(
    model,
    current_logits,
    past_key_values,
    candidate_tokens: list[int],
    device: str,
    min_verify_margin: float,
):
    import torch

    if not candidate_tokens:
        return [], past_key_values, current_logits, 0
    if not token_passes_margin(current_logits, candidate_tokens[0], min_verify_margin):
        return [], past_key_values, current_logits, 0

    trial_past = clone_cache(past_key_values)
    base_cache_length = cache_length(trial_past)
    input_ids = torch.tensor([candidate_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids, past_key_values=trial_past, use_cache=True)

    logits = output.logits
    for offset in range(1, len(candidate_tokens)):
        if not token_passes_margin(logits[:, offset - 1, :], candidate_tokens[offset], min_verify_margin):
            accepted = candidate_tokens[:offset]
            next_past = output.past_key_values
            if base_cache_length is not None:
                next_past = crop_cache(next_past, base_cache_length + len(accepted))
            return accepted, next_past, logits[:, offset - 1, :], 1
    return candidate_tokens, output.past_key_values, logits[:, -1, :], 1


def verify_candidate_sequential(
    model,
    current_logits,
    past_key_values,
    candidate_tokens: list[int],
    device: str,
    min_verify_margin: float,
):
    if not candidate_tokens:
        return [], past_key_values, current_logits, 0

    accepted: list[int] = []
    trial_past = clone_cache(past_key_values)
    trial_logits = current_logits
    forwards = 0
    for token in candidate_tokens:
        if not token_passes_margin(trial_logits, token, min_verify_margin):
            if accepted:
                return accepted, trial_past, trial_logits, forwards
            return [], past_key_values, current_logits, forwards
        accepted.append(token)
        trial_logits, trial_past = advance_cache(model, trial_past, token, device)
        forwards += 1
    return accepted, trial_past, trial_logits, forwards


def verify_candidate_hybrid(
    model,
    current_logits,
    past_key_values,
    candidate_tokens: list[int],
    device: str,
    min_verify_margin: float,
    anchor_tokens: int,
):
    if not candidate_tokens:
        return [], past_key_values, current_logits, 0

    anchored: list[int] = []
    trial_past = clone_cache(past_key_values)
    trial_logits = current_logits
    forwards = 0
    anchor_count = min(max(0, anchor_tokens), len(candidate_tokens))

    for token in candidate_tokens[:anchor_count]:
        if not token_passes_margin(trial_logits, token, min_verify_margin):
            return [], past_key_values, current_logits, forwards
        anchored.append(token)
        trial_logits, trial_past = advance_cache(model, trial_past, token, device)
        forwards += 1

    tail = candidate_tokens[anchor_count:]
    if not tail:
        return anchored, trial_past, trial_logits, forwards

    tail_accepted, tail_past, tail_logits, tail_forwards = verify_candidate_block(
        model,
        trial_logits,
        trial_past,
        tail,
        device,
        min_verify_margin,
    )
    forwards += tail_forwards
    if not tail_accepted:
        if anchored:
            return anchored, trial_past, trial_logits, forwards
        return [], past_key_values, current_logits, forwards
    return anchored + tail_accepted, tail_past, tail_logits, forwards


def verify_candidate_cached(
    model,
    current_logits,
    past_key_values,
    candidate_tokens: list[int],
    device: str,
    verify_mode: str,
    min_verify_margin: float,
    anchor_tokens: int,
):
    if verify_mode == "sequential":
        return verify_candidate_sequential(
            model, current_logits, past_key_values, candidate_tokens, device, min_verify_margin
        )
    if verify_mode == "hybrid":
        return verify_candidate_hybrid(
            model,
            current_logits,
            past_key_values,
            candidate_tokens,
            device,
            min_verify_margin,
            anchor_tokens,
        )
    return verify_candidate_block(model, current_logits, past_key_values, candidate_tokens, device, min_verify_margin)


def baseline_generate(model, tokenizer, prompt_ids: list[int], max_new_tokens: int, device: str) -> RunStats:
    generated = list(prompt_ids)
    output_tokens: list[int] = []
    forwards = 0
    start = time.perf_counter()
    current_logits, past_key_values = prefill(model, generated, device)
    prefill_elapsed = time.perf_counter() - start
    forwards += 1
    decode_start = time.perf_counter()

    while len(output_tokens) < max_new_tokens:
        token = next_greedy_from_logits(current_logits)
        output_tokens.append(token)
        generated.append(token)
        if len(output_tokens) >= max_new_tokens:
            break
        current_logits, past_key_values = advance_cache(model, past_key_values, token, device)
        forwards += 1

    decode_elapsed = time.perf_counter() - decode_start
    elapsed = time.perf_counter() - start
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return RunStats(
        mode="baseline_cached",
        generated_tokens=len(output_tokens),
        model_forwards=forwards,
        accepted_draft_tokens=0,
        accepted_draft_spans=0,
        normal_tokens=len(output_tokens),
        prefill_ms=int(prefill_elapsed * 1000),
        decode_ms=int(decode_elapsed * 1000),
        elapsed_ms=int(elapsed * 1000),
        tokens_per_second=(len(output_tokens) / elapsed) if elapsed > 0 else 0,
        decode_tokens_per_second=(len(output_tokens) / decode_elapsed) if decode_elapsed > 0 else 0,
        output=output,
    )


def speculative_generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    source_tokens: list[int],
    max_new_tokens: int,
    ngram: int,
    max_draft_tokens: int,
    device: str,
    candidate_limit: int,
    verify_mode: str,
    min_verify_margin: float,
    anchor_tokens: int,
    draft_policy: str,
    initial_draft_tokens: int,
    min_draft_tokens: int,
    draft_step: int,
) -> RunStats:
    generated = list(prompt_ids)
    output_tokens: list[int] = []
    index = build_index(source_tokens, ngram)
    forwards = 0
    accepted_draft_tokens = 0
    accepted_draft_spans = 0
    normal_tokens = 0
    start = time.perf_counter()
    current_logits, past_key_values = prefill(model, generated, device)
    prefill_elapsed = time.perf_counter() - start
    forwards += 1
    decode_start = time.perf_counter()
    current_draft_tokens = max_draft_tokens
    if draft_policy == "adaptive":
        current_draft_tokens = min(max_draft_tokens, max(min_draft_tokens, initial_draft_tokens))

    while len(output_tokens) < max_new_tokens:
        candidates = find_candidates(generated, source_tokens, index, ngram, current_draft_tokens, candidate_limit)
        accepted: list[int] = []
        for candidate in candidates:
            remaining = max_new_tokens - len(output_tokens)
            candidate_tokens = candidate.tokens[:remaining]
            accepted, next_past, next_logits, verify_forwards = verify_candidate_cached(
                model,
                current_logits,
                past_key_values,
                candidate_tokens,
                device,
                verify_mode,
                min_verify_margin,
                anchor_tokens,
            )
            forwards += verify_forwards
            if accepted:
                past_key_values = next_past
                current_logits = next_logits
                generated.extend(accepted)
                output_tokens.extend(accepted)
                accepted_draft_tokens += len(accepted)
                accepted_draft_spans += 1
                if draft_policy == "adaptive" and len(accepted) >= current_draft_tokens:
                    current_draft_tokens = min(max_draft_tokens, current_draft_tokens + draft_step)
                break
        if len(output_tokens) >= max_new_tokens:
            break
        if candidates and accepted:
            continue
        if draft_policy == "adaptive":
            current_draft_tokens = max(min_draft_tokens, current_draft_tokens - draft_step)

        token = next_greedy_from_logits(current_logits)
        generated.append(token)
        output_tokens.append(token)
        normal_tokens += 1
        if len(output_tokens) >= max_new_tokens:
            break
        current_logits, past_key_values = advance_cache(model, past_key_values, token, device)
        forwards += 1

    decode_elapsed = time.perf_counter() - decode_start
    elapsed = time.perf_counter() - start
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return RunStats(
        mode="corpus_speculative_cached",
        generated_tokens=len(output_tokens),
        model_forwards=forwards,
        accepted_draft_tokens=accepted_draft_tokens,
        accepted_draft_spans=accepted_draft_spans,
        normal_tokens=normal_tokens,
        prefill_ms=int(prefill_elapsed * 1000),
        decode_ms=int(decode_elapsed * 1000),
        elapsed_ms=int(elapsed * 1000),
        tokens_per_second=(len(output_tokens) / elapsed) if elapsed > 0 else 0,
        decode_tokens_per_second=(len(output_tokens) / decode_elapsed) if decode_elapsed > 0 else 0,
        output=output,
    )


def load_runtime(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=torch.float16 if device == "mps" else torch.float32,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def prepare_inputs(args: argparse.Namespace, tokenizer) -> tuple[list[int], list[int], int]:
    prompt = read_text(args.prompt)
    context = read_context(args.context, args.max_context_chars)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if args.source_mode == "context":
        source_text = context
    elif args.source_mode == "prompt":
        source_text = prompt
    else:
        source_text = prompt + context
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)
    return prompt_ids, source_ids, len(context_ids)


def compare_with_runtime(
    args: argparse.Namespace,
    tokenizer,
    model,
    device,
    prompt_ids,
    source_ids,
    context_token_count: int,
) -> CompareStats:
    warmup_model(model, prompt_ids, device)
    if args.warmup_tokens > 0:
        _ = baseline_generate(model, tokenizer, prompt_ids, args.warmup_tokens, device)
        _ = speculative_generate(
            model,
            tokenizer,
            prompt_ids,
            source_ids,
            args.warmup_tokens,
            args.ngram,
            args.max_draft_tokens,
            device,
            args.candidate_limit,
            args.verify_mode,
            args.min_verify_margin,
            args.anchor_tokens,
            args.draft_policy,
            args.initial_draft_tokens,
            args.min_draft_tokens,
            args.draft_step,
        )
    baseline = baseline_generate(model, tokenizer, prompt_ids, args.max_new_tokens, device)
    speculative = speculative_generate(
        model,
        tokenizer,
        prompt_ids,
        source_ids,
        args.max_new_tokens,
        args.ngram,
        args.max_draft_tokens,
        device,
        args.candidate_limit,
        args.verify_mode,
        args.min_verify_margin,
        args.anchor_tokens,
        args.draft_policy,
        args.initial_draft_tokens,
        args.min_draft_tokens,
        args.draft_step,
    )

    speedup = baseline.elapsed_ms / speculative.elapsed_ms if speculative.elapsed_ms > 0 else 0
    forward_reduction = (
        ((baseline.model_forwards - speculative.model_forwards) / baseline.model_forwards) * 100
        if baseline.model_forwards
        else 0
    )
    output_match = baseline.output == speculative.output
    if not output_match:
        verdict = "output_mismatch"
    elif speculative.accepted_draft_tokens == 0:
        verdict = "no_draft_acceptance"
    elif forward_reduction >= 20:
        verdict = "mechanism_viable"
    elif forward_reduction >= 5:
        verdict = "weak_mechanism_signal"
    else:
        verdict = "no_clear_mechanism_gain"
    return CompareStats(
        schema_version="machboost.hf_corpus_speculate.v1",
        model=args.model,
        prompt_tokens=len(prompt_ids),
        context_tokens=context_token_count,
        source_tokens=len(source_ids),
        max_new_tokens=args.max_new_tokens,
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
        draft_policy=args.draft_policy,
        source_mode=args.source_mode,
        verify_mode=args.verify_mode,
        anchor_tokens=args.anchor_tokens,
        min_verify_margin=args.min_verify_margin,
        warmup_tokens=args.warmup_tokens,
        baseline=baseline,
        speculative=speculative,
        output_match=output_match,
        wall_clock_speedup=speedup,
        forward_reduction_percent=forward_reduction,
        verdict=verdict,
        note="Prototype verifier using Hugging Face KV cache. Forward reduction is the main mechanism metric; wall-clock is hardware/runtime dependent.",
    )


def run_compare(args: argparse.Namespace) -> CompareStats:
    tokenizer, model, device = load_runtime(args)
    prompt_ids, source_ids, context_token_count = prepare_inputs(args, tokenizer)
    return compare_with_runtime(args, tokenizer, model, device, prompt_ids, source_ids, context_token_count)


def parse_draft_sweep(value: str) -> list[int]:
    drafts: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        drafts.append(int(part))
    return drafts or [2, 4, 6, 8]


def run_auto_draft(args: argparse.Namespace) -> AutoDraftStats:
    tokenizer, model, device = load_runtime(args)
    prompt_ids, source_ids, context_token_count = prepare_inputs(args, tokenizer)
    warmup_model(model, prompt_ids, device)
    if args.warmup_tokens > 0:
        _ = baseline_generate(model, tokenizer, prompt_ids, args.warmup_tokens, device)

    baseline = baseline_generate(model, tokenizer, prompt_ids, args.max_new_tokens, device)
    runs: list[CompareStats] = []
    best: CompareStats | None = None

    for draft_tokens in parse_draft_sweep(args.draft_sweep):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.max_draft_tokens = draft_tokens
        speculative = speculative_generate(
            model,
            tokenizer,
            prompt_ids,
            source_ids,
            trial_args.max_new_tokens,
            trial_args.ngram,
            trial_args.max_draft_tokens,
            device,
            trial_args.candidate_limit,
            trial_args.verify_mode,
            trial_args.min_verify_margin,
            trial_args.anchor_tokens,
            trial_args.draft_policy,
            trial_args.initial_draft_tokens,
            trial_args.min_draft_tokens,
            trial_args.draft_step,
        )
        speedup = baseline.elapsed_ms / speculative.elapsed_ms if speculative.elapsed_ms > 0 else 0
        forward_reduction = (
            ((baseline.model_forwards - speculative.model_forwards) / baseline.model_forwards) * 100
            if baseline.model_forwards
            else 0
        )
        output_match = baseline.output == speculative.output
        if not output_match:
            verdict = "output_mismatch"
        elif speculative.accepted_draft_tokens == 0:
            verdict = "no_draft_acceptance"
        elif forward_reduction >= 20:
            verdict = "mechanism_viable"
        elif forward_reduction >= 5:
            verdict = "weak_mechanism_signal"
        else:
            verdict = "no_clear_mechanism_gain"
        result = CompareStats(
            schema_version="machboost.hf_corpus_speculate.v1",
            model=args.model,
            prompt_tokens=len(prompt_ids),
            context_tokens=context_token_count,
            source_tokens=len(source_ids),
            max_new_tokens=args.max_new_tokens,
            ngram=args.ngram,
            max_draft_tokens=draft_tokens,
            draft_policy=args.draft_policy,
            source_mode=args.source_mode,
            verify_mode=args.verify_mode,
            anchor_tokens=args.anchor_tokens,
            min_verify_margin=args.min_verify_margin,
            warmup_tokens=args.warmup_tokens,
            baseline=baseline,
            speculative=speculative,
            output_match=output_match,
            wall_clock_speedup=speedup,
            forward_reduction_percent=forward_reduction,
            verdict=verdict,
            note="Prototype verifier using Hugging Face KV cache. Forward reduction is the main mechanism metric; wall-clock is hardware/runtime dependent.",
        )
        runs.append(result)
        if result.output_match and (best is None or result.wall_clock_speedup > best.wall_clock_speedup):
            best = result

    if best is None:
        best = max(runs, key=lambda item: item.wall_clock_speedup)
    verdict = "mechanism_viable" if best.output_match and best.forward_reduction_percent >= 20 else best.verdict
    return AutoDraftStats(
        schema_version="machboost.hf_auto_draft.v1",
        model=args.model,
        prompt_tokens=len(prompt_ids),
        context_tokens=context_token_count,
        source_tokens=len(source_ids),
        max_new_tokens=args.max_new_tokens,
        ngram=args.ngram,
        draft_policy=args.draft_policy,
        source_mode=args.source_mode,
        verify_mode=args.verify_mode,
        anchor_tokens=args.anchor_tokens,
        min_verify_margin=args.min_verify_margin,
        warmup_tokens=args.warmup_tokens,
        baseline=baseline,
        runs=runs,
        best_max_draft_tokens=best.max_draft_tokens,
        best_wall_clock_speedup=best.wall_clock_speedup,
        best_forward_reduction_percent=best.forward_reduction_percent,
        verdict=verdict,
    )


def run_self_test() -> dict:
    source = [1, 2, 3, 4, 5, 6, 7, 8]
    generated = [1, 2, 3, 4]
    index = build_index(source, 2)
    candidate = find_candidate(generated, source, index, ngram=2, max_draft_tokens=4)
    ok = candidate is not None and candidate.tokens == [5, 6, 7, 8]
    return {
        "schema_version": "machboost.hf_corpus_speculate.self_test.v1",
        "ok": ok,
        "candidate_tokens": candidate.tokens if candidate else [],
        "matched_suffix_tokens": candidate.matched_suffix_tokens if candidate else 0,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local corpus speculative decoding for HF causal LMs.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", help="Prompt file")
    parser.add_argument("--context", help="Context file or directory")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--max-draft-tokens", type=int, default=16)
    parser.add_argument("--draft-policy", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--initial-draft-tokens", type=int, default=2)
    parser.add_argument("--min-draft-tokens", type=int, default=1)
    parser.add_argument("--draft-step", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--auto-draft", action="store_true")
    parser.add_argument("--draft-sweep", default="2,4,6,8")
    parser.add_argument("--source-mode", choices=["prompt-context", "context", "prompt"], default="prompt-context")
    parser.add_argument("--verify-mode", choices=["block", "sequential", "hybrid"], default="block")
    parser.add_argument("--anchor-tokens", type=int, default=1)
    parser.add_argument("--min-verify-margin", type=float, default=0.0)
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = run_self_test()
    else:
        if not args.prompt:
            raise SystemExit("--prompt is required unless --self-test is used")
        result = asdict(run_auto_draft(args) if args.auto_draft else run_compare(args))

    if args.json or args.self_test:
        print(json.dumps(result, indent=2))
    else:
        print(f"model: {result['model']}")
        print(f"verdict: {result['verdict']}")
        if "best_max_draft_tokens" in result:
            print(f"best max draft tokens: {result['best_max_draft_tokens']}")
            print(f"best wall-clock speedup: {result['best_wall_clock_speedup']:.2f}x")
            print(f"best forward reduction: {result['best_forward_reduction_percent']:.1f}%")
            return 0
        print(f"baseline: {result['baseline']['elapsed_ms']}ms, {result['baseline']['tokens_per_second']:.2f} tok/s")
        print(
            "baseline decode: "
            f"{result['baseline']['decode_ms']}ms, {result['baseline']['decode_tokens_per_second']:.2f} tok/s"
        )
        print(
            "speculative: "
            f"{result['speculative']['elapsed_ms']}ms, {result['speculative']['tokens_per_second']:.2f} tok/s"
        )
        print(
            "speculative decode: "
            f"{result['speculative']['decode_ms']}ms, {result['speculative']['decode_tokens_per_second']:.2f} tok/s"
        )
        print(f"wall-clock speedup: {result['wall_clock_speedup']:.2f}x")
        print(f"forward reduction: {result['forward_reduction_percent']:.1f}%")
        print(f"accepted draft tokens: {result['speculative']['accepted_draft_tokens']}")
        print(f"output match: {result['output_match']}")
        print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
