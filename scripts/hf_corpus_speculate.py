#!/usr/bin/env python3
"""Local corpus speculative decoding spike for Hugging Face causal LMs.

This script is intentionally standalone. It tests the MachBoost mechanism:
draft likely continuations from local context, then verify those candidate
tokens with the target model.
"""

from __future__ import annotations

import argparse
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
    elapsed_ms: int
    tokens_per_second: float
    output: str


@dataclass
class CompareStats:
    schema_version: str
    model: str
    prompt_tokens: int
    context_tokens: int
    max_new_tokens: int
    ngram: int
    max_draft_tokens: int
    warmup_tokens: int
    baseline: RunStats
    speculative: RunStats
    output_match: bool
    wall_clock_speedup: float
    forward_reduction_percent: float
    verdict: str
    note: str


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


def verify_candidate_cached(model, current_logits, past_key_values, candidate_tokens: list[int], device: str):
    import torch

    if not candidate_tokens:
        return [], past_key_values, current_logits, 0
    if next_greedy_from_logits(current_logits) != candidate_tokens[0]:
        return [], past_key_values, current_logits, 0

    input_ids = torch.tensor([candidate_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids, past_key_values=past_key_values, use_cache=True)

    logits = output.logits
    for offset in range(1, len(candidate_tokens)):
        predicted = next_greedy_from_logits(logits[:, offset - 1, :])
        if predicted != candidate_tokens[offset]:
            return [], past_key_values, current_logits, 1
    return candidate_tokens, output.past_key_values, logits[:, -1, :], 1


def baseline_generate(model, tokenizer, prompt_ids: list[int], max_new_tokens: int, device: str) -> RunStats:
    generated = list(prompt_ids)
    output_tokens: list[int] = []
    forwards = 0
    start = time.perf_counter()
    current_logits, past_key_values = prefill(model, generated, device)
    forwards += 1

    while len(output_tokens) < max_new_tokens:
        token = next_greedy_from_logits(current_logits)
        output_tokens.append(token)
        generated.append(token)
        if len(output_tokens) >= max_new_tokens:
            break
        current_logits, past_key_values = advance_cache(model, past_key_values, token, device)
        forwards += 1

    elapsed = time.perf_counter() - start
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return RunStats(
        mode="baseline_cached",
        generated_tokens=len(output_tokens),
        model_forwards=forwards,
        accepted_draft_tokens=0,
        accepted_draft_spans=0,
        normal_tokens=len(output_tokens),
        elapsed_ms=int(elapsed * 1000),
        tokens_per_second=(len(output_tokens) / elapsed) if elapsed > 0 else 0,
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
    forwards += 1

    while len(output_tokens) < max_new_tokens:
        candidates = find_candidates(generated, source_tokens, index, ngram, max_draft_tokens, candidate_limit)
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
            )
            forwards += verify_forwards
            if accepted:
                past_key_values = next_past
                current_logits = next_logits
                generated.extend(accepted)
                output_tokens.extend(accepted)
                accepted_draft_tokens += len(accepted)
                accepted_draft_spans += 1
                break
        if len(output_tokens) >= max_new_tokens:
            break
        if candidates and accepted:
            continue

        token = next_greedy_from_logits(current_logits)
        generated.append(token)
        output_tokens.append(token)
        normal_tokens += 1
        if len(output_tokens) >= max_new_tokens:
            break
        current_logits, past_key_values = advance_cache(model, past_key_values, token, device)
        forwards += 1

    elapsed = time.perf_counter() - start
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return RunStats(
        mode="corpus_speculative_cached",
        generated_tokens=len(output_tokens),
        model_forwards=forwards,
        accepted_draft_tokens=accepted_draft_tokens,
        accepted_draft_spans=accepted_draft_spans,
        normal_tokens=normal_tokens,
        elapsed_ms=int(elapsed * 1000),
        tokens_per_second=(len(output_tokens) / elapsed) if elapsed > 0 else 0,
        output=output,
    )


def run_compare(args: argparse.Namespace) -> CompareStats:
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

    prompt = read_text(args.prompt)
    context = read_context(args.context, args.max_context_chars)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    source_ids = tokenizer.encode(prompt + context, add_special_tokens=False)

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
    )

    speedup = baseline.elapsed_ms / speculative.elapsed_ms if speculative.elapsed_ms > 0 else 0
    forward_reduction = (
        ((baseline.model_forwards - speculative.model_forwards) / baseline.model_forwards) * 100
        if baseline.model_forwards
        else 0
    )
    if speculative.accepted_draft_tokens == 0:
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
        context_tokens=len(source_ids) - len(prompt_ids),
        max_new_tokens=args.max_new_tokens,
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
        warmup_tokens=args.warmup_tokens,
        baseline=baseline,
        speculative=speculative,
        output_match=baseline.output == speculative.output,
        wall_clock_speedup=speedup,
        forward_reduction_percent=forward_reduction,
        verdict=verdict,
        note="Prototype verifier using Hugging Face KV cache. Forward reduction is the main mechanism metric; wall-clock is hardware/runtime dependent.",
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
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=4)
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
        result = asdict(run_compare(args))

    if args.json or args.self_test:
        print(json.dumps(result, indent=2))
    else:
        print(f"model: {result['model']}")
        print(f"verdict: {result['verdict']}")
        print(f"baseline: {result['baseline']['elapsed_ms']}ms, {result['baseline']['tokens_per_second']:.2f} tok/s")
        print(
            "speculative: "
            f"{result['speculative']['elapsed_ms']}ms, {result['speculative']['tokens_per_second']:.2f} tok/s"
        )
        print(f"wall-clock speedup: {result['wall_clock_speedup']:.2f}x")
        print(f"forward reduction: {result['forward_reduction_percent']:.1f}%")
        print(f"accepted draft tokens: {result['speculative']['accepted_draft_tokens']}")
        print(f"output match: {result['output_match']}")
        print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
