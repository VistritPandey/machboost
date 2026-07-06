#!/usr/bin/env python3
"""Compare MachBoost with Hugging Face prompt lookup decoding.

This script measures four paths on the same prompt fixture:

1. serial Hugging Face ``generate`` with greedy decoding;
2. Hugging Face built-in prompt lookup decoding;
3. MachBoost using prompt tokens as the draft source;
4. MachBoost using local context/prompt-context tokens as the draft source.

The comparison is intentionally narrow. Hugging Face prompt lookup only drafts
from tokens already present in ``input_ids``. MachBoost can draft from any
caller-provided corpus, but accepted tokens still have to match the target
model's greedy continuation exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backend_bench_matrix as matrix
import hf_corpus_speculate as hfs


SCHEMA_VERSION = "machboost.hf_prompt_lookup_compare.v1"


@dataclass(frozen=True)
class MethodRow:
    fixture: str
    workflow: str
    expectation: str
    nonce: str
    method: str
    source_mode: str
    output_match: bool
    elapsed_ms: float
    tokens_per_second: float
    speedup_vs_serial: float
    model_forwards: int
    forward_reduction_percent: float
    accepted_draft_tokens: int
    generated_tokens: int
    raw_generated_tokens: int
    generated_token_ids: list[int]
    output_preview: str
    note: str


def parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def deterministic_nonce(seed: str, repeat: int, fixture: str, *, fresh: bool) -> str:
    salt = time.time_ns() if fresh else "fixed"
    digest = hashlib.sha256(f"{seed}:{repeat}:{fixture}:{salt}".encode("utf-8")).hexdigest()
    return f"mb-{digest[:10]}"


def build_fixtures(args: argparse.Namespace) -> list[matrix.Fixture]:
    names = parse_str_list(args.fixtures)
    return [
        matrix.build_fixture(name, deterministic_nonce(args.seed, repeat, name, fresh=args.fresh_nonce))
        for repeat in range(1, args.repeat + 1)
        for name in names
    ]


def synchronize(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def greedy_generate_hf(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: str,
    prompt_lookup_num_tokens: Optional[int] = None,
    max_matching_ngram_size: Optional[int] = None,
) -> dict[str, Any]:
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if prompt_lookup_num_tokens is not None:
        kwargs["prompt_lookup_num_tokens"] = prompt_lookup_num_tokens
        if max_matching_ngram_size is not None:
            kwargs["max_matching_ngram_size"] = max_matching_ngram_size

    forward_calls = 0
    original_forward = model.forward

    def counted_forward(*forward_args, **forward_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*forward_args, **forward_kwargs)

    model.forward = counted_forward
    try:
        synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, attention_mask=attention_mask, **kwargs)
        synchronize(device)
        elapsed = time.perf_counter() - started
    finally:
        model.forward = original_forward

    sequences = output.sequences if hasattr(output, "sequences") else output
    raw_generated = [int(token) for token in sequences[0, input_ids.shape[-1] :].detach().cpu().tolist()]
    generated = raw_generated[:max_new_tokens]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "elapsed_ms": elapsed * 1000,
        "token_ids": generated,
        "raw_generated_tokens": len(raw_generated),
        "text": text,
        "tokens_per_second": (len(generated) / elapsed) if elapsed > 0 else 0.0,
        "model_forwards": forward_calls,
    }


def machboost_generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    source_ids: list[int],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    stats = hfs.speculative_generate(
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
    return {
        "elapsed_ms": float(stats.elapsed_ms),
        "token_ids": list(stats.token_ids),
        "raw_generated_tokens": len(stats.token_ids),
        "text": stats.output,
        "tokens_per_second": stats.tokens_per_second,
        "model_forwards": stats.model_forwards,
        "accepted_draft_tokens": stats.accepted_draft_tokens,
    }


def make_row(
    fixture: matrix.Fixture,
    *,
    method: str,
    source_mode: str,
    result: dict[str, Any],
    serial: dict[str, Any],
    note: str,
) -> MethodRow:
    serial_elapsed = float(serial["elapsed_ms"])
    elapsed = float(result["elapsed_ms"])
    serial_forwards = int(serial["model_forwards"])
    forwards = int(result["model_forwards"])
    return MethodRow(
        fixture=fixture.name,
        workflow=fixture.workflow,
        expectation=fixture.expectation,
        nonce=fixture.nonce,
        method=method,
        source_mode=source_mode,
        output_match=list(serial["token_ids"]) == list(result["token_ids"]),
        elapsed_ms=elapsed,
        tokens_per_second=float(result["tokens_per_second"]),
        speedup_vs_serial=(serial_elapsed / elapsed) if elapsed > 0 else 0.0,
        model_forwards=forwards,
        forward_reduction_percent=(
            ((serial_forwards - forwards) / serial_forwards) * 100 if serial_forwards > 0 else 0.0
        ),
        accepted_draft_tokens=int(result.get("accepted_draft_tokens", 0)),
        generated_tokens=len(result["token_ids"]),
        raw_generated_tokens=int(result.get("raw_generated_tokens", len(result["token_ids"]))),
        generated_token_ids=list(result["token_ids"]),
        output_preview=matrix.preview(str(result["text"])),
        note=note,
    )


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = build_fixtures(args)
    if args.list_fixtures:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixtures": [asdict(fixture) for fixture in fixtures],
        }
    if args.dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "settings": settings(args),
            "fixtures": [asdict(fixture) for fixture in fixtures],
        }

    runtime_args = argparse.Namespace(
        model=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    tokenizer, model, device = hfs.load_runtime(runtime_args)

    rows: list[MethodRow] = []
    prompt_lookup_sweep = parse_int_list(args.prompt_lookup_sweep)
    source_modes = parse_str_list(args.machboost_source_modes)

    for fixture in fixtures:
        print(f"[{fixture.name}] serial/HF prompt lookup/MachBoost", file=sys.stderr, flush=True)
        prompt_ids = tokenizer.encode(fixture.prompt, add_special_tokens=False)
        hfs.warmup_model(model, prompt_ids, device)
        serial = greedy_generate_hf(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        rows.append(
            make_row(
                fixture,
                method="hf_serial_generate",
                source_mode="prompt",
                result=serial,
                serial=serial,
                note="Greedy Hugging Face generate baseline.",
            )
        )

        for lookup_tokens in prompt_lookup_sweep:
            lookup = greedy_generate_hf(
                model,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
                prompt_lookup_num_tokens=lookup_tokens,
                max_matching_ngram_size=args.max_matching_ngram_size,
            )
            rows.append(
                make_row(
                    fixture,
                    method=f"hf_prompt_lookup_{lookup_tokens}",
                    source_mode="prompt",
                    result=lookup,
                    serial=serial,
                    note=(
                        "Hugging Face prompt lookup; candidates come only from tokens "
                        "already present in the prompt."
                    ),
                )
            )

        for source_mode in source_modes:
            source_ids = tokenizer.encode(matrix.source_text(fixture, source_mode), add_special_tokens=False)
            boosted = machboost_generate(model, tokenizer, prompt_ids, source_ids, args, device)
            rows.append(
                make_row(
                    fixture,
                    method=f"machboost_{source_mode}",
                    source_mode=source_mode,
                    result=boosted,
                    serial=serial,
                    note=(
                        "MachBoost corpus drafter with target-model verification; "
                        "accepted tokens must match greedy generation."
                    ),
                )
            )

    data_rows = [asdict(row) for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": False,
        "settings": settings(args),
        "fixtures": [asdict(fixture) for fixture in fixtures],
        "summaries": summarize_methods(rows),
        "fixture_summaries": summarize_fixture_methods(rows),
        "rows": data_rows,
    }


def settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "repeat": args.repeat,
        "fixtures": parse_str_list(args.fixtures),
        "max_new_tokens": args.max_new_tokens,
        "ngram": args.ngram,
        "max_draft_tokens": args.max_draft_tokens,
        "candidate_limit": args.candidate_limit,
        "prompt_lookup_sweep": parse_int_list(args.prompt_lookup_sweep),
        "max_matching_ngram_size": args.max_matching_ngram_size,
        "machboost_source_modes": parse_str_list(args.machboost_source_modes),
        "verify_mode": args.verify_mode,
        "anchor_tokens": args.anchor_tokens,
        "draft_policy": args.draft_policy,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "fresh_nonce": args.fresh_nonce,
    }


def summarize_methods(rows: Iterable[MethodRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[MethodRow]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)
    summaries = []
    for method, method_rows in sorted(grouped.items()):
        count = len(method_rows)
        summaries.append(
            {
                "method": method,
                "rows": count,
                "output_match_rate": sum(1 for row in method_rows if row.output_match) / count if count else 0.0,
                "median_speedup_vs_serial": median([row.speedup_vs_serial for row in method_rows]),
                "p90_speedup_vs_serial": percentile([row.speedup_vs_serial for row in method_rows], 90),
                "median_tokens_per_second": median([row.tokens_per_second for row in method_rows]),
                "median_model_forwards": median([row.model_forwards for row in method_rows]),
                "median_forward_reduction_percent": median(
                    [row.forward_reduction_percent for row in method_rows]
                ),
                "median_accepted_draft_tokens": median([row.accepted_draft_tokens for row in method_rows]),
            }
        )
    return summaries


def summarize_fixture_methods(rows: Iterable[MethodRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[MethodRow]] = {}
    for row in rows:
        grouped.setdefault((row.fixture, row.method), []).append(row)
    summaries = []
    for (fixture, method), fixture_rows in sorted(grouped.items()):
        count = len(fixture_rows)
        first = fixture_rows[0]
        summaries.append(
            {
                "fixture": fixture,
                "method": method,
                "workflow": first.workflow,
                "expectation": first.expectation,
                "rows": count,
                "output_match_rate": sum(1 for row in fixture_rows if row.output_match) / count if count else 0.0,
                "median_speedup_vs_serial": median([row.speedup_vs_serial for row in fixture_rows]),
                "median_tokens_per_second": median([row.tokens_per_second for row in fixture_rows]),
                "median_forward_reduction_percent": median(
                    [row.forward_reduction_percent for row in fixture_rows]
                ),
                "median_accepted_draft_tokens": median([row.accepted_draft_tokens for row in fixture_rows]),
            }
        )
    return summaries


def median(values: Iterable[float | int]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def percentile(values: Iterable[float | int], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def format_table(result: dict[str, Any]) -> str:
    if result.get("dry_run"):
        return json.dumps(result, indent=2)
    summaries = result.get("summaries", [])
    if not summaries:
        return "No rows."
    headers = ["method", "match", "speedup", "tok/s", "forwards", "fwd red", "accepted"]
    table_rows = []
    for row in summaries:
        table_rows.append(
            [
                row["method"],
                f"{row['output_match_rate'] * 100:.0f}%",
                f"{row['median_speedup_vs_serial']:.2f}x",
                f"{row['median_tokens_per_second']:.2f}",
                f"{row['median_model_forwards']:.0f}",
                f"{row['median_forward_reduction_percent']:.1f}%",
                f"{row['median_accepted_draft_tokens']:.0f}",
            ]
        )
    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        " | ".join("-" * width for width in widths),
    ]
    for row in table_rows:
        lines.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def run_self_test() -> dict[str, Any]:
    rows = [
        MethodRow(
            fixture="copy",
            workflow="unit",
            expectation="positive",
            nonce="mb-test",
            method="hf_serial_generate",
            source_mode="prompt",
            output_match=True,
            elapsed_ms=100.0,
            tokens_per_second=10.0,
            speedup_vs_serial=1.0,
            model_forwards=8,
            forward_reduction_percent=0.0,
            accepted_draft_tokens=0,
            generated_tokens=8,
            raw_generated_tokens=8,
            generated_token_ids=[1, 2],
            output_preview="ok",
            note="baseline",
        ),
        MethodRow(
            fixture="copy",
            workflow="unit",
            expectation="positive",
            nonce="mb-test",
            method="machboost_prompt",
            source_mode="prompt",
            output_match=True,
            elapsed_ms=50.0,
            tokens_per_second=20.0,
            speedup_vs_serial=2.0,
            model_forwards=4,
            forward_reduction_percent=50.0,
            accepted_draft_tokens=6,
            generated_tokens=8,
            raw_generated_tokens=8,
            generated_token_ids=[1, 2],
            output_preview="ok",
            note="boosted",
        ),
    ]
    summaries = summarize_methods(rows)
    ok = (
        parse_int_list("4,8") == [4, 8]
        and len(summaries) == 2
        and summaries[1]["median_speedup_vs_serial"] == 2.0
    )
    return {
        "schema_version": SCHEMA_VERSION + ".self_test",
        "ok": ok,
        "summaries": summaries,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MachBoost with HF prompt lookup decoding.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--fixtures", default="real_readme_api,real_core_code,policy,json,rag,code")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", default="machboost-hf-lookup")
    parser.add_argument("--fresh-nonce", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-lookup-sweep", default="8")
    parser.add_argument("--max-matching-ngram-size", type=int, default=2)
    parser.add_argument("--machboost-source-modes", default="prompt,context,prompt-context")
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=1)
    parser.add_argument("--verify-mode", choices=["block", "hybrid", "sequential"], default="hybrid")
    parser.add_argument("--anchor-tokens", type=int, default=1)
    parser.add_argument("--min-verify-margin", type=float, default=0.0)
    parser.add_argument("--draft-policy", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--initial-draft-tokens", type=int, default=2)
    parser.add_argument("--min-draft-tokens", type=int, default=1)
    parser.add_argument("--draft-step", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-fixtures", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = run_self_test()
    else:
        result = run_comparison(args)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.json or args.self_test or args.list_fixtures:
        print(json.dumps(result, indent=2))
    else:
        print(format_table(result))
    return 0 if not args.self_test or result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
