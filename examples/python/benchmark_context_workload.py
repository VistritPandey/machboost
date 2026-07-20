from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from machboost import benchmark_context_acceleration, context_fingerprint

from context_example_utils import DEFAULT_MODEL, load_accelerator, read_text_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native and MachBoost generation on one loaded model. A speedup is valid "
            "only when every paired token sequence matches."
        )
    )
    parser.add_argument("--context", action="append", required=True, help="Text file or directory; repeatable.")
    parser.add_argument("--prompt", action="append", default=[], help="Workload prompt; repeatable.")
    parser.add_argument("--prompt-file", action="append", default=[], help="Prompt file; repeatable.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("mlx", "hf"), default="mlx")
    parser.add_argument("--runs", type=int, default=6, help="Even number of measured pairs.")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--max-draft-tokens", type=int, default=64)
    parser.add_argument("--reentry-probe-tokens", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 2 or args.runs % 2:
        raise SystemExit("--runs must be an even number of at least 2")
    cases = [(f"prompt-{index + 1}", prompt) for index, prompt in enumerate(args.prompt)]
    for raw_path in args.prompt_file:
        path = Path(raw_path).expanduser().resolve()
        cases.append((path.name, path.read_text(encoding="utf-8")))
    if not cases:
        raise SystemExit("provide at least one --prompt or --prompt-file")

    documents = read_text_paths(args.context, max_chars=args.max_context_chars)
    if not documents:
        raise SystemExit("no readable context found")
    context_texts = [document.text for document in documents]

    load_started = time.perf_counter()
    accelerator = load_accelerator(
        model=args.model,
        backend=args.backend,
        context_texts=context_texts,
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
        reentry_probe_tokens=args.reentry_probe_tokens,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    load_seconds = time.perf_counter() - load_started

    artifacts = []
    fingerprint = context_fingerprint(context_texts)
    for name, prompt in cases:
        artifact = benchmark_context_acceleration(
            accelerator,
            prompt,
            model=args.model,
            backend=args.backend,
            context_fingerprint=fingerprint,
            context_chars=sum(len(text) for text in context_texts),
            runs=args.runs,
            warmups=args.warmups,
            max_tokens=args.max_tokens,
        )
        artifact["case"] = name
        artifacts.append(artifact)

    result = build_result(artifacts, load_seconds=load_seconds)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"saved {output}", file=sys.stderr)
    raise SystemExit(0 if result["summary"]["valid"] else 1)


def build_result(artifacts: list[dict], *, load_seconds: float) -> dict:
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    valid = bool(rows) and all(row["output_match"] for row in rows)
    engaged = [row for row in rows if row["accepted_draft_tokens"] > 0]
    speedups = [float(row["diagnostic_speedup"]) for row in rows]
    return {
        "schema_version": "machboost.example_context_workload.v1",
        "model_load_seconds": load_seconds,
        "summary": {
            "cases": len(artifacts),
            "measured_pairs": len(rows),
            "valid": valid,
            "output_match_rate": (
                sum(1 for row in rows if row["output_match"]) / len(rows) if rows else 0.0
            ),
            "algorithm_engaged_rate": len(engaged) / len(rows) if rows else 0.0,
            "median_native_wall_seconds": _median(
                row["native"]["wall_seconds"] for row in rows
            ),
            "median_machboost_wall_seconds": _median(
                row["machboost"]["wall_seconds"] for row in rows
            ),
            "valid_median_speedup": _median(speedups) if valid else None,
            "diagnostic_median_speedup": _median(speedups),
            "median_accepted_draft_tokens": _median(
                row["accepted_draft_tokens"] for row in rows
            ),
        },
        "cases": artifacts,
        "notes": [
            "Both paths use the same loaded model, weights, prompt, and tokenizer.",
            "A valid speedup requires exact token equality for every measured pair.",
            "Zero accepted draft tokens means the context algorithm did not engage.",
            "Results apply only to the measured model, workload, machine, and settings.",
        ],
    }


def print_report(result: dict) -> None:
    for artifact in result["cases"]:
        summary = artifact["summary"]
        speedup = summary["median_speedup"]
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "invalid"
        print(
            f"{artifact['case']}: exact={summary['output_match_rate']:.1%} "
            f"engaged={summary['algorithm_engaged_rate']:.1%} speedup={speedup_text}"
        )
    summary = result["summary"]
    speedup = summary["valid_median_speedup"]
    print("\nSame-model workload result")
    print(f"  model load:       {result['model_load_seconds']:.3f}s")
    print(f"  measured pairs:   {summary['measured_pairs']}")
    print(f"  exact pairs:      {summary['output_match_rate']:.1%}")
    print(f"  algorithm engaged:{summary['algorithm_engaged_rate']:>7.1%}")
    print(f"  native median:    {summary['median_native_wall_seconds']:.3f}s")
    print(f"  MachBoost median: {summary['median_machboost_wall_seconds']:.3f}s")
    print(f"  accepted tokens:  {summary['median_accepted_draft_tokens']:.1f}")
    print(f"  valid speedup:    {f'{speedup:.3f}x' if speedup is not None else 'not reported'}")
    if summary["algorithm_engaged_rate"] == 0:
        print("  note: the supplied context did not accelerate this workload")


def _median(values) -> float:
    data = [float(value) for value in values]
    return float(statistics.median(data)) if data else 0.0


if __name__ == "__main__":
    main()
