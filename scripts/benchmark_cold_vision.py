from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

from machboost import MachBoostClient, ensure_server


DATASETS = {
    "chartqa": ("lmms-lab/ChartQA", "test"),
    "textvqa": ("lmms-lab/textvqa", "validation"),
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    index: int
    image: str
    image_digest: str
    question: str
    answers: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive first-view acceleration on unique public images."
    )
    parser.add_argument("--model", default="qwen3-vl:2b")
    parser.add_argument(
        "--datasets",
        default="chartqa,textvqa",
        help="Comma-separated public datasets: chartqa,textvqa.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--cold-mode", choices=["adaptive", "fast", "balanced", "quality"], default="adaptive")
    parser.add_argument("--vision-max-edge", type=int)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--work-dir", type=Path, default=Path("results/tmp/cold_vision"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = tuple(name.strip().lower() for name in args.datasets.split(",") if name.strip())
    unknown = sorted(set(dataset_names) - set(DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {', '.join(unknown)}")
    if not dataset_names:
        raise ValueError("at least one dataset is required")
    if args.samples_per_dataset < 1:
        raise ValueError("samples per dataset must be at least 1")

    samples, warmups = load_public_samples(
        dataset_names,
        samples_per_dataset=args.samples_per_dataset,
        work_dir=args.work_dir,
    )
    client, server_started = ensure_server(args.endpoint, timeout=min(30.0, args.timeout))
    client.timeout = args.timeout
    client.stop(args.model)
    loaded = client.load(
        args.model,
        options={"vision_cache_size": 4},
        keep_alive="forever",
    )

    warmup_rows = []
    for sample in warmups:
        warmup_rows.append(
            run_request(
                client,
                args.model,
                sample,
                mode="baseline",
                cold_mode="off",
                max_tokens=args.max_tokens,
                vision_max_edge=None,
            )
        )
        warmup_rows.append(
            run_request(
                client,
                args.model,
                sample,
                mode="accelerated",
                cold_mode=args.cold_mode,
                max_tokens=args.max_tokens,
                vision_max_edge=args.vision_max_edge,
            )
        )

    rows: list[dict[str, Any]] = []
    for pair_index, sample in enumerate(samples):
        order = ("baseline", "accelerated") if pair_index % 2 == 0 else ("accelerated", "baseline")
        pair: dict[str, dict[str, Any]] = {}
        for mode in order:
            row = run_request(
                client,
                args.model,
                sample,
                mode=mode,
                cold_mode="off" if mode == "baseline" else args.cold_mode,
                max_tokens=args.max_tokens,
                vision_max_edge=None if mode == "baseline" else args.vision_max_edge,
            )
            pair[mode] = row
            rows.append(row)

        baseline = pair["baseline"]
        accelerated = pair["accelerated"]
        paired_speedup = baseline["client_total_seconds"] / accelerated["client_total_seconds"]
        literal_equal = baseline["output"] == accelerated["output"]
        normalized_equal = normalize_answer(baseline["output"]) == normalize_answer(
            accelerated["output"]
        )
        for row in pair.values():
            row["paired_total_speedup"] = paired_speedup
            row["paired_literal_output_equal"] = literal_equal
            row["paired_normalized_output_equal"] = normalized_equal

    summary = summarize(rows)
    output = args.output or Path(
        f"results/local/cold_vision_{args.model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "machboost.cold_vision_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "paired unique-image cold requests with visual and prompt caches disabled",
        "model": args.model,
        "resolved_model": loaded["instance"]["model"],
        "backend": loaded["instance"]["backend"],
        "datasets": list(dataset_names),
        "dataset_sources": {name: f"https://huggingface.co/datasets/{DATASETS[name][0]}" for name in dataset_names},
        "samples_per_dataset": args.samples_per_dataset,
        "max_tokens": args.max_tokens,
        "cold_mode": args.cold_mode,
        "vision_max_edge": args.vision_max_edge,
        "server_started": server_started,
        "load_duration_seconds": loaded["load_duration_seconds"],
        "warmup_policy": "one held-out unique image per dataset and mode; warm-up rows excluded",
        "order_policy": "alternating baseline-first and accelerated-first pairs",
        "machine": machine_data(),
        "packages": package_versions(),
        "summary": summary,
        "warmups": warmup_rows,
        "rows": rows,
        "resident_models": client.ps(),
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"artifact: {output}")


def load_public_samples(
    dataset_names: Sequence[str],
    *,
    samples_per_dataset: int,
    work_dir: Path,
) -> tuple[list[Sample], list[Sample]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Public cold-vision benchmarks require `pip install datasets`.") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    warmups: list[Sample] = []
    for dataset_name in dataset_names:
        hub_name, split = DATASETS[dataset_name]
        rows = load_dataset(hub_name, split=split, streaming=True)
        unique: list[Sample] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            image = row["image"].convert("RGB")
            digest = image_digest(image)
            if digest in seen:
                continue
            seen.add(digest)
            target = work_dir / dataset_name / f"{digest}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                image.save(target)
            unique.append(
                Sample(
                    dataset=dataset_name,
                    index=index,
                    image=str(target.resolve()),
                    image_digest=digest,
                    question=str(row["question"]),
                    answers=answers_for(dataset_name, row),
                )
            )
            if len(unique) >= samples_per_dataset + 1:
                break
        if len(unique) < samples_per_dataset + 1:
            raise RuntimeError(
                f"{dataset_name} yielded only {len(unique)} unique images; "
                f"needed {samples_per_dataset + 1}"
            )
        warmups.append(unique[0])
        samples.extend(unique[1:])
    return samples, warmups


def answers_for(dataset_name: str, row: dict[str, Any]) -> tuple[str, ...]:
    if dataset_name == "chartqa":
        return (str(row["answer"]),)
    raw = row["answers"]
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return tuple(str(answer) for answer in raw)


def image_digest(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.mode).encode("ascii"))
    digest.update(repr(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def run_request(
    client: MachBoostClient,
    model: str,
    sample: Sample,
    *,
    mode: str,
    cold_mode: str,
    max_tokens: int,
    vision_max_edge: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_predict": max_tokens,
        "temperature": 0.0,
        "no_vision_cache": True,
        "vision_cache_size": 4,
        "cold_vision": cold_mode,
    }
    if vision_max_edge is not None:
        options["vision_max_edge"] = int(vision_max_edge)

    prompt = f"Answer only, with no explanation: {sample.question}"
    started = time.perf_counter()
    first_text_at = None
    output: list[str] = []
    final: dict[str, Any] = {}
    events = client.generate(
        model,
        prompt,
        images=[sample.image],
        options=options,
        keep_alive="forever",
        stream=True,
    )
    for event in events:
        text = str(event.get("response") or "")
        if text:
            if first_text_at is None:
                first_text_at = time.perf_counter()
            output.append(text)
        if event.get("done"):
            final = event
    finished = time.perf_counter()
    if not final:
        raise RuntimeError("generation stream ended without final metrics")

    stats = dict((final.get("machboost") or {}).get("stats") or {})
    rendered = "".join(output).strip()
    return {
        **asdict(sample),
        "mode": mode,
        "output": rendered,
        "expected_match": answer_matches(rendered, sample.answers),
        "client_total_seconds": finished - started,
        "client_ttft_seconds": None if first_text_at is None else first_text_at - started,
        "server_total_seconds": float(final.get("total_duration", 0)) / 1_000_000_000,
        "generated_tokens": int(final.get("eval_count") or stats.get("generated_tokens") or 0),
        "prompt_tokens": int(stats.get("prompt_tokens") or 0),
        "prompt_tokens_per_second": float(stats.get("prompt_tokens_per_second") or 0.0),
        "generation_tokens_per_second": float(stats.get("generation_tokens_per_second") or 0.0),
        "peak_memory_gb": float(stats.get("peak_memory_gb") or 0.0),
        "mean_token_logprob": _optional_float(stats.get("mean_token_logprob")),
        "minimum_token_logprob": _optional_float(stats.get("minimum_token_logprob")),
        "visual_cache_hit": bool(stats.get("visual_cache_hit")),
        "visual_cache_miss": bool(stats.get("visual_cache_miss")),
        "prompt_cache_prefix_tokens": int(stats.get("prompt_cache_prefix_tokens") or 0),
        "cold_vision": dict(stats.get("cold_vision") or {}),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row for row in rows if row["mode"] == "baseline"]
    accelerated = [row for row in rows if row["mode"] == "accelerated"]
    if not baseline or len(baseline) != len(accelerated):
        raise ValueError("summary requires equally sized baseline and accelerated rows")
    baseline_total = statistics.median(row["client_total_seconds"] for row in baseline)
    accelerated_total = statistics.median(row["client_total_seconds"] for row in accelerated)
    baseline_ttft = statistics.median(row["client_ttft_seconds"] for row in baseline)
    accelerated_ttft = statistics.median(row["client_ttft_seconds"] for row in accelerated)
    baseline_tokens = statistics.median(row["prompt_tokens"] for row in baseline)
    accelerated_tokens = statistics.median(row["prompt_tokens"] for row in accelerated)
    decisions = [row.get("cold_vision") or {} for row in accelerated]
    return {
        "pairs": len(accelerated),
        "unique_images": len({row["image_digest"] for row in accelerated}),
        "baseline_median_total_seconds": baseline_total,
        "accelerated_median_total_seconds": accelerated_total,
        "median_total_speedup": baseline_total / accelerated_total,
        "median_paired_total_speedup": statistics.median(
            row["paired_total_speedup"] for row in accelerated
        ),
        "baseline_median_ttft_seconds": baseline_ttft,
        "accelerated_median_ttft_seconds": accelerated_ttft,
        "median_ttft_speedup": baseline_ttft / accelerated_ttft,
        "baseline_median_prompt_tokens": baseline_tokens,
        "accelerated_median_prompt_tokens": accelerated_tokens,
        "median_prompt_token_reduction_rate": (
            0.0 if baseline_tokens <= 0 else 1.0 - accelerated_tokens / baseline_tokens
        ),
        "baseline_expected_match_rate": _rate(row["expected_match"] for row in baseline),
        "accelerated_expected_match_rate": _rate(row["expected_match"] for row in accelerated),
        "paired_literal_output_equal_rate": _rate(
            row["paired_literal_output_equal"] for row in accelerated
        ),
        "paired_normalized_output_equal_rate": _rate(
            row["paired_normalized_output_equal"] for row in accelerated
        ),
        "accelerated_policy_enabled_rate": _rate(
            decision.get("enabled", False) for decision in decisions
        ),
        "selected_max_edges": sorted(
            {int(decision["target_max_edge"]) for decision in decisions if decision.get("target_max_edge")}
        ),
        "cache_hit_count": sum(
            row["visual_cache_hit"] or row["prompt_cache_prefix_tokens"] > 0 for row in rows
        ),
        "datasets": {
            dataset: _dataset_summary(rows, dataset)
            for dataset in sorted({row["dataset"] for row in rows})
        },
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _dataset_summary(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    subset = [row for row in rows if row["dataset"] == dataset]
    baseline = [row for row in subset if row["mode"] == "baseline"]
    accelerated = [row for row in subset if row["mode"] == "accelerated"]
    return {
        "pairs": len(accelerated),
        "median_paired_total_speedup": statistics.median(
            row["paired_total_speedup"] for row in accelerated
        ),
        "baseline_expected_match_rate": _rate(row["expected_match"] for row in baseline),
        "accelerated_expected_match_rate": _rate(row["expected_match"] for row in accelerated),
        "paired_normalized_output_equal_rate": _rate(
            row["paired_normalized_output_equal"] for row in accelerated
        ),
    }


def normalize_answer(value: str) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def answer_matches(output: str, expected: Iterable[str]) -> bool:
    normalized_output = normalize_answer(output)
    return any(
        normalized_expected and normalized_expected in normalized_output
        for normalized_expected in (normalize_answer(answer) for answer in expected)
    )


def _rate(values: Iterable[bool]) -> float:
    items = tuple(bool(value) for value in values)
    return sum(items) / len(items) if items else 0.0


def machine_data() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("machboost", "datasets", "mlx", "mlx-lm", "mlx-vlm", "transformers", "Pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


if __name__ == "__main__":
    main()
