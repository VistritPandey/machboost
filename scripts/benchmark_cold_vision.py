from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

from machboost import MachBoostClient, ensure_server
from machboost.vision_auto import VISION_TOKEN_REQUEST_MODES


@dataclass(frozen=True)
class DatasetSpec:
    repository: str
    split: str
    config: str | None = None

    @property
    def source(self) -> str:
        return f"https://huggingface.co/datasets/{self.repository}"


DATASETS = {
    "chartqa": DatasetSpec("lmms-lab/ChartQA", "test"),
    "docvqa": DatasetSpec("lmms-lab/DocVQA", "validation", "DocVQA"),
    "mmmu": DatasetSpec("lmms-lab/MMMU", "dev"),
    "textvqa": DatasetSpec("lmms-lab/textvqa", "validation"),
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    index: int
    image: str
    image_digest: str
    question: str
    answers: tuple[str, ...]
    images: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive first-view acceleration on unique public images."
    )
    parser.add_argument("--model", default="qwen3-vl:2b")
    parser.add_argument(
        "--datasets",
        default="chartqa,textvqa",
        help="Comma-separated public datasets: chartqa,docvqa,mmmu,textvqa.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument(
        "--cold-mode",
        choices=["off", "adaptive", "fast", "balanced", "quality"],
        default="adaptive",
    )
    parser.add_argument("--vision-max-edge", type=int)
    parser.add_argument(
        "--vision-tokens",
        choices=VISION_TOKEN_REQUEST_MODES,
        default="off",
    )
    parser.add_argument("--vision-token-ratio", type=float, default=0.35)
    parser.add_argument("--vision-token-layer", type=int)
    parser.add_argument("--vision-token-bucket", type=int)
    parser.add_argument("--vision-calibration", type=Path)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Replay uncertain accelerated answers at full resolution below this mean token log-probability.",
    )
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
                vision_token_mode="off",
                vision_token_ratio=args.vision_token_ratio,
                vision_token_layer=None,
                vision_token_bucket=None,
                vision_calibration=None,
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
                vision_token_mode=args.vision_tokens,
                vision_token_ratio=args.vision_token_ratio,
                vision_token_layer=args.vision_token_layer,
                vision_token_bucket=args.vision_token_bucket,
                vision_calibration=args.vision_calibration,
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
                vision_token_mode="off" if mode == "baseline" else args.vision_tokens,
                vision_token_ratio=args.vision_token_ratio,
                vision_token_layer=None if mode == "baseline" else args.vision_token_layer,
                vision_token_bucket=None if mode == "baseline" else args.vision_token_bucket,
                vision_calibration=None if mode == "baseline" else args.vision_calibration,
            )
            if mode == "accelerated" and args.confidence_threshold is not None:
                row = verify_uncertain_request(
                    client,
                    args.model,
                    sample,
                    first_pass=row,
                    confidence_threshold=args.confidence_threshold,
                    max_tokens=args.max_tokens,
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
        "dataset_sources": {name: DATASETS[name].source for name in dataset_names},
        "samples_per_dataset": args.samples_per_dataset,
        "max_tokens": args.max_tokens,
        "cold_mode": args.cold_mode,
        "vision_max_edge": args.vision_max_edge,
        "vision_tokens": args.vision_tokens,
        "vision_token_ratio": args.vision_token_ratio,
        "vision_token_layer": args.vision_token_layer,
        "vision_token_bucket": args.vision_token_bucket,
        "vision_calibration": (
            None if args.vision_calibration is None else str(args.vision_calibration.resolve())
        ),
        "confidence_threshold": args.confidence_threshold,
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
        spec = DATASETS[dataset_name]
        rows = load_dataset(
            spec.repository,
            spec.config,
            split=spec.split,
            streaming=True,
        )
        unique: list[Sample] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            row_images = tuple(image.convert("RGB") for image in images_for(dataset_name, row))
            if not row_images:
                continue
            digest = images_digest(row_images)
            if digest in seen:
                continue
            seen.add(digest)
            target_dir = work_dir / dataset_name / digest
            target_dir.mkdir(parents=True, exist_ok=True)
            image_paths = []
            for image_index, image in enumerate(row_images):
                target = target_dir / f"{image_index}.png"
                if not target.exists():
                    image.save(target)
                image_paths.append(str(target.resolve()))
            unique.append(
                Sample(
                    dataset=dataset_name,
                    index=index,
                    image=image_paths[0],
                    image_digest=digest,
                    question=question_for(dataset_name, row),
                    answers=answers_for(dataset_name, row),
                    images=tuple(image_paths),
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
    if dataset_name == "mmmu":
        answer = str(row["answer"])
        options = _literal_list(row.get("options"))
        index = ord(answer.upper()) - ord("A") if len(answer) == 1 else -1
        selected = str(options[index]) if 0 <= index < len(options) else ""
        return tuple(value for value in (answer, selected) if value)
    raw = row["answers"]
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return tuple(str(answer) for answer in raw)


def images_for(dataset_name: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if dataset_name == "mmmu":
        return tuple(
            row.get(f"image_{index}")
            for index in range(1, 8)
            if row.get(f"image_{index}") is not None
        )
    image = row.get("image")
    return () if image is None else (image,)


def question_for(dataset_name: str, row: dict[str, Any]) -> str:
    question = str(row["question"])
    if dataset_name != "mmmu":
        return question
    options = _literal_list(row.get("options"))
    if not options:
        return question
    rendered = "\n".join(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(options)
    )
    return f"{question}\nOptions:\n{rendered}\nReturn only the option letter."


def _literal_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value or ())


def image_digest(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.mode).encode("ascii"))
    digest.update(repr(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def images_digest(images: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(bytes.fromhex(image_digest(image)))
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
    vision_token_mode: str = "off",
    vision_token_ratio: float = 0.35,
    vision_token_layer: int | None = None,
    vision_token_bucket: int | None = None,
    vision_calibration: Path | str | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_predict": max_tokens,
        "temperature": 0.0,
        "no_vision_cache": True,
        "vision_cache_size": 4,
        "cold_vision": cold_mode,
        "vision_tokens": vision_token_mode,
        "vision_token_ratio": vision_token_ratio,
    }
    if vision_max_edge is not None:
        options["vision_max_edge"] = int(vision_max_edge)
    if vision_token_layer is not None:
        options["vision_token_layer"] = int(vision_token_layer)
    if vision_token_bucket is not None:
        options["vision_token_bucket"] = int(vision_token_bucket)
    if vision_calibration is not None:
        options["vision_calibration"] = str(Path(vision_calibration).expanduser().resolve())

    prompt = f"Answer only, with no explanation: {sample.question}"
    started = time.perf_counter()
    first_text_at = None
    output: list[str] = []
    final: dict[str, Any] = {}
    events = client.generate(
        model,
        prompt,
        images=list(sample.images or (sample.image,)),
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
        "post_fusion_vision": dict(stats.get("post_fusion_vision") or {}),
    }


def verify_uncertain_request(
    client: MachBoostClient,
    model: str,
    sample: Sample,
    *,
    first_pass: dict[str, Any],
    confidence_threshold: float,
    max_tokens: int,
) -> dict[str, Any]:
    confidence = first_pass.get("mean_token_logprob")
    should_fallback = confidence is None or float(confidence) < confidence_threshold
    verification = {
        "enabled": True,
        "confidence_threshold": confidence_threshold,
        "fallback": should_fallback,
        "first_pass_output": first_pass["output"],
        "first_pass_expected_match": first_pass["expected_match"],
        "first_pass_total_seconds": first_pass["client_total_seconds"],
        "first_pass_ttft_seconds": first_pass["client_ttft_seconds"],
        "first_pass_mean_token_logprob": confidence,
        "first_pass_minimum_token_logprob": first_pass.get("minimum_token_logprob"),
    }
    if not should_fallback:
        return {**first_pass, "verification": verification}

    fallback = run_request(
        client,
        model,
        sample,
        mode="fallback",
        cold_mode="off",
        max_tokens=max_tokens,
        vision_max_edge=None,
        vision_token_mode="off",
    )
    verification.update(
        {
            "fallback_output": fallback["output"],
            "fallback_expected_match": fallback["expected_match"],
            "fallback_total_seconds": fallback["client_total_seconds"],
            "fallback_ttft_seconds": fallback["client_ttft_seconds"],
        }
    )
    return {
        **fallback,
        "mode": "accelerated",
        "client_total_seconds": (
            first_pass["client_total_seconds"] + fallback["client_total_seconds"]
        ),
        "client_ttft_seconds": (
            first_pass["client_total_seconds"] + fallback["client_ttft_seconds"]
        ),
        "server_total_seconds": (
            first_pass["server_total_seconds"] + fallback["server_total_seconds"]
        ),
        "generated_tokens": first_pass["generated_tokens"] + fallback["generated_tokens"],
        "prompt_tokens": first_pass["prompt_tokens"] + fallback["prompt_tokens"],
        "cold_vision": first_pass["cold_vision"],
        "verification": verification,
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
    verifications = [row.get("verification") or {} for row in accelerated]
    post_fusion = [row.get("post_fusion_vision") or {} for row in accelerated]
    confidence = paired_bootstrap_intervals(baseline, accelerated)
    return {
        "pairs": len(accelerated),
        "unique_images": len({row["image_digest"] for row in accelerated}),
        "baseline_median_total_seconds": baseline_total,
        "accelerated_median_total_seconds": accelerated_total,
        "median_total_speedup": baseline_total / accelerated_total,
        "aggregate_total_speedup": sum(row["client_total_seconds"] for row in baseline)
        / sum(row["client_total_seconds"] for row in accelerated),
        "aggregate_total_speedup_ci95": confidence["aggregate_total_speedup"],
        "median_paired_total_speedup": statistics.median(
            row["paired_total_speedup"] for row in accelerated
        ),
        "median_paired_total_speedup_ci95": confidence["median_paired_total_speedup"],
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
        "expected_match_rate_delta": (
            _rate(row["expected_match"] for row in accelerated)
            - _rate(row["expected_match"] for row in baseline)
        ),
        "expected_match_rate_delta_ci95": confidence["expected_match_rate_delta"],
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
        "verification_fallback_rate": _rate(
            verification.get("fallback", False) for verification in verifications
        ),
        "first_pass_expected_match_rate": _rate(
            verification.get("first_pass_expected_match", row["expected_match"])
            for row, verification in zip(accelerated, verifications)
        ),
        "post_fusion_enabled_rate": _rate(
            decision.get("enabled", False) for decision in post_fusion
        ),
        "median_actual_visual_retention_ratio": _optional_median(
            decision.get("actual_visual_retention_ratio") for decision in post_fusion
        ),
        "datasets": {
            dataset: _dataset_summary(rows, dataset)
            for dataset in sorted({row["dataset"] for row in rows})
        },
    }


def paired_bootstrap_intervals(
    baseline: Sequence[dict[str, Any]],
    accelerated: Sequence[dict[str, Any]],
    *,
    draws: int = 2000,
    seed: int = 20260716,
) -> dict[str, list[float]]:
    if not baseline or len(baseline) != len(accelerated):
        raise ValueError("paired bootstrap requires equally sized non-empty rows")
    count = len(baseline)
    rng = random.Random(seed)
    speedups = []
    paired_medians = []
    accuracy_deltas = []
    for _ in range(max(1, int(draws))):
        selected = [rng.randrange(count) for _ in range(count)]
        baseline_seconds = sum(float(baseline[index]["client_total_seconds"]) for index in selected)
        accelerated_seconds = sum(
            float(accelerated[index]["client_total_seconds"]) for index in selected
        )
        speedups.append(baseline_seconds / accelerated_seconds)
        paired_medians.append(
            statistics.median(
                float(baseline[index]["client_total_seconds"])
                / float(accelerated[index]["client_total_seconds"])
                for index in selected
            )
        )
        accuracy_deltas.append(
            sum(
                int(bool(accelerated[index]["expected_match"]))
                - int(bool(baseline[index]["expected_match"]))
                for index in selected
            )
            / count
        )
    return {
        "aggregate_total_speedup": _interval(speedups),
        "median_paired_total_speedup": _interval(paired_medians),
        "expected_match_rate_delta": _interval(accuracy_deltas),
    }


def _interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    return [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_median(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else statistics.median(present)


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
    output_tokens = _answer_tokens(output)
    compact_output = normalize_answer(output)
    for answer in expected:
        expected_tokens = _answer_tokens(answer)
        if not expected_tokens:
            continue
        width = len(expected_tokens)
        if any(
            output_tokens[index : index + width] == expected_tokens
            for index in range(len(output_tokens) - width + 1)
        ):
            return True
        if normalize_answer(answer) == compact_output:
            return True
    return False


def _answer_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


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
