from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from machboost import ensure_server
from machboost.vision_auto import VISION_TOKEN_REQUEST_MODES
try:
    from scripts.benchmark_cold_vision import (
        DATASETS,
        Sample,
        load_public_samples,
        machine_data,
        normalize_answer,
        package_versions,
        run_request,
        summarize,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from benchmark_cold_vision import (
        DATASETS,
        Sample,
        load_public_samples,
        machine_data,
        normalize_answer,
        package_versions,
        run_request,
        summarize,
    )


DEFAULT_PROFILES = ",".join(
    (
        "random:0.35:3:0",
        "merge:0.35:3:0",
        "adaptive:0.35:3:0",
        "adaptive:0.35:3:32",
        "adaptive:0.50:6:32",
        "auto",
    )
)


@dataclass(frozen=True)
class VisionProfile:
    name: str
    mode: str
    retain_ratio: float = 0.35
    prune_after_layer: int | None = None
    token_bucket: int | None = None

    @property
    def slug(self) -> str:
        layer = "auto" if self.prune_after_layer is None else str(self.prune_after_layer)
        bucket = "auto" if self.token_bucket is None else str(self.token_bucket)
        return f"{self.mode}-r{self.retain_ratio:g}-l{layer}-b{bucket}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple post-fusion visual-token policies against one shared baseline."
    )
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument(
        "--datasets",
        default="chartqa,docvqa,mmmu,textvqa",
        help="Comma-separated public datasets: chartqa,docvqa,mmmu,textvqa.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument(
        "--profiles",
        default=DEFAULT_PROFILES,
        help="Comma-separated mode[:ratio[:layer[:bucket]]] candidates.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--work-dir", type=Path, default=Path("results/tmp/vision_tokens"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_profiles(value: str) -> tuple[VisionProfile, ...]:
    profiles = tuple(
        parse_profile(item.strip()) for item in value.split(",") if item.strip()
    )
    if not profiles:
        raise ValueError("at least one visual-token profile is required")
    slugs = [profile.slug for profile in profiles]
    if len(slugs) != len(set(slugs)):
        raise ValueError("visual-token profiles must be unique")
    return profiles


def parse_profile(value: str) -> VisionProfile:
    parts = value.split(":")
    if len(parts) > 4:
        raise ValueError(f"invalid visual-token profile: {value}")
    mode = parts[0].strip().lower()
    if mode not in VISION_TOKEN_REQUEST_MODES or mode == "off":
        raise ValueError(
            "candidate mode must be one of: "
            + ", ".join(mode for mode in VISION_TOKEN_REQUEST_MODES if mode != "off")
        )
    ratio = float(parts[1]) if len(parts) > 1 and parts[1] else 0.35
    layer = int(parts[2]) if len(parts) > 2 and parts[2] else None
    bucket = int(parts[3]) if len(parts) > 3 and parts[3] else None
    if not 0.1 <= ratio <= 1.0:
        raise ValueError("visual-token profile ratio must be between 0.1 and 1.0")
    if layer is not None and layer < 1:
        raise ValueError("visual-token profile layer must be at least 1")
    if bucket is not None and bucket < 0:
        raise ValueError("visual-token profile bucket cannot be negative")
    return VisionProfile(
        name=value,
        mode=mode,
        retain_ratio=ratio,
        prune_after_layer=layer,
        token_bucket=bucket,
    )


def main() -> None:
    args = parse_args()
    dataset_names = tuple(
        name.strip().lower() for name in args.datasets.split(",") if name.strip()
    )
    unknown = sorted(set(dataset_names) - set(DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {', '.join(unknown)}")
    if not dataset_names:
        raise ValueError("at least one dataset is required")
    if args.samples_per_dataset < 1:
        raise ValueError("samples per dataset must be at least 1")
    profiles = parse_profiles(args.profiles)
    output = args.output or Path(
        "results/local/"
        f"vision_tokens_{args.model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

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

    warmup_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    profile_rows: dict[str, list[dict[str, Any]]] = {
        profile.slug: [] for profile in profiles
    }
    _write_checkpoint(
        output,
        args=args,
        dataset_names=dataset_names,
        profiles=profiles,
        baseline_rows=baseline_rows,
        profile_rows=profile_rows,
        warmup_rows=warmup_rows,
    )
    for sample in warmups:
        _print_progress("warmup", sample, "baseline")
        warmup_rows.append(_run_baseline(client, args.model, sample, args.max_tokens))
        for profile in profiles:
            _print_progress("warmup", sample, profile.slug)
            warmup_rows.append(
                _run_profile(client, args.model, sample, profile, args.max_tokens)
            )
        _write_checkpoint(
            output,
            args=args,
            dataset_names=dataset_names,
            profiles=profiles,
            baseline_rows=baseline_rows,
            profile_rows=profile_rows,
            warmup_rows=warmup_rows,
        )

    methods: tuple[VisionProfile | None, ...] = (None, *profiles)
    for sample_index, sample in enumerate(samples):
        rotated = methods[sample_index % len(methods) :] + methods[: sample_index % len(methods)]
        baseline: dict[str, Any] | None = None
        candidates: dict[str, dict[str, Any]] = {}
        for profile in rotated:
            if profile is None:
                _print_progress("evaluate", sample, "baseline")
                baseline = _run_baseline(client, args.model, sample, args.max_tokens)
            else:
                _print_progress("evaluate", sample, profile.slug)
                candidates[profile.slug] = _run_profile(
                    client,
                    args.model,
                    sample,
                    profile,
                    args.max_tokens,
                )
        if baseline is None:
            raise RuntimeError("baseline request was not executed")
        baseline_rows.append(baseline)
        for profile in profiles:
            candidate = candidates[profile.slug]
            _add_pair_metrics(baseline, candidate)
            profile_rows[profile.slug].append(candidate)
        _write_checkpoint(
            output,
            args=args,
            dataset_names=dataset_names,
            profiles=profiles,
            baseline_rows=baseline_rows,
            profile_rows=profile_rows,
            warmup_rows=warmup_rows,
        )

    summaries = {
        profile.slug: summarize(
            [dict(row) for row in baseline_rows]
            + [dict(row) for row in profile_rows[profile.slug]]
        )
        for profile in profiles
    }
    artifact = {
        "schema_version": "machboost.vision_token_ablation.v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "shared-baseline paired unique-image post-fusion visual-token ablation",
        "model": args.model,
        "resolved_model": loaded["instance"]["model"],
        "backend": loaded["instance"]["backend"],
        "datasets": list(dataset_names),
        "dataset_sources": {name: DATASETS[name].source for name in dataset_names},
        "samples_per_dataset": args.samples_per_dataset,
        "max_tokens": args.max_tokens,
        "profiles": [asdict(profile) | {"slug": profile.slug} for profile in profiles],
        "server_started": server_started,
        "load_duration_seconds": loaded["load_duration_seconds"],
        "warmup_policy": "one held-out unique image per dataset and method; excluded from metrics",
        "order_policy": "rotating baseline and candidate order for each sample",
        "machine": machine_data(),
        "packages": package_versions(),
        "summaries": summaries,
        "warmups": warmup_rows,
        "baseline_rows": baseline_rows,
        "profile_rows": profile_rows,
        "resident_models": client.ps(),
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"artifact: {output}")


def _print_progress(stage: str, sample: Sample, method: str) -> None:
    print(
        f"[{stage}] dataset={sample.dataset} index={sample.index} method={method}",
        flush=True,
    )


def _write_checkpoint(
    output: Path,
    *,
    args: argparse.Namespace,
    dataset_names: Sequence[str],
    profiles: Sequence[VisionProfile],
    baseline_rows: list[dict[str, Any]],
    profile_rows: dict[str, list[dict[str, Any]]],
    warmup_rows: list[dict[str, Any]],
) -> None:
    summaries = {
        profile.slug: summarize(
            [dict(row) for row in baseline_rows]
            + [dict(row) for row in profile_rows[profile.slug]]
        )
        for profile in profiles
        if baseline_rows and len(profile_rows[profile.slug]) == len(baseline_rows)
    }
    artifact = {
        "schema_version": "machboost.vision_token_ablation.v1",
        "status": "running",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "datasets": list(dataset_names),
        "dataset_sources": {name: DATASETS[name].source for name in dataset_names},
        "samples_per_dataset": args.samples_per_dataset,
        "max_tokens": args.max_tokens,
        "profiles": [asdict(profile) | {"slug": profile.slug} for profile in profiles],
        "completed_samples": len(baseline_rows),
        "summaries": summaries,
        "warmups": warmup_rows,
        "baseline_rows": baseline_rows,
        "profile_rows": profile_rows,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _run_baseline(client, model: str, sample: Sample, max_tokens: int) -> dict[str, Any]:
    return run_request(
        client,
        model,
        sample,
        mode="baseline",
        cold_mode="off",
        max_tokens=max_tokens,
        vision_max_edge=None,
        vision_token_mode="off",
    )


def _run_profile(
    client,
    model: str,
    sample: Sample,
    profile: VisionProfile,
    max_tokens: int,
) -> dict[str, Any]:
    row = run_request(
        client,
        model,
        sample,
        mode="accelerated",
        cold_mode="off",
        max_tokens=max_tokens,
        vision_max_edge=None,
        vision_token_mode=profile.mode,
        vision_token_ratio=profile.retain_ratio,
        vision_token_layer=profile.prune_after_layer,
        vision_token_bucket=profile.token_bucket,
    )
    row["profile"] = profile.slug
    return row


def _add_pair_metrics(baseline: dict[str, Any], accelerated: dict[str, Any]) -> None:
    accelerated["paired_total_speedup"] = (
        baseline["client_total_seconds"] / accelerated["client_total_seconds"]
    )
    accelerated["paired_literal_output_equal"] = baseline["output"] == accelerated["output"]
    accelerated["paired_normalized_output_equal"] = normalize_answer(
        baseline["output"]
    ) == normalize_answer(accelerated["output"])
    baseline.setdefault("paired_total_speedup", 1.0)
    baseline.setdefault("paired_literal_output_equal", True)
    baseline.setdefault("paired_normalized_output_equal", True)


if __name__ == "__main__":
    main()
