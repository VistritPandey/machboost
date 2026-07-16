from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machboost import TemporalVideoSampler, ensure_server
from machboost.vision_auto import VISION_TOKEN_REQUEST_MODES
try:
    from scripts.benchmark_cold_vision import (
        Sample,
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
        Sample,
        machine_data,
        normalize_answer,
        package_versions,
        run_request,
        summarize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare uniform video frames with temporal-change frame selection."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", action="append", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--change-threshold", type=float, default=0.08)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--vision-tokens",
        choices=VISION_TOKEN_REQUEST_MODES,
        default="off",
        help="Optional post-fusion policy for the temporal candidate only.",
    )
    parser.add_argument("--vision-token-ratio", type=float, default=0.35)
    parser.add_argument("--vision-token-layer", type=int)
    parser.add_argument("--vision-token-bucket", type=int)
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    video = args.video.expanduser().resolve()
    sampler = TemporalVideoSampler()

    ingest = sampler.sample_uniform(video, fps=args.fps, max_frames=args.max_frames)
    uniform = sampler.sample_uniform(video, fps=args.fps, max_frames=args.max_frames)
    temporal = sampler.sample(
        video,
        fps=args.fps,
        change_threshold=args.change_threshold,
        max_frames=args.max_frames,
    )
    digest = file_digest(video)
    baseline_sample = Sample(
        dataset="video",
        index=0,
        image=uniform.images[0],
        image_digest=digest,
        question=args.question,
        answers=tuple(args.answer),
        images=uniform.images,
    )
    temporal_sample = Sample(
        dataset="video",
        index=0,
        image=temporal.images[0],
        image_digest=digest,
        question=args.question,
        answers=tuple(args.answer),
        images=temporal.images,
    )

    client, server_started = ensure_server(args.endpoint, timeout=min(30.0, args.timeout))
    client.timeout = args.timeout
    client.stop(args.model)
    loaded = client.load(
        args.model,
        options={"vision_cache_size": 4},
        keep_alive="forever",
    )

    warmups = [
        _run_baseline(client, args.model, baseline_sample, args.max_tokens),
        _run_temporal(client, args.model, temporal_sample, args),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        order = ("baseline", "accelerated") if repeat % 2 == 0 else (
            "accelerated",
            "baseline",
        )
        pair: dict[str, dict[str, Any]] = {}
        for mode in order:
            row = (
                _run_baseline(client, args.model, baseline_sample, args.max_tokens)
                if mode == "baseline"
                else _run_temporal(client, args.model, temporal_sample, args)
            )
            row["repeat"] = repeat
            pair[mode] = row
            rows.append(row)
        add_pair_metrics(pair["baseline"], pair["accelerated"])

    summary = summarize(rows)
    summary["uniform_frames"] = uniform.selected_frames
    summary["temporal_frames"] = temporal.selected_frames
    summary["frame_reduction_rate"] = (
        0.0
        if uniform.selected_frames == 0
        else 1.0 - temporal.selected_frames / uniform.selected_frames
    )
    summary["uniform_preprocessing_seconds"] = uniform.elapsed_seconds
    summary["temporal_preprocessing_seconds"] = temporal.elapsed_seconds
    summary["amortized_end_to_end_speedup"] = end_to_end_speedup(
        rows,
        baseline_preprocessing_seconds=uniform.elapsed_seconds,
        accelerated_preprocessing_seconds=temporal.elapsed_seconds,
    )

    output = args.output or Path(
        "results/local/"
        f"video_frames_{args.model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "machboost.video_frame_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "paired uniform-frame versus temporal-change-frame VLM requests with caches disabled",
        "model": args.model,
        "resolved_model": loaded["instance"]["model"],
        "backend": loaded["instance"]["backend"],
        "video": str(video),
        "video_digest": digest,
        "question": args.question,
        "answers": args.answer,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "vision_tokens": args.vision_tokens,
        "vision_token_ratio": args.vision_token_ratio,
        "vision_token_layer": args.vision_token_layer,
        "vision_token_bucket": args.vision_token_bucket,
        "server_started": server_started,
        "load_duration_seconds": loaded["load_duration_seconds"],
        "shared_ingest_seconds": ingest.elapsed_seconds,
        "uniform_selection": uniform.to_dict(),
        "temporal_selection": temporal.to_dict(),
        "order_policy": "alternating uniform-first and temporal-first pairs",
        "machine": machine_data(),
        "packages": package_versions(),
        "summary": summary,
        "warmups": warmups,
        "rows": rows,
        "resident_models": client.ps(),
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"artifact: {output}")


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


def _run_temporal(client, model: str, sample: Sample, args) -> dict[str, Any]:
    return run_request(
        client,
        model,
        sample,
        mode="accelerated",
        cold_mode="off",
        max_tokens=args.max_tokens,
        vision_max_edge=None,
        vision_token_mode=args.vision_tokens,
        vision_token_ratio=args.vision_token_ratio,
        vision_token_layer=args.vision_token_layer,
        vision_token_bucket=args.vision_token_bucket,
    )


def add_pair_metrics(baseline: dict[str, Any], accelerated: dict[str, Any]) -> None:
    speedup = float(baseline["client_total_seconds"]) / float(
        accelerated["client_total_seconds"]
    )
    literal_equal = baseline["output"] == accelerated["output"]
    normalized_equal = normalize_answer(baseline["output"]) == normalize_answer(
        accelerated["output"]
    )
    for row in (baseline, accelerated):
        row["paired_total_speedup"] = speedup
        row["paired_literal_output_equal"] = literal_equal
        row["paired_normalized_output_equal"] = normalized_equal


def end_to_end_speedup(
    rows: list[dict[str, Any]],
    *,
    baseline_preprocessing_seconds: float,
    accelerated_preprocessing_seconds: float,
) -> float:
    baseline = [row for row in rows if row["mode"] == "baseline"]
    accelerated = [row for row in rows if row["mode"] == "accelerated"]
    if not baseline or len(baseline) != len(accelerated):
        raise ValueError("end-to-end summary requires paired rows")
    baseline_total = baseline_preprocessing_seconds + sum(
        float(row["client_total_seconds"]) for row in baseline
    )
    accelerated_total = accelerated_preprocessing_seconds + sum(
        float(row["client_total_seconds"]) for row in accelerated
    )
    return baseline_total / accelerated_total


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
