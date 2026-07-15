from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_METADATA = {
    "qwen3-vl:2b": {
        "family": "Qwen3-VL",
        "variant_label": "2B",
        "official_total_parameters": "2B",
        "cache_strategy": "full visual prompt-state reuse",
    },
    "qwen3-vl:4b": {
        "family": "Qwen3-VL",
        "variant_label": "4B",
        "official_total_parameters": "4B",
        "cache_strategy": "full visual prompt-state reuse",
    },
    "qwen3-vl:8b": {
        "family": "Qwen3-VL",
        "variant_label": "8B",
        "official_total_parameters": "9B",
        "cache_strategy": "full visual prompt-state reuse",
    },
    "qwen3.5:0.8b": {
        "family": "Qwen3.5",
        "variant_label": "0.8B",
        "official_total_parameters": "0.9B",
        "cache_strategy": "projected features plus hybrid-state checkpoint",
    },
    "qwen3.5:4b": {
        "family": "Qwen3.5",
        "variant_label": "4B",
        "official_total_parameters": "5B",
        "cache_strategy": "projected features plus hybrid-state checkpoint",
    },
    "qwen3.5:9b": {
        "family": "Qwen3.5",
        "variant_label": "9B",
        "official_total_parameters": "10B",
        "cache_strategy": "projected features plus hybrid-state checkpoint",
    },
}

SOURCES = {
    "qwen3_vl": "https://huggingface.co/collections/Qwen/qwen3-vl",
    "qwen3_5": "https://huggingface.co/collections/Qwen/qwen35",
    "qwen3_6": "https://huggingface.co/collections/Qwen/qwen36",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MachBoost vision benchmarks.")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = [load_artifact(path) for path in args.artifacts]
    validate_common_setup(artifacts)
    models = [summarize_model(artifact) for artifact in artifacts]
    models.sort(key=lambda row: (row["family"], size_key(row["variant_label"])))

    first = artifacts[0]
    image_path = Path(first["image"])
    all_rows = [row for artifact in artifacts for row in artifact["rows"]]
    cached_rows = [row for row in all_rows if row["mode"] == "cached"]
    reusable_rows = [row for row in cached_rows if row["prompt_cache_prefix_tokens"] > 0]
    no_prefix_rows = [row for row in cached_rows if row["prompt_cache_prefix_tokens"] == 0]
    matrix = {
        "schema_version": "machboost.vision_matrix.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "machine": first["machine"],
            "packages": first["packages"],
            "image": str(image_path),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_bytes": first["image_bytes"],
            "prompts": 4,
            "repeats": first["repeats"],
            "pairs_per_model": first["repeats"] * 4,
            "max_tokens": first["max_tokens"],
            "temperature": 0.0,
            "baseline": "same resident model with visual caches disabled",
            "accelerated": "same resident model with MachBoost visual caches enabled",
            "order": "baseline/cached order alternates by repeat",
            "warmup_policy": first["warmup_policy"],
        },
        "aggregate": {
            "measured_models": len(models),
            "paired_requests": len(cached_rows),
            "median_model_paired_speedup": statistics.median(
                model["median_paired_total_speedup"] for model in models
            ),
            "min_model_paired_speedup": min(
                model["median_paired_total_speedup"] for model in models
            ),
            "max_model_paired_speedup": max(
                model["median_paired_total_speedup"] for model in models
            ),
            "median_reusable_prefix_speedup": statistics.median(
                row["paired_total_speedup"] for row in reusable_rows
            ),
            "median_no_prefix_speedup": statistics.median(
                row["paired_total_speedup"] for row in no_prefix_rows
            ),
            "baseline_expected_match_rate": rate(all_rows, "baseline", "expected_match"),
            "cached_expected_match_rate": rate(all_rows, "cached", "expected_match"),
            "paired_output_equal_rate": sum(
                bool(row["paired_output_equal"]) for row in cached_rows
            )
            / len(cached_rows),
        },
        "models": models,
        "excluded": [
            {
                "family": "Qwen3.6",
                "status": "excluded",
                "reason": (
                    "No official release is under 10B total parameters; the official "
                    "collection lists 27B (28B total) and 35B-A3B (36B total, 3B active)."
                ),
                "source": SOURCES["qwen3_6"],
            }
        ],
        "sources": SOURCES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(matrix["aggregate"], indent=2))
    print(f"artifact: {args.output}")


def load_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "machboost.vision_benchmark.v1":
        raise ValueError(f"unsupported benchmark schema: {path}")
    if data.get("model") not in MODEL_METADATA:
        raise ValueError(f"missing model metadata for {data.get('model')!r}")
    return data


def validate_common_setup(artifacts: list[dict[str, Any]]) -> None:
    if not artifacts:
        raise ValueError("at least one artifact is required")
    expected = {
        key: artifacts[0][key]
        for key in ("image", "image_bytes", "repeats", "max_tokens", "machine")
    }
    for artifact in artifacts[1:]:
        actual = {key: artifact[key] for key in expected}
        if actual != expected:
            raise ValueError(f"benchmark setup mismatch for {artifact['model']}")


def summarize_model(artifact: dict[str, Any]) -> dict[str, Any]:
    metadata = MODEL_METADATA[artifact["model"]]
    cached = [row for row in artifact["rows"] if row["mode"] == "cached"]
    reusable = [row for row in cached if row["prompt_cache_prefix_tokens"] > 0]
    no_prefix = [row for row in cached if row["prompt_cache_prefix_tokens"] == 0]
    summary = artifact["summary"]
    return {
        "model": artifact["model"],
        "resolved_model": artifact["resolved_model"],
        **metadata,
        "baseline_median_total_seconds": summary["baseline_median_total_seconds"],
        "cached_median_total_seconds": summary["cached_median_total_seconds"],
        "median_paired_total_speedup": summary["median_paired_total_speedup"],
        "median_ttft_speedup": summary["median_ttft_speedup"],
        "paired_speedup_min": min(row["paired_total_speedup"] for row in cached),
        "paired_speedup_max": max(row["paired_total_speedup"] for row in cached),
        "median_reusable_prefix_speedup": statistics.median(
            row["paired_total_speedup"] for row in reusable
        ),
        "median_no_prefix_speedup": optional_median(
            row["paired_total_speedup"] for row in no_prefix
        ),
        "baseline_expected_match_rate": summary["baseline_expected_match_rate"],
        "cached_expected_match_rate": summary["cached_expected_match_rate"],
        "paired_output_equal_rate": summary["paired_output_equal_rate"],
        "projected_feature_hit_rate": summary["cached_hit_rate"],
        "prompt_prefix_hit_rate": summary["cached_prompt_prefix_hit_rate"],
        "artifact_created_at": artifact["created_at"],
    }


def rate(rows: list[dict[str, Any]], mode: str, key: str) -> float:
    selected = [row for row in rows if row["mode"] == mode]
    return sum(bool(row[key]) for row in selected) / len(selected)


def optional_median(values: Any) -> float | None:
    collected = list(values)
    return statistics.median(collected) if collected else None


def size_key(label: str) -> float:
    return float(label.removesuffix("B"))


if __name__ == "__main__":
    main()
