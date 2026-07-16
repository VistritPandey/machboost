from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from machboost.vision_auto import CALIBRATION_SCHEMA
from scripts.benchmark_cold_vision import normalize_answer, summarize


ABLATION_SCHEMA = "machboost.vision_token_ablation.v1"


@dataclass(frozen=True)
class Candidate:
    workload: str
    profile: str
    mode: str
    retain_ratio: float
    prune_after_layer: int
    token_bucket: int
    summary: dict[str, Any]
    source: str

    @property
    def eligible_speedup(self) -> float:
        interval = self.summary.get("aggregate_total_speedup_ci95") or (0.0, 0.0)
        return float(interval[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deployment-safe visual-token policies from ablation artifacts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision-calibration.json"))
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--max-quality-drop", type=float, default=0.02)
    parser.add_argument("--min-output-agreement", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = calibrate(
        args.artifacts,
        min_pairs=args.min_pairs,
        min_speedup=args.min_speedup,
        max_quality_drop=args.max_quality_drop,
        min_output_agreement=args.min_output_agreement,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(calibration["workloads"], indent=2))
    print(f"calibration: {args.output}")


def calibrate(
    artifact_paths: Sequence[Path],
    *,
    min_pairs: int = 10,
    min_speedup: float = 1.05,
    max_quality_drop: float = 0.02,
    min_output_agreement: float = 0.80,
) -> dict[str, Any]:
    if min_pairs < 1:
        raise ValueError("minimum pairs must be at least 1")
    candidates: list[Candidate] = []
    sources = []
    for path in artifact_paths:
        resolved = Path(path).expanduser().resolve()
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
        candidates.extend(candidates_from_artifact(artifact, source=str(resolved)))
        sources.append(str(resolved))

    workloads: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    workload_names = sorted({candidate.workload for candidate in candidates})
    for workload in workload_names:
        workload_candidates = [
            candidate for candidate in candidates if candidate.workload == workload
        ]
        eligible = [
            candidate
            for candidate in workload_candidates
            if candidate.mode != "random"
            and int(candidate.summary["pairs"]) >= min_pairs
            and candidate.eligible_speedup >= min_speedup
            and float(candidate.summary["expected_match_rate_delta"])
            >= -max_quality_drop
            and float(candidate.summary["paired_normalized_output_equal_rate"])
            >= min_output_agreement
        ]
        selected = max(
            eligible,
            key=lambda candidate: (
                float(candidate.summary["aggregate_total_speedup"]),
                candidate.eligible_speedup,
            ),
            default=None,
        )
        if selected is None:
            workloads[workload] = {
                "mode": "off",
                "enabled": False,
                "retain_ratio": 1.0,
                "prune_after_layer": 3,
                "token_bucket": 0,
                "reason": "no measured candidate passed the calibration gates",
            }
        else:
            workloads[workload] = {
                "mode": selected.mode,
                "enabled": True,
                "retain_ratio": selected.retain_ratio,
                "prune_after_layer": selected.prune_after_layer,
                "token_bucket": selected.token_bucket,
                "reason": (
                    f"offline calibration selected {selected.profile} from "
                    f"{selected.summary['pairs']} paired samples"
                ),
            }
        evidence[workload] = {
            "selected_profile": None if selected is None else selected.profile,
            "candidates": [candidate_evidence(candidate) for candidate in workload_candidates],
        }

    return {
        "schema": CALIBRATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "min_pairs": min_pairs,
            "min_speedup_ci95_lower": min_speedup,
            "max_expected_match_rate_drop": max_quality_drop,
            "min_normalized_output_agreement": min_output_agreement,
            "random_control_eligible": False,
        },
        "workloads": workloads,
        "evidence": evidence,
        "sources": sources,
    }


def candidates_from_artifact(
    artifact: Mapping[str, Any],
    *,
    source: str,
) -> list[Candidate]:
    if artifact.get("schema_version") != ABLATION_SCHEMA:
        raise ValueError(f"unsupported vision ablation schema in {source}")
    baseline_by_key = {
        _row_key(row): row for row in artifact.get("baseline_rows") or ()
    }
    profiles = {
        str(profile["slug"]): profile for profile in artifact.get("profiles") or ()
    }
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for profile, rows in (artifact.get("profile_rows") or {}).items():
        if profile not in profiles:
            continue
        for row in rows:
            baseline = baseline_by_key.get(_row_key(row))
            if baseline is None:
                continue
            workload = str(
                ((row.get("post_fusion_vision") or {}).get("policy") or {}).get(
                    "workload", "general"
                )
            )
            grouped.setdefault((workload, profile), []).append((baseline, row))

    candidates = []
    for (workload, profile_slug), pairs in grouped.items():
        profile = profiles[profile_slug]
        baseline_rows = []
        accelerated_rows = []
        for baseline, accelerated in pairs:
            baseline_copy = dict(baseline)
            accelerated_copy = dict(accelerated)
            _ensure_pair_metrics(baseline_copy, accelerated_copy)
            baseline_rows.append(baseline_copy)
            accelerated_rows.append(accelerated_copy)
        summary = summarize([*baseline_rows, *accelerated_rows])
        candidates.append(
            Candidate(
                workload=workload,
                profile=profile_slug,
                mode=str(profile["mode"]),
                retain_ratio=float(profile.get("retain_ratio", 0.35)),
                prune_after_layer=int(profile.get("prune_after_layer") or 3),
                token_bucket=int(profile.get("token_bucket") or 0),
                summary=summary,
                source=source,
            )
        )
    return candidates


def candidate_evidence(candidate: Candidate) -> dict[str, Any]:
    summary = candidate.summary
    return {
        "profile": candidate.profile,
        "mode": candidate.mode,
        "pairs": summary["pairs"],
        "aggregate_total_speedup": summary["aggregate_total_speedup"],
        "aggregate_total_speedup_ci95": summary["aggregate_total_speedup_ci95"],
        "expected_match_rate_delta": summary["expected_match_rate_delta"],
        "expected_match_rate_delta_ci95": summary["expected_match_rate_delta_ci95"],
        "paired_normalized_output_equal_rate": summary[
            "paired_normalized_output_equal_rate"
        ],
        "source": candidate.source,
    }


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset"), row.get("index"), row.get("image_digest"))


def _ensure_pair_metrics(
    baseline: dict[str, Any], accelerated: dict[str, Any]
) -> None:
    accelerated["paired_total_speedup"] = (
        float(baseline["client_total_seconds"])
        / float(accelerated["client_total_seconds"])
    )
    accelerated["paired_literal_output_equal"] = (
        baseline.get("output") == accelerated.get("output")
    )
    accelerated["paired_normalized_output_equal"] = normalize_answer(
        str(baseline.get("output") or "")
    ) == normalize_answer(str(accelerated.get("output") or ""))
    baseline.setdefault("paired_total_speedup", 1.0)
    baseline.setdefault("paired_literal_output_equal", True)
    baseline.setdefault("paired_normalized_output_equal", True)


if __name__ == "__main__":
    main()
