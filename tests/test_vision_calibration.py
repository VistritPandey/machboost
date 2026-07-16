from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from machboost.vision_auto import CALIBRATION_SCHEMA
from scripts.calibrate_vision_tokens import calibrate


class VisionCalibrationTests(unittest.TestCase):
    def test_calibration_selects_fast_quality_preserving_policy(self) -> None:
        artifact = fixture_artifact()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ablation.json"
            source.write_text(json.dumps(artifact), encoding="utf-8")

            result = calibrate(
                [source],
                min_pairs=2,
                min_speedup=1.5,
                max_quality_drop=0.0,
                min_output_agreement=1.0,
            )

        self.assertEqual(result["schema"], CALIBRATION_SCHEMA)
        selected = result["workloads"]["document-text"]
        self.assertTrue(selected["enabled"])
        self.assertEqual(selected["mode"], "adaptive")
        self.assertEqual(selected["prune_after_layer"], 6)
        self.assertEqual(
            result["evidence"]["document-text"]["selected_profile"],
            "adaptive-r0.5-l6-b32",
        )

    def test_random_control_is_never_deployed(self) -> None:
        artifact = fixture_artifact(include_adaptive=False)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ablation.json"
            source.write_text(json.dumps(artifact), encoding="utf-8")

            result = calibrate(
                [source],
                min_pairs=2,
                min_speedup=1.1,
                max_quality_drop=0.0,
                min_output_agreement=1.0,
            )

        self.assertFalse(result["workloads"]["document-text"]["enabled"])
        self.assertEqual(result["workloads"]["document-text"]["mode"], "off")

    def test_incomplete_checkpoint_is_rejected(self) -> None:
        artifact = fixture_artifact()
        artifact["status"] = "failed"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ablation.json"
            source.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "is not complete: failed"):
                calibrate([source], min_pairs=2)


def fixture_artifact(*, include_adaptive: bool = True) -> dict:
    baseline_rows = [fixture_row(index, "baseline", seconds=4.0) for index in range(2)]
    profiles = [
        {
            "slug": "random-r0.35-l3-b0",
            "name": "random:0.35:3:0",
            "mode": "random",
            "retain_ratio": 0.35,
            "prune_after_layer": 3,
            "token_bucket": 0,
        }
    ]
    profile_rows = {
        "random-r0.35-l3-b0": [
            fixture_row(index, "accelerated", seconds=1.0) for index in range(2)
        ]
    }
    if include_adaptive:
        profiles.append(
            {
                "slug": "adaptive-r0.5-l6-b32",
                "name": "adaptive:0.5:6:32",
                "mode": "adaptive",
                "retain_ratio": 0.5,
                "prune_after_layer": 6,
                "token_bucket": 32,
            }
        )
        profile_rows["adaptive-r0.5-l6-b32"] = [
            fixture_row(index, "accelerated", seconds=2.0) for index in range(2)
        ]
    return {
        "schema_version": "machboost.vision_token_ablation.v1",
        "profiles": profiles,
        "baseline_rows": baseline_rows,
        "profile_rows": profile_rows,
    }


def fixture_row(index: int, mode: str, *, seconds: float) -> dict:
    return {
        "dataset": "docvqa",
        "index": index,
        "image_digest": f"image-{index}",
        "mode": mode,
        "output": "42",
        "expected_match": True,
        "client_total_seconds": seconds,
        "client_ttft_seconds": seconds * 0.9,
        "prompt_tokens": 512 if mode == "baseline" else 256,
        "visual_cache_hit": False,
        "prompt_cache_prefix_tokens": 0,
        "cold_vision": {},
        "post_fusion_vision": {
            "enabled": mode == "accelerated",
            "actual_visual_retention_ratio": 0.5,
            "policy": {"workload": "document-text"},
        },
    }


if __name__ == "__main__":
    unittest.main()
