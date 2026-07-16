from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from machboost.vision_auto import (
    CALIBRATION_SCHEMA,
    VisionImageSignals,
    choose_vision_token_policy,
    classify_vision_workload,
    inspect_vision_images,
    load_vision_calibration,
)


class VisionAutoPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simple = VisionImageSignals(
            count=1,
            max_edge=1200,
            entropy=0.50,
            edge_density=0.12,
        )

    def test_classifies_prompt_and_multi_image_workloads(self) -> None:
        self.assertEqual(classify_vision_workload("Describe the scene."), "general")
        self.assertEqual(classify_vision_workload("Read the invoice total."), "document-text")
        self.assertEqual(classify_vision_workload("Which bar chart is highest?"), "chart")
        self.assertEqual(classify_vision_workload("What is left of the chair?"), "spatial")
        self.assertEqual(
            classify_vision_workload("Compare them.", image_count=2),
            "multi-image",
        )

    def test_auto_uses_workload_specific_budget_and_depth(self) -> None:
        general = choose_vision_token_policy(
            "Describe the scene.",
            ["image.png"],
            mode="auto",
            image_signals=self.simple,
        )
        document = choose_vision_token_policy(
            "Read the invoice total.",
            ["image.png"],
            mode="auto",
            image_signals=self.simple,
        )

        self.assertEqual((general.retain_ratio, general.prune_after_layer), (0.35, 3))
        self.assertEqual((document.retain_ratio, document.prune_after_layer), (0.50, 6))
        self.assertEqual(general.token_bucket, 32)
        self.assertEqual(general.source, "builtin")

    def test_high_detail_signal_increases_budget(self) -> None:
        detailed = VisionImageSignals(
            count=1,
            max_edge=2400,
            entropy=0.90,
            edge_density=0.35,
        )

        decision = choose_vision_token_policy(
            "Read this document.",
            ["image.png"],
            mode="auto",
            image_signals=detailed,
        )

        self.assertEqual(decision.retain_ratio, 0.55)
        self.assertEqual(decision.prune_after_layer, 6)

    def test_calibration_overrides_builtin_profile(self) -> None:
        calibration = {
            "schema_version": CALIBRATION_SCHEMA,
            "workloads": {
                "document-text": {
                    "mode": "adaptive",
                    "retain_ratio": 0.62,
                    "prune_after_layer": 9,
                    "token_bucket": 64,
                    "reason": "measured document profile",
                }
            },
        }

        decision = choose_vision_token_policy(
            "Read the receipt.",
            ["image.png"],
            mode="auto",
            calibration=calibration,
            image_signals=self.simple,
        )

        self.assertEqual(decision.source, "calibration")
        self.assertEqual(decision.retain_ratio, 0.62)
        self.assertEqual(decision.prune_after_layer, 9)
        self.assertEqual(decision.token_bucket, 64)

    def test_manual_policy_preserves_explicit_controls(self) -> None:
        decision = choose_vision_token_policy(
            "Anything",
            ["image.png"],
            mode="merge",
            retain_ratio=0.25,
            prune_after_layer=4,
            token_bucket=0,
            image_signals=self.simple,
        )

        self.assertEqual(decision.mode, "merge")
        self.assertEqual(decision.retain_ratio, 0.25)
        self.assertEqual(decision.prune_after_layer, 4)
        self.assertEqual(decision.token_bucket, 0)
        self.assertEqual(decision.source, "request")

    def test_load_calibration_validates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": CALIBRATION_SCHEMA,
                        "workloads": {"default": {"retain_ratio": 0.5}},
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_vision_calibration(path)
            self.assertEqual(loaded["workloads"]["default"]["retain_ratio"], 0.5)

            path.write_text('{"schema_version":"unknown","workloads":{}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_vision_calibration(path)

    def test_inspect_images_reports_real_signals(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (20, 10), "white").save(path)

            signals = inspect_vision_images([str(path)])

        self.assertEqual(signals.count, 1)
        self.assertEqual(signals.max_edge, 20)
        self.assertGreaterEqual(signals.entropy, 0.0)
        self.assertGreaterEqual(signals.edge_density, 0.0)


if __name__ == "__main__":
    unittest.main()
