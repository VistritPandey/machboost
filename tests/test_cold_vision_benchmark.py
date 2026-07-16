from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_cold_vision import (
    Sample,
    answer_matches,
    answers_for,
    image_digest,
    images_for,
    paired_bootstrap_intervals,
    question_for,
    run_request,
    summarize,
    verify_uncertain_request,
)


class ColdVisionBenchmarkTests(unittest.TestCase):
    def test_answer_matching_accepts_any_reference(self) -> None:
        self.assertTrue(answer_matches("Dakota Digital", ("dakota", "other")))
        self.assertTrue(answer_matches("$42,700", ("42700",)))
        self.assertFalse(answer_matches("READY", ("ATLAS",)))
        self.assertFalse(answer_matches("The light is visible", ("l", "no")))
        self.assertTrue(answer_matches("GleamLight / Philippe Molitor", ("philippe molitor",)))

    def test_summary_reports_unique_cold_pairs_and_quality(self) -> None:
        rows = []
        for index, dataset in enumerate(("chartqa", "textvqa")):
            common = {
                "dataset": dataset,
                "image_digest": f"digest-{index}",
                "expected_match": True,
                "visual_cache_hit": False,
                "prompt_cache_prefix_tokens": 0,
                "paired_total_speedup": 4.0,
                "paired_literal_output_equal": index == 0,
                "paired_normalized_output_equal": True,
            }
            rows.append(
                {
                    **common,
                    "mode": "baseline",
                    "client_total_seconds": 4.0,
                    "client_ttft_seconds": 3.8,
                    "prompt_tokens": 800,
                    "cold_vision": {"enabled": False},
                }
            )
            rows.append(
                {
                    **common,
                    "mode": "accelerated",
                    "client_total_seconds": 1.0,
                    "client_ttft_seconds": 0.9,
                    "prompt_tokens": 200,
                    "cold_vision": {
                        "enabled": True,
                        "target_max_edge": 512 if index else 672,
                    },
                    "post_fusion_vision": {
                        "enabled": True,
                        "actual_visual_retention_ratio": 0.35,
                    },
                }
            )

        summary = summarize(rows)

        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["unique_images"], 2)
        self.assertEqual(summary["median_paired_total_speedup"], 4.0)
        self.assertEqual(summary["aggregate_total_speedup"], 4.0)
        self.assertEqual(summary["median_prompt_token_reduction_rate"], 0.75)
        self.assertEqual(summary["baseline_expected_match_rate"], 1.0)
        self.assertEqual(summary["accelerated_expected_match_rate"], 1.0)
        self.assertEqual(summary["paired_normalized_output_equal_rate"], 1.0)
        self.assertEqual(summary["selected_max_edges"], [512, 672])
        self.assertEqual(summary["cache_hit_count"], 0)
        self.assertEqual(summary["verification_fallback_rate"], 0.0)
        self.assertEqual(summary["first_pass_expected_match_rate"], 1.0)
        self.assertEqual(summary["post_fusion_enabled_rate"], 1.0)
        self.assertEqual(summary["median_actual_visual_retention_ratio"], 0.35)
        self.assertEqual(summary["datasets"]["chartqa"]["pairs"], 1)

    def test_confidence_gate_replays_uncertain_first_pass(self) -> None:
        sample = Sample("textvqa", 1, "/tmp/image.png", "digest", "Question?", ("yes",))
        first_pass = {
            "output": "no",
            "expected_match": False,
            "client_total_seconds": 0.5,
            "client_ttft_seconds": 0.4,
            "server_total_seconds": 0.45,
            "generated_tokens": 2,
            "prompt_tokens": 200,
            "mean_token_logprob": -0.25,
            "minimum_token_logprob": -0.5,
            "cold_vision": {"enabled": True, "target_max_edge": 512},
        }
        fallback = {
            **first_pass,
            "mode": "fallback",
            "output": "yes",
            "expected_match": True,
            "client_total_seconds": 1.5,
            "client_ttft_seconds": 1.4,
            "server_total_seconds": 1.45,
            "generated_tokens": 1,
            "prompt_tokens": 600,
            "cold_vision": {"enabled": False},
        }

        with patch(
            "scripts.benchmark_cold_vision.run_request", return_value=fallback
        ) as replay:
            result = verify_uncertain_request(
                object(),
                "qwen3-vl:4b",
                sample,
                first_pass=first_pass,
                confidence_threshold=-0.04,
                max_tokens=16,
            )

        replay.assert_called_once()
        self.assertEqual(result["output"], "yes")
        self.assertEqual(result["client_total_seconds"], 2.0)
        self.assertEqual(result["prompt_tokens"], 800)
        self.assertEqual(result["cold_vision"]["target_max_edge"], 512)
        self.assertTrue(result["verification"]["fallback"])

    def test_confidence_gate_accepts_confident_first_pass(self) -> None:
        first_pass = {
            "output": "yes",
            "expected_match": True,
            "mean_token_logprob": -0.01,
            "minimum_token_logprob": -0.02,
            "client_total_seconds": 0.5,
            "client_ttft_seconds": 0.4,
        }
        sample = Sample("textvqa", 1, "/tmp/image.png", "digest", "Question?", ("yes",))

        result = verify_uncertain_request(
            object(),
            "qwen3-vl:4b",
            sample,
            first_pass=first_pass,
            confidence_threshold=-0.04,
            max_tokens=16,
        )

        self.assertEqual(result["output"], "yes")
        self.assertFalse(result["verification"]["fallback"])

    def test_summary_rejects_unpaired_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "equally sized"):
            summarize(
                [
                    {
                        "mode": "baseline",
                    }
                ]
            )

    def test_mmmu_sample_preserves_images_options_and_answer_text(self) -> None:
        first = object()
        second = object()
        row = {
            "question": "Compare <image 1> with <image 2>.",
            "options": "['alpha', 'beta', 'gamma']",
            "answer": "B",
            "image_1": first,
            "image_2": second,
            **{f"image_{index}": None for index in range(3, 8)},
        }

        self.assertEqual(images_for("mmmu", row), (first, second))
        self.assertEqual(answers_for("mmmu", row), ("B", "beta"))
        question = question_for("mmmu", row)
        self.assertIn("A. alpha", question)
        self.assertIn("B. beta", question)
        self.assertTrue(question.endswith("Return only the option letter."))

    def test_paired_bootstrap_reports_speed_and_accuracy_intervals(self) -> None:
        baseline = [
            {"client_total_seconds": 4.0, "expected_match": True},
            {"client_total_seconds": 6.0, "expected_match": False},
        ]
        accelerated = [
            {"client_total_seconds": 2.0, "expected_match": True},
            {"client_total_seconds": 3.0, "expected_match": True},
        ]

        intervals = paired_bootstrap_intervals(
            baseline,
            accelerated,
            draws=100,
            seed=7,
        )

        self.assertEqual(intervals["aggregate_total_speedup"], [2.0, 2.0])
        self.assertEqual(intervals["median_paired_total_speedup"], [2.0, 2.0])
        self.assertEqual(intervals["expected_match_rate_delta"], [0.0, 1.0])

    def test_request_forwards_shape_aware_policy_controls(self) -> None:
        class Client:
            options = None

            def generate(self, model, prompt, *, images, options, keep_alive, stream):
                self.options = options
                return iter(
                    [
                        {"response": "yes", "done": False},
                        {
                            "response": "",
                            "done": True,
                            "eval_count": 1,
                            "total_duration": 1_000_000,
                            "machboost": {"stats": {"prompt_tokens": 128}},
                        },
                    ]
                )

        client = Client()
        sample = Sample("docvqa", 1, "/tmp/image.png", "digest", "Read?", ("yes",))
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp) / "calibration.json"
            run_request(
                client,
                "qwen3-vl:8b",
                sample,
                mode="accelerated",
                cold_mode="off",
                max_tokens=8,
                vision_max_edge=None,
                vision_token_mode="auto",
                vision_token_ratio=0.5,
                vision_token_layer=6,
                vision_token_bucket=32,
                vision_calibration=calibration,
            )

        self.assertEqual(client.options["vision_tokens"], "auto")
        self.assertEqual(client.options["vision_token_layer"], 6)
        self.assertEqual(client.options["vision_token_bucket"], 32)
        self.assertEqual(client.options["vision_calibration"], str(calibration.resolve()))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is optional")
    def test_image_digest_changes_with_pixels(self) -> None:
        from PIL import Image

        first = Image.new("RGB", (4, 4), "white")
        second = Image.new("RGB", (4, 4), "black")

        self.assertEqual(image_digest(first), image_digest(first.copy()))
        self.assertNotEqual(image_digest(first), image_digest(second))


if __name__ == "__main__":
    unittest.main()
