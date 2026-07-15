from __future__ import annotations

import importlib.util
import unittest

from scripts.benchmark_cold_vision import answer_matches, image_digest, summarize


class ColdVisionBenchmarkTests(unittest.TestCase):
    def test_answer_matching_accepts_any_reference(self) -> None:
        self.assertTrue(answer_matches("Dakota Digital", ("dakota", "other")))
        self.assertTrue(answer_matches("$42,700", ("42700",)))
        self.assertFalse(answer_matches("READY", ("ATLAS",)))

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
                }
            )

        summary = summarize(rows)

        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["unique_images"], 2)
        self.assertEqual(summary["median_paired_total_speedup"], 4.0)
        self.assertEqual(summary["median_prompt_token_reduction_rate"], 0.75)
        self.assertEqual(summary["baseline_expected_match_rate"], 1.0)
        self.assertEqual(summary["accelerated_expected_match_rate"], 1.0)
        self.assertEqual(summary["paired_normalized_output_equal_rate"], 1.0)
        self.assertEqual(summary["selected_max_edges"], [512, 672])
        self.assertEqual(summary["cache_hit_count"], 0)
        self.assertEqual(summary["datasets"]["chartqa"]["pairs"], 1)

    def test_summary_rejects_unpaired_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "equally sized"):
            summarize(
                [
                    {
                        "mode": "baseline",
                    }
                ]
            )

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is optional")
    def test_image_digest_changes_with_pixels(self) -> None:
        from PIL import Image

        first = Image.new("RGB", (4, 4), "white")
        second = Image.new("RGB", (4, 4), "black")

        self.assertEqual(image_digest(first), image_digest(first.copy()))
        self.assertNotEqual(image_digest(first), image_digest(second))


if __name__ == "__main__":
    unittest.main()
