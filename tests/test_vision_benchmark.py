from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_vision_cache import answer_matches, create_fixture, summarize


class VisionBenchmarkTests(unittest.TestCase):
    def test_answer_matching_ignores_case_and_punctuation(self):
        self.assertTrue(answer_matches("The budget is $42,700.", "$42,700"))
        self.assertTrue(answer_matches("blue square", "BLUE SQUARE"))
        self.assertFalse(answer_matches("READY", "ATLAS"))

    def test_summary_reports_speed_and_quality(self):
        rows = []
        for repeat in range(2):
            rows.extend(
                [
                    {
                        "mode": "baseline",
                        "client_total_seconds": 3.0 + repeat,
                        "client_ttft_seconds": 2.5 + repeat,
                        "prompt_tokens_per_second": 100.0,
                        "paired_output_equal": True,
                        "paired_total_speedup": 3.0,
                        "expected_match": True,
                        "visual_cache_hit": False,
                        "prompt_cache_prefix_tokens": 0,
                    },
                    {
                        "mode": "cached",
                        "client_total_seconds": 1.0 + repeat,
                        "client_ttft_seconds": 0.5 + repeat,
                        "prompt_tokens_per_second": 400.0,
                        "paired_output_equal": True,
                        "paired_total_speedup": 3.0,
                        "expected_match": True,
                        "visual_cache_hit": True,
                        "prompt_cache_prefix_tokens": 900,
                    },
                ]
            )

        summary = summarize(rows)

        self.assertEqual(summary["rows_per_mode"], 2)
        self.assertGreater(summary["median_total_speedup"], 2.0)
        self.assertEqual(summary["median_paired_total_speedup"], 3.0)
        self.assertGreaterEqual(summary["median_ttft_speedup"], 3.0)
        self.assertEqual(summary["paired_output_equal_rate"], 1.0)
        self.assertEqual(summary["cached_hit_rate"], 1.0)
        self.assertEqual(summary["cached_prompt_prefix_hit_rate"], 1.0)
        self.assertEqual(summary["cached_median_prompt_prefix_tokens"], 900)

    @unittest.skipUnless(
        importlib.util.find_spec("PIL"), "Pillow is an optional vision dependency"
    )
    def test_generated_fixture_is_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            create_fixture(path)

            self.assertGreater(path.stat().st_size, 1_000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
