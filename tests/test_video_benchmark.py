from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_video_frames import (
    add_pair_metrics,
    end_to_end_speedup,
    file_digest,
)


class VideoBenchmarkTests(unittest.TestCase):
    def test_pair_metrics_compare_uniform_and_temporal_outputs(self) -> None:
        baseline = {"client_total_seconds": 6.0, "output": "Red light"}
        accelerated = {"client_total_seconds": 2.0, "output": "red-light"}

        add_pair_metrics(baseline, accelerated)

        self.assertEqual(accelerated["paired_total_speedup"], 3.0)
        self.assertFalse(accelerated["paired_literal_output_equal"])
        self.assertTrue(accelerated["paired_normalized_output_equal"])

    def test_end_to_end_speedup_includes_preprocessing(self) -> None:
        rows = [
            {"mode": "baseline", "client_total_seconds": 4.0},
            {"mode": "accelerated", "client_total_seconds": 2.0},
            {"mode": "baseline", "client_total_seconds": 4.0},
            {"mode": "accelerated", "client_total_seconds": 2.0},
        ]

        speedup = end_to_end_speedup(
            rows,
            baseline_preprocessing_seconds=0.1,
            accelerated_preprocessing_seconds=0.2,
        )

        self.assertAlmostEqual(speedup, 8.1 / 4.2)

    def test_video_digest_streams_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(b"video fixture")
            first = file_digest(path)
            path.write_bytes(b"changed fixture")
            second = file_digest(path)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
