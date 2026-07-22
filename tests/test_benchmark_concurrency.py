from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_concurrency.py"
SPEC = importlib.util.spec_from_file_location("benchmark_concurrency", SCRIPT)
benchmark_concurrency = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_concurrency)


class ConcurrencyBenchmarkTests(unittest.TestCase):
    def test_summary_reports_throughput_latency_queue_and_overload(self) -> None:
        rows = [
            {
                "ok": True,
                "wall_seconds": 0.5,
                "queue_wait_seconds": 0.0,
                "time_to_first_token_seconds": 0.1,
                "eval_count": 20,
                "replica": 0,
                "output_sha256": "a",
            },
            {
                "ok": True,
                "wall_seconds": 1.0,
                "queue_wait_seconds": 0.25,
                "time_to_first_token_seconds": 0.4,
                "eval_count": 40,
                "replica": 1,
                "output_sha256": "b",
            },
            {
                "ok": False,
                "wall_seconds": 0.01,
                "error_code": "queue_full",
            },
        ]

        summary = benchmark_concurrency.summarize(rows, [2.0])

        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["overloaded_requests"], 1)
        self.assertEqual(summary["requests_per_second"], 1.0)
        self.assertEqual(summary["aggregate_tokens_per_second"], 30.0)
        self.assertEqual(summary["median_latency_seconds"], 0.75)
        self.assertEqual(summary["replicas_used"], [0, 1])

    def test_percentile_interpolates_and_handles_empty_input(self) -> None:
        self.assertIsNone(benchmark_concurrency.percentile([], 0.95))
        self.assertEqual(benchmark_concurrency.percentile([1.0], 0.95), 1.0)
        self.assertAlmostEqual(
            benchmark_concurrency.percentile([0.0, 1.0], 0.95),
            0.95,
        )


if __name__ == "__main__":
    unittest.main()
