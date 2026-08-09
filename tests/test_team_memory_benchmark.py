import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_team_memory import run_benchmark


class TeamMemoryBenchmarkTests(unittest.TestCase):
    def test_five_developer_scenarios_pass_with_expected_savings(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_benchmark(Path(temporary) / "team.sqlite3")

        self.assertEqual(result["schema"], "machboost.team-memory-benchmark.v1")
        self.assertEqual(len(result["developers"]), 5)
        self.assertEqual(result["scenario_count"], 5)
        self.assertTrue(result["passed"])
        self.assertTrue(all(item["passed"] for item in result["scenarios"]))
        self.assertEqual(result["savings"]["exact_cache_hits"], 5)
        self.assertEqual(result["savings"]["avoided_prompt_tokens"], 12_000)
        self.assertEqual(result["savings"]["avoided_completion_tokens"], 600)
        self.assertAlmostEqual(result["savings"]["avoided_cost_usd"], 0.06)
        self.assertIn("does not claim model decode speedup", result["note"])


if __name__ == "__main__":
    unittest.main()
