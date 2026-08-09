import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import benchmark_repository_reuse as benchmark


def result(*, wall, prefill, digest="same", rubric=3, cached=0):
    return {
        "wall_seconds": wall,
        "ttft_seconds": prefill,
        "prompt_tokens": 100,
        "prompt_eval_seconds": prefill,
        "completion_tokens": 20,
        "decode_seconds": 0.2,
        "cached_prompt_tokens": cached,
        "evaluated_prompt_tokens": 100 - cached,
        "output_sha256": digest,
        "output_chars": 80,
        "rubric_hits": rubric,
        "rubric_total": 4,
        "citations": None,
    }


class RepositoryReuseBenchmarkTests(unittest.TestCase):
    def test_workspace_metadata_returns_only_public_aggregate_fields(self):
        with patch.object(
            benchmark,
            "request_json",
            return_value={
                "workspaces": [
                    {
                        "id": "workspace-1",
                        "name": "Private repo",
                        "path": "/secret/path",
                        "revision": "abc123",
                        "file_count": 10,
                        "chunk_count": 20,
                        "total_bytes": 30,
                    }
                ]
            },
        ):
            metadata = benchmark.workspace_metadata(
                "http://localhost", "workspace-1", token="secret"
            )

        self.assertNotIn("path", metadata)
        self.assertEqual(metadata["revision"], "abc123")
        self.assertEqual(metadata["chunk_count"], 20)

    def test_pair_primes_a_separate_thread_and_preserves_output_equality(self):
        args = SimpleNamespace(
            primer="explain architecture",
            target="implement adjacent change",
            primer_tokens=16,
            max_tokens=32,
        )
        calls = []

        def fake_chat(_args, prompt, *, cache, namespace, max_tokens):
            calls.append((prompt, cache, namespace, max_tokens))
            if prompt == args.primer:
                return result(wall=0.2, prefill=0.1, digest="primer")
            if cache:
                return result(wall=1.0, prefill=0.25, cached=70)
            return result(wall=2.0, prefill=1.0)

        with patch.object(benchmark, "chat", side_effect=fake_chat):
            measured = benchmark.run_pair(
                args, round_number=1, baseline_first=True
            )

        self.assertEqual([call[1] for call in calls], [False, True, True])
        self.assertNotEqual(calls[0][2], calls[1][2])
        self.assertEqual(calls[1][2], calls[2][2])
        self.assertTrue(measured["output_equal"])
        self.assertEqual(measured["wall_speedup"], 2.0)
        self.assertEqual(measured["prefill_speedup"], 4.0)

    def test_summary_reports_medians_and_quality_counts(self):
        rounds = [
            {
                "output_equal": True,
                "wall_speedup": 2.0,
                "prefill_speedup": 4.0,
                "baseline": result(wall=2.0, prefill=1.0, rubric=3),
                "machboost": result(
                    wall=1.0, prefill=0.25, rubric=3, cached=70
                ),
            },
            {
                "output_equal": True,
                "wall_speedup": 1.5,
                "prefill_speedup": 3.0,
                "baseline": result(wall=3.0, prefill=1.5, rubric=4),
                "machboost": result(
                    wall=2.0, prefill=0.5, rubric=4, cached=60
                ),
            },
        ]

        summary = benchmark.summarize(rounds)

        self.assertEqual(summary["exact_output_pairs"], 2)
        self.assertEqual(summary["median_wall_speedup"], 1.75)
        self.assertEqual(summary["median_prefill_speedup"], 3.5)
        self.assertEqual(summary["median_cached_prompt_tokens"], 65)
        self.assertEqual(summary["rubric_total"], 4)


if __name__ == "__main__":
    unittest.main()
