from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "python"
sys.path.insert(0, str(EXAMPLES))

from benchmark_context_workload import build_result  # noqa: E402
from context_example_utils import (  # noqa: E402
    Passage,
    read_text_paths,
    retrieve_passages,
    split_passages,
)


class ContextExampleUtilsTests(unittest.TestCase):
    def test_reads_text_and_excludes_completion_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = root / "policy.md"
            included.write_text("Deployments require two approvals.", encoding="utf-8")
            excluded = root / "current.py"
            excluded.write_text("secret_future_code()", encoding="utf-8")
            binary = root / "weights.bin"
            binary.write_bytes(b"\x00\xff")

            passages = read_text_paths([str(root)], exclude=[excluded])

        self.assertEqual([passage.source for passage in passages], [str(included.resolve())])
        self.assertEqual(passages[0].text, "Deployments require two approvals.")

    def test_retrieval_prioritizes_question_terms(self) -> None:
        documents = [
            Passage("travel.md", "The office is beside the train station."),
            Passage("release.md", "Production deployment requires two reviewer approvals."),
        ]

        retrieved = retrieve_passages(
            "What approvals are required for production deployment?",
            split_passages(documents),
            limit=1,
        )

        self.assertEqual(retrieved[0].source, "release.md")
        self.assertGreater(retrieved[0].score, 0)


class ContextWorkloadReportTests(unittest.TestCase):
    def test_invalidates_speedup_when_any_pair_mismatches(self) -> None:
        artifacts = [
            {
                "rows": [
                    _row(output_match=True, speedup=2.0, accepted=8),
                    _row(output_match=False, speedup=3.0, accepted=8),
                ]
            }
        ]

        result = build_result(artifacts, load_seconds=1.25)

        self.assertFalse(result["summary"]["valid"])
        self.assertEqual(result["summary"]["output_match_rate"], 0.5)
        self.assertIsNone(result["summary"]["valid_median_speedup"])
        self.assertEqual(result["summary"]["diagnostic_median_speedup"], 2.5)

    def test_reports_zero_engagement_for_native_fallback(self) -> None:
        result = build_result(
            [{"rows": [_row(output_match=True, speedup=1.0, accepted=0)]}],
            load_seconds=0.5,
        )

        self.assertTrue(result["summary"]["valid"])
        self.assertEqual(result["summary"]["algorithm_engaged_rate"], 0.0)
        self.assertEqual(result["summary"]["valid_median_speedup"], 1.0)


def _row(*, output_match: bool, speedup: float, accepted: int) -> dict:
    return {
        "output_match": output_match,
        "diagnostic_speedup": speedup,
        "accepted_draft_tokens": accepted,
        "native": {"wall_seconds": 2.0},
        "machboost": {"wall_seconds": 1.0},
    }


if __name__ == "__main__":
    unittest.main()
