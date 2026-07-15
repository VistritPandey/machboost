from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_vision_matrix.py"
SPEC = importlib.util.spec_from_file_location("summarize_vision_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)

ARTIFACTS = (
    "vision_cache_qwen3vl_2b_20260714.json",
    "vision_cache_qwen3vl_4b_20260714.json",
    "vision_cache_qwen3vl_8b_20260714.json",
    "vision_cache_qwen35_08b_20260714.json",
    "vision_cache_qwen35_4b_20260714.json",
    "vision_cache_qwen35_9b_20260714.json",
)


class VisionMatrixTests(unittest.TestCase):
    def test_committed_artifacts_share_one_setup_and_pass_quality_gate(self):
        artifacts = [
            MATRIX.load_artifact(ROOT / "results" / name) for name in ARTIFACTS
        ]

        MATRIX.validate_common_setup(artifacts)
        summaries = [MATRIX.summarize_model(artifact) for artifact in artifacts]

        self.assertEqual(len(summaries), 6)
        self.assertTrue(
            all(row["baseline_expected_match_rate"] == 1.0 for row in summaries)
        )
        self.assertTrue(
            all(row["cached_expected_match_rate"] == 1.0 for row in summaries)
        )
        self.assertTrue(all(row["median_paired_total_speedup"] > 2.0 for row in summaries))

    def test_parameter_labels_do_not_treat_quantized_size_as_model_size(self):
        qwen35 = MATRIX.MODEL_METADATA["qwen3.5:9b"]
        qwen3vl = MATRIX.MODEL_METADATA["qwen3-vl:8b"]

        self.assertEqual(qwen35["official_total_parameters"], "10B")
        self.assertEqual(qwen3vl["official_total_parameters"], "9B")
        self.assertIn("qwen36", MATRIX.SOURCES["qwen3_6"])


if __name__ == "__main__":
    unittest.main()
