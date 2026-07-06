import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hf_prompt_lookup_compare.py"


def load_script():
    spec = importlib.util.spec_from_file_location("hf_prompt_lookup_compare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HFPromptLookupCompareTests(unittest.TestCase):
    def test_self_test_passes(self):
        module = load_script()

        result = module.run_self_test()

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema_version"], "machboost.hf_prompt_lookup_compare.v1.self_test")

    def test_summarize_methods_groups_by_method(self):
        module = load_script()
        rows = [
            module.MethodRow(
                fixture="a",
                workflow="unit",
                expectation="positive",
                nonce="mb-a",
                method="serial",
                source_mode="prompt",
                output_match=True,
                elapsed_ms=100.0,
                tokens_per_second=10.0,
                speedup_vs_serial=1.0,
                model_forwards=8,
                forward_reduction_percent=0.0,
                accepted_draft_tokens=0,
                generated_tokens=8,
                raw_generated_tokens=8,
                generated_token_ids=[1, 2],
                output_preview="ok",
                note="baseline",
            ),
            module.MethodRow(
                fixture="a",
                workflow="unit",
                expectation="positive",
                nonce="mb-a",
                method="boost",
                source_mode="context",
                output_match=True,
                elapsed_ms=50.0,
                tokens_per_second=20.0,
                speedup_vs_serial=2.0,
                model_forwards=4,
                forward_reduction_percent=50.0,
                accepted_draft_tokens=6,
                generated_tokens=8,
                raw_generated_tokens=8,
                generated_token_ids=[1, 2],
                output_preview="ok",
                note="boosted",
            ),
        ]

        summaries = {row["method"]: row for row in module.summarize_methods(rows)}

        self.assertEqual(summaries["serial"]["output_match_rate"], 1.0)
        self.assertEqual(summaries["boost"]["median_speedup_vs_serial"], 2.0)
        self.assertEqual(summaries["boost"]["median_forward_reduction_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
