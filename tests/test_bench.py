import unittest

from machboost.bench import (
    BenchmarkCase,
    GatePolicy,
    benchmark,
    benchmark_cases,
    decide_gate,
    summarize_results,
)


class ScriptedService:
    def __init__(self, prompt, completion):
        self.prompt_len = len(self.encode(prompt))
        self.completion = tuple(self.encode(completion))
        self.forward_calls = 0
        self.reset_count = 0

    def encode(self, text):
        return tuple(ord(char) for char in text)

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)

    def reset_cache(self):
        self.reset_count += 1

    def next_token(self, prefix_tokens):
        self.forward_calls += 1
        offset = max(0, len(prefix_tokens) - self.prompt_len)
        if offset >= len(self.completion):
            return None
        return self.completion[offset]

    def verify(self, prefix_tokens, candidate_tokens):
        self.forward_calls += 1
        offset = max(0, len(prefix_tokens) - self.prompt_len)
        accepted = 0
        for token in candidate_tokens:
            target_pos = offset + accepted
            if target_pos >= len(self.completion) or token != self.completion[target_pos]:
                break
            accepted += 1
        if accepted == len(candidate_tokens):
            return accepted, None
        residual_pos = offset + accepted
        residual = self.completion[residual_pos] if residual_pos < len(self.completion) else None
        return accepted, residual


class BenchTests(unittest.TestCase):
    def test_benchmark_enables_high_overlap_exact_run(self):
        prompt = "Continue: "
        completion = "alpha beta gamma"
        service = ScriptedService(prompt, completion)

        result = benchmark(
            service,
            prompt,
            context=completion,
            max_tokens=len(completion),
            ngram=2,
            max_draft_tokens=8,
            gate_policy=GatePolicy(min_speedup=0.0, min_acceptance_rate=0.5),
        )

        self.assertTrue(result.output_match)
        self.assertTrue(result.decision.enabled)
        self.assertEqual(result.baseline.text, completion)
        self.assertEqual(result.boosted.text, completion)
        self.assertGreater(result.acceptance_rate, 0.5)
        self.assertGreater(result.forward_reduction_rate, 0.0)
        self.assertIn("estimated_speedup", result.to_dict()["boosted"]["stats"])

    def test_benchmark_disables_low_overlap_run(self):
        prompt = "Continue: "
        completion = "abcdef"
        service = ScriptedService(prompt, completion)

        result = benchmark(
            service,
            prompt,
            context="unrelated context",
            max_tokens=len(completion),
            ngram=3,
            max_draft_tokens=4,
            gate_policy=GatePolicy(min_speedup=1.0, min_acceptance_rate=0.5),
        )

        self.assertTrue(result.output_match)
        self.assertFalse(result.decision.enabled)
        self.assertEqual(result.acceptance_rate, 0.0)

    def test_decide_gate_requires_exactness_by_default(self):
        decision = decide_gate(
            output_match=False,
            speedup=2.0,
            acceptance_rate=1.0,
            generated_tokens=8,
        )

        self.assertFalse(decision.enabled)
        self.assertIn("did not match", decision.reason)

    def test_benchmark_cases_and_summary(self):
        prompt = "Continue: "
        completion = "alpha beta"
        service = ScriptedService(prompt, completion)
        cases = [
            BenchmarkCase("positive", prompt, context=completion, max_tokens=len(completion)),
            BenchmarkCase("negative", prompt, context="zzz", max_tokens=len(completion)),
        ]

        results = benchmark_cases(
            service,
            cases,
            ngram=2,
            max_draft_tokens=8,
            gate_policy=GatePolicy(min_speedup=1.0, min_acceptance_rate=0.5),
        )
        summary = summarize_results(results)

        self.assertEqual(len(results), 2)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["output_match_rate"], 1.0)
        self.assertGreater(summary["median_speedup"], 0.0)


if __name__ == "__main__":
    unittest.main()
