from __future__ import annotations

from types import SimpleNamespace
import unittest

from machboost.context_bench import (
    CONTEXT_BENCH_SCHEMA,
    benchmark_context_acceleration,
    context_fingerprint,
)


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeAccelerator:
    def __init__(self, clock: ManualClock, *, mismatch_run: int | None = None) -> None:
        self.clock = clock
        self.boost_enabled = True
        self.mismatch_run = mismatch_run
        self.boosted_calls = 0
        self.trace = []

    def generate_result(self, prompt, *, max_tokens):
        mode = "machboost" if self.boost_enabled else "native"
        self.trace.append(mode)
        tokens = tuple(range(min(8, max_tokens)))
        if self.boost_enabled:
            self.boosted_calls += 1
            self.clock.advance(0.5)
            if self.boosted_calls == self.mismatch_run:
                tokens = tokens[:-1] + (99,)
            stats = SimpleNamespace(
                target_calls=2,
                verify_calls=1,
                accepted_draft_tokens=6,
                accepted_draft_spans=1,
            )
        else:
            self.clock.advance(1.0)
            stats = SimpleNamespace(
                target_calls=8,
                verify_calls=0,
                accepted_draft_tokens=0,
                accepted_draft_spans=0,
            )
        return SimpleNamespace(tokens=tokens, text="same output", stats=stats)


class ContextBenchmarkTests(unittest.TestCase):
    def test_records_balanced_exact_same_model_speedup(self) -> None:
        clock = ManualClock()
        accelerator = FakeAccelerator(clock)

        artifact = benchmark_context_acceleration(
            accelerator,
            "complete this",
            model="example-model",
            backend="mlx",
            context_fingerprint=context_fingerprint(["complete this output"]),
            context_chars=20,
            runs=4,
            warmups=0,
            max_tokens=8,
            clock=clock,
        )

        self.assertEqual(artifact["schema_version"], CONTEXT_BENCH_SCHEMA)
        self.assertTrue(artifact["summary"]["valid"])
        self.assertEqual(artifact["summary"]["output_match_rate"], 1.0)
        self.assertEqual(artifact["summary"]["algorithm_engaged_rate"], 1.0)
        self.assertEqual(artifact["summary"]["median_speedup"], 2.0)
        self.assertEqual(artifact["summary"]["median_accepted_draft_tokens"], 6.0)
        self.assertEqual(artifact["summary"]["median_target_call_reduction"], 0.75)
        self.assertEqual(
            [row["order"] for row in artifact["rows"]],
            [
                ["native", "machboost"],
                ["machboost", "native"],
                ["native", "machboost"],
                ["machboost", "native"],
            ],
        )

    def test_invalidates_speedup_when_one_token_differs(self) -> None:
        clock = ManualClock()
        accelerator = FakeAccelerator(clock, mismatch_run=2)

        artifact = benchmark_context_acceleration(
            accelerator,
            "complete this",
            model="example-model",
            backend="mlx",
            context_fingerprint=context_fingerprint(["complete this output"]),
            context_chars=20,
            runs=2,
            warmups=0,
            max_tokens=8,
            clock=clock,
        )

        self.assertFalse(artifact["summary"]["valid"])
        self.assertEqual(artifact["summary"]["output_match_rate"], 0.5)
        self.assertIsNone(artifact["summary"]["median_speedup"])
        self.assertIsNone(artifact["rows"][1]["speedup"])
        self.assertEqual(artifact["rows"][1]["first_mismatch_token_index"], 7)
        self.assertEqual(artifact["summary"]["diagnostic_median_speedup"], 2.0)

    def test_rejects_unbalanced_measured_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "even number"):
            benchmark_context_acceleration(
                FakeAccelerator(ManualClock()),
                "prompt",
                model="example-model",
                backend="mlx",
                context_fingerprint="abc",
                context_chars=3,
                runs=3,
            )


if __name__ == "__main__":
    unittest.main()
