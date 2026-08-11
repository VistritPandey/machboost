from __future__ import annotations

from types import SimpleNamespace
import unittest

from machboost.latency import LATENCY_SCHEMA, benchmark_chat_latency


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


class FakeMachBoostClient:
    def __init__(self, trace=None) -> None:
        self.messages = []
        self.options = []
        self.trace = trace

    def load(self, model, *, options, keep_alive, warmup=False):
        return {
            "load_duration_seconds": 1.25,
            "warmup_duration_seconds": 0.5 if warmup else 0.0,
            "instance": {
                "model": "mlx-community/example",
                "backend": options.get("backend", "mlx"),
            },
        }

    def chat(self, model, messages, *, options, keep_alive, stream):
        self.messages.append(messages)
        self.options.append(options)
        if self.trace is not None:
            self.trace.append("machboost")
        return iter(
            [
                {"message": {"content": "hello"}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "total_duration": 500_000_000,
                    "load_duration": 0,
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 200_000_000,
                    "machboost": {
                        "backend": "mlx",
                        "time_to_first_token_seconds": 0.1,
                    },
                },
            ]
        )


class FakeOllamaAdapter:
    model = "example:latest"

    def __init__(self, trace=None) -> None:
        self.messages = []
        self.options = []
        self.think_values = []
        self.trace = trace

    def chat(self, messages, *, options, keep_alive, stream, think=None):
        self.messages.append(messages)
        self.options.append(options)
        self.think_values.append(think)
        if self.trace is not None:
            self.trace.append("ollama")
        yield SimpleNamespace(content="hello", done=False, raw={})
        yield SimpleNamespace(
            content="",
            done=True,
            raw={
                "total_duration": 800_000_000,
                "load_duration": 0,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 200_000_000,
                "eval_count": 4,
                "eval_duration": 400_000_000,
            },
        )


class ChatLatencyTests(unittest.TestCase):
    def test_compares_warm_streaming_latency_with_unique_prompts(self) -> None:
        trace = []
        machboost = FakeMachBoostClient(trace)
        ollama = FakeOllamaAdapter(trace)

        artifact = benchmark_chat_latency(
            "example",
            prompt="hey",
            system="Be concise.",
            runs=2,
            warmups=1,
            max_tokens=8,
            machboost_client=machboost,
            ollama_adapter=ollama,
            clock=StepClock(),
        )

        self.assertEqual(artifact["schema_version"], LATENCY_SCHEMA)
        self.assertEqual(artifact["engines"]["machboost"]["summary"]["runs"], 2)
        self.assertEqual(artifact["engines"]["ollama"]["summary"]["runs"], 2)
        self.assertEqual(
            artifact["engines"]["machboost"]["summary"]["median_tokens_per_second"],
            20.0,
        )
        self.assertEqual(
            artifact["engines"]["ollama"]["summary"]["median_tokens_per_second"],
            10.0,
        )
        self.assertEqual(
            artifact["engines"]["machboost"]["compile_warmup_seconds"],
            0.5,
        )
        self.assertTrue(artifact["comparison"]["median_output_equal"])
        self.assertEqual(len({row[0]["content"] for row in machboost.messages}), 3)
        self.assertEqual(
            trace,
            [
                "machboost",
                "ollama",
                "ollama",
                "machboost",
                "machboost",
                "ollama",
            ],
        )
        self.assertEqual(
            artifact["config"]["execution_order"],
            "alternating_by_round",
        )
        self.assertEqual(ollama.think_values, [False, False, False])

    def test_labels_same_ollama_mlx_engine_as_gateway_overhead(self) -> None:
        machboost = FakeMachBoostClient()
        ollama = FakeOllamaAdapter()
        artifact = benchmark_chat_latency(
            "muse-glimmer:30b-mlx",
            prompt="Write a short response.",
            system="Be concise.",
            runs=1,
            warmups=0,
            max_tokens=16,
            backend="ollama-mlx",
            machboost_client=machboost,
            ollama_adapter=ollama,
            draft_num_predict=15,
            clock=StepClock(),
        )

        self.assertEqual(
            artifact["config"]["comparison_kind"],
            "same_engine_gateway_overhead",
        )
        self.assertIsNotNone(
            artifact["comparison"]["machboost_gateway_overhead_percent"]
        )
        self.assertIn("same installed Ollama MLX model", " ".join(artifact["notes"]))
        self.assertEqual(artifact["config"]["draft_num_predict"], 15)
        self.assertEqual(machboost.options[0]["draft_num_predict"], 15)
        self.assertEqual(ollama.options[0]["draft_num_predict"], 15)

    def test_rejects_empty_measurement_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "runs must be at least 1"):
            benchmark_chat_latency(
                "example",
                prompt="hey",
                system="Be concise.",
                runs=0,
                engine="ollama",
            )


if __name__ == "__main__":
    unittest.main()
