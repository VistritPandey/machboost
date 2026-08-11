import base64
import threading
import unittest

from machboost.adapters.ollama import OllamaChatChunk
from machboost.adapters.ollama_mlx import (
    OllamaMLXAccelerator,
    OllamaMLXCancelled,
    inject_reasoning_strength,
    normalize_reasoning,
)


class FakeAdapter:
    def __init__(self, chunks):
        self.model = "muse-glimmer:30b-mlx"
        self.chunks = chunks
        self.calls = []
        self.unloads = 0

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return iter(self.chunks)

    def unload(self):
        self.unloads += 1


def chunk(*, content="", thinking="", tool_calls=(), done=False, raw=None):
    return OllamaChatChunk(
        model="muse-glimmer:30b-mlx",
        role="assistant",
        content=content,
        thinking=thinking,
        tool_calls=tuple(tool_calls),
        done=done,
        raw=raw or {},
    )


class OllamaMLXAcceleratorTests(unittest.TestCase):
    def test_streams_text_and_preserves_reasoning_tools_and_metrics(self):
        call = {
            "id": "call_7",
            "function": {
                "index": 0,
                "name": "lookup_score",
                "arguments": {"team": "Argentina"},
            },
        }
        final = {
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "prompt_eval_duration": 2_000_000_000,
            "eval_count": 20,
            "eval_duration": 1_000_000_000,
            "load_duration": 50_000_000,
            "total_duration": 3_100_000_000,
        }
        adapter = FakeAdapter(
            [
                chunk(thinking="Inspecting image."),
                chunk(content="Argentina "),
                chunk(content="won.", tool_calls=[call], done=True, raw=final),
            ]
        )
        accelerator = OllamaMLXAccelerator(adapter)
        emitted = []
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_score",
                    "parameters": {"type": "object"},
                },
            }
        ]

        text, stats = accelerator.generate_chat(
            [
                {
                    "role": "user",
                    "content": "Who won?",
                    "images": ["data:image/png;base64,aW1hZ2U="],
                }
            ],
            max_tokens=128,
            on_text=emitted.append,
            temperature=1.0,
            enable_thinking="high",
            tools=tools,
        )

        self.assertEqual(text, "Argentina won.")
        self.assertEqual(emitted, ["Argentina ", "won."])
        self.assertEqual(stats.thinking, "Inspecting image.")
        self.assertEqual(stats.tool_calls, (call,))
        self.assertEqual(stats.prompt_tokens_per_second, 50.0)
        self.assertEqual(stats.generation_tokens_per_second, 20.0)
        self.assertTrue(stats.native_speculative_decoding)
        messages, kwargs = adapter.calls[0]
        self.assertTrue(kwargs["think"])
        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(kwargs["options"]["num_predict"], 128)
        self.assertEqual(messages[0]["content"], "Reasoning strength: high")
        self.assertEqual(base64.b64decode(messages[1]["images"][0]), b"image")

    def test_context_is_stable_system_prefix(self):
        adapter = FakeAdapter([chunk(done=True, raw={"done": True})])
        accelerator = OllamaMLXAccelerator(
            adapter,
            context_texts=["# file: service.py\nPORT = 8080"],
        )

        accelerator.generate_chat(
            [{"role": "user", "content": "Which port?"}],
            max_tokens=8,
        )

        messages, _ = adapter.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("PORT = 8080", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "Which port?")

    def test_cancellation_interrupts_reasoning_stream(self):
        adapter = FakeAdapter([chunk(thinking="hidden")])
        accelerator = OllamaMLXAccelerator(adapter)
        cancelled = threading.Event()
        cancelled.set()

        with self.assertRaises(OllamaMLXCancelled):
            accelerator.generate_chat(
                [{"role": "user", "content": "Think"}],
                max_tokens=8,
                enable_thinking=True,
                cancel_event=cancelled,
            )

    def test_close_unloads_once(self):
        adapter = FakeAdapter([])
        accelerator = OllamaMLXAccelerator(adapter)

        accelerator.close()
        accelerator.close()

        self.assertEqual(adapter.unloads, 1)

    def test_reasoning_levels_are_validated(self):
        self.assertEqual(normalize_reasoning(False), (False, None))
        self.assertEqual(normalize_reasoning("xhigh"), (True, "xhigh"))
        with self.assertRaisesRegex(ValueError, "reasoning strength"):
            normalize_reasoning(True, "extreme")

    def test_reasoning_instruction_prepends_existing_system_message(self):
        messages = inject_reasoning_strength(
            [{"role": "system", "content": "Be concise."}],
            "low",
        )

        self.assertEqual(
            messages,
            [{"role": "system", "content": "Reasoning strength: low\nBe concise."}],
        )


if __name__ == "__main__":
    unittest.main()
