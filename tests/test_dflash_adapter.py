from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from machboost.adapters.dflash import DFlashAccelerator, _load_runtime_bundle_compat


class FakeDetokenizer:
    def reset(self):
        self.text = ""
        self.offset = 0

    def add_token(self, token):
        self.text += {1: "Hello", 2: " ", 3: "world", 4: "!"}[int(token)]

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment

    def finalize(self):
        return None


class FakeTokenizer:
    def __init__(self):
        self.detokenizer = FakeDetokenizer()
        self.detokenizer.reset()
        self.chat_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_messages = list(messages)
        self.chat_kwargs = dict(kwargs)
        return "<chat>"

    def encode(self, text, **kwargs):
        return [len(text), int(bool(kwargs.get("add_special_tokens")))]

    def decode(self, tokens, **kwargs):
        return ":".join(str(token) for token in tokens)


@dataclass
class TokenEvent:
    token_id: int


@dataclass
class SummaryEvent:
    elapsed_us: float = 500_000
    prompt_token_count: int = 20
    generation_tokens: int = 4
    accepted_from_draft: int = 3
    acceptance_ratio: float = 0.75
    cycles_completed: int = 2
    phase_timings_us: dict | None = None
    peak_memory_gb: float = 6.25
    tokens_per_cycle: float = 2.0

    def __post_init__(self):
        if self.phase_timings_us is None:
            self.phase_timings_us = {"prefill": 100_000}


class DFlashAdapterTests(unittest.TestCase):
    def setUp(self):
        tokenizer = FakeTokenizer()
        self.bundle = SimpleNamespace(
            target_model=object(),
            tokenizer=tokenizer,
            draft_model=object(),
            draft_backend=object(),
            target_ops=object(),
            resolved_model_ref="acme/target",
            resolved_draft_ref="acme/draft",
        )

        def stream_generate(**kwargs):
            self.stream_kwargs = kwargs
            yield from (TokenEvent(1), TokenEvent(2), TokenEvent(3), TokenEvent(4))
            yield SummaryEvent()

        self.accelerator = DFlashAccelerator(
            self.bundle,
            runtime_context=object(),
            stream_generate_fn=stream_generate,
            stop_token_ids_fn=lambda _: [99],
            token_event_type=TokenEvent,
            summary_event_type=SummaryEvent,
        )

    def test_streams_text_and_reports_decode_metrics(self):
        chunks = []

        text, stats = self.accelerator.generate(
            "prompt",
            max_tokens=4,
            on_text=chunks.append,
        )

        self.assertEqual(text, "Hello world!")
        self.assertEqual("".join(chunks), text)
        self.assertEqual(stats.generated_tokens, 4)
        self.assertEqual(stats.accepted_draft_tokens, 3)
        self.assertEqual(stats.target_calls, 2)
        self.assertEqual(stats.baseline_target_calls, 4)
        self.assertAlmostEqual(stats.prompt_tokens_per_second, 200.0)
        self.assertAlmostEqual(stats.generation_tokens_per_second, 10.0)
        self.assertEqual(self.stream_kwargs["stop_token_ids"], [99])
        self.assertFalse(self.stream_kwargs["use_chat_template"])

    def test_chat_applies_template_before_generation(self):
        text, _ = self.accelerator.generate_chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=4,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )

        self.assertEqual(text, "Hello world!")
        self.assertEqual(self.stream_kwargs["prompt"], "<chat>")
        self.assertEqual(
            self.bundle.tokenizer.chat_messages,
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(
            self.bundle.tokenizer.chat_kwargs["tools"][0]["function"]["name"],
            "lookup",
        )

    def test_rejects_sampling_instead_of_changing_semantics(self):
        with self.assertRaisesRegex(ValueError, "greedy decoding only"):
            self.accelerator.generate(
                "prompt",
                max_tokens=4,
                generation_options={"temperature": 0.5},
            )

    def test_rejects_unwired_repository_context(self):
        with self.assertRaisesRegex(ValueError, "not wired yet"):
            self.accelerator.generate("prompt", max_tokens=4, context=["repo"])

    def test_exposes_tokenizer_service_for_context_limits(self):
        self.assertEqual(self.accelerator.encode("abc"), (3, 0))
        self.assertEqual(self.accelerator.decode((3, 0)), "3:0")

    def test_rejects_stop_strings_instead_of_ignoring_them(self):
        with self.assertRaisesRegex(ValueError, "stop strings"):
            self.accelerator.generate(
                "prompt",
                max_tokens=4,
                stop_strings=["STOP"],
            )

    def test_resident_replicas_do_not_share_a_python_generation_lock(self):
        other = DFlashAccelerator(
            self.bundle,
            runtime_context=object(),
            stream_generate_fn=lambda **_: iter(()),
            stop_token_ids_fn=lambda _: [],
            token_event_type=TokenEvent,
            summary_event_type=SummaryEvent,
        )

        self.assertIsNot(
            self.accelerator._generation_lock,
            other._generation_lock,
        )

    def test_normalizes_nested_checkpoint_config_and_restores_runtime(self):
        class DraftArgs:
            @classmethod
            def from_dict(cls, params):
                return dict(params)

        original = DraftArgs.__dict__["from_dict"]

        def load_runtime_bundle(**kwargs):
            return DraftArgs.from_dict(
                {
                    "dflash_config": {"block_size": 16},
                    "rope_parameters": {"rope_theta": 10_000_000},
                    **kwargs,
                }
            )

        result = _load_runtime_bundle_compat(
            load_runtime_bundle,
            DraftArgs,
            model_ref="target",
        )

        self.assertEqual(result["block_size"], 16)
        self.assertEqual(result["rope_theta"], 10_000_000)
        self.assertIs(DraftArgs.__dict__["from_dict"], original)


if __name__ == "__main__":
    unittest.main()
