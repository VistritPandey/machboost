from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, patch

from machboost.adapters.mlx_vlm import MLXVLMAccelerator, _resolution_scoped_images
from machboost.vision_auto import VisionImageSignals
from machboost.vision_policy import ColdVisionDecision


@dataclass
class FakeGenerationRow:
    text: str
    generation_tokens: int = 4
    prompt_tokens: int = 20
    prompt_tps: float = 80.0
    generation_tps: float = 40.0
    peak_memory: float = 1.5
    token: int | None = None
    logprobs: object | None = None


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeModel:
    config = {"model_type": "fake_vlm"}


class FakeTokenizer:
    def __init__(self, decoded: str) -> None:
        self.decoded = decoded

    def decode(self, token_ids, *, skip_special_tokens=False):
        return self.decoded


class FakeProcessor:
    def __init__(self, decoded: str) -> None:
        self.tokenizer = FakeTokenizer(decoded)


class FakeVisionStream:
    def __init__(self) -> None:
        self.calls = []
        self.encoder_calls = 0

    def __call__(self, model, processor, prompt, *, image=None, vision_cache=None, **kwargs):
        self.calls.append({"prompt": prompt, "image": image, "kwargs": kwargs})
        if image is not None:
            features = vision_cache.get(image) if vision_cache is not None else None
            if features is None:
                self.encoder_calls += 1
                if vision_cache is not None:
                    vision_cache.put(image, f"features-{self.encoder_calls}")
        yield FakeGenerationRow("blue ")
        yield FakeGenerationRow("square")


class FakePromptCacheState:
    def __init__(self, token_ids=None):
        self.token_ids = list(token_ids or ())

    def find_prefix_length(self, token_ids):
        for index, (cached, current) in enumerate(zip(self.token_ids, token_ids)):
            if cached != current:
                return index
        return min(len(self.token_ids), len(token_ids))


class FakeArray:
    def __init__(self, values):
        self.values = list(values)

    def flatten(self):
        return self

    def tolist(self):
        return list(self.values)


class MLXVLMAcceleratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image = self.root / "fixture.png"
        self.image.write_bytes(b"\x89PNG\r\n\x1a\nvision-fixture")
        self.stream = FakeVisionStream()
        self.templates = []

        def template(processor, config, prompt, **kwargs):
            self.templates.append((prompt, kwargs))
            return "templated prompt"

        self.accelerator = MLXVLMAccelerator(
            FakeModel(),
            object(),
            model_name="fake-vlm",
            asset_cache_dir=self.root / "assets",
            stream_generate_fn=self.stream,
            apply_chat_template_fn=template,
        )

    def tearDown(self):
        self.accelerator.close()
        self.directory.cleanup()

    def test_repeated_image_skips_second_encoder_call(self):
        message = {"role": "user", "content": "What shape?", "images": [str(self.image)]}

        first_text, first_stats = self.accelerator.generate_chat([message], max_tokens=8)
        second_text, second_stats = self.accelerator.generate_chat([message], max_tokens=8)

        self.assertEqual(first_text, "blue square")
        self.assertEqual(second_text, first_text)
        self.assertEqual(self.stream.encoder_calls, 1)
        self.assertTrue(first_stats.visual_cache_miss)
        self.assertFalse(first_stats.visual_cache_hit)
        self.assertTrue(second_stats.visual_cache_hit)
        self.assertFalse(second_stats.visual_cache_miss)
        self.assertEqual(second_stats.visual_cache_entries, 1)

    def test_disabled_cache_runs_encoder_for_each_request(self):
        message = {"role": "user", "content": "What shape?", "images": [str(self.image)]}

        _, first = self.accelerator.generate_chat([message], max_tokens=8, use_vision_cache=False)
        _, second = self.accelerator.generate_chat([message], max_tokens=8, use_vision_cache=False)

        self.assertEqual(self.stream.encoder_calls, 2)
        self.assertFalse(first.visual_cache_hit)
        self.assertFalse(first.visual_cache_miss)
        self.assertFalse(second.visual_cache_hit)
        self.assertEqual(self.accelerator.cache_info()["size"], 0)

    def test_streams_text_and_reports_backend_metrics(self):
        emitted = []
        text, stats = self.accelerator.generate(
            "Describe the image.",
            images=[str(self.image)],
            max_tokens=8,
            on_text=emitted.append,
        )

        self.assertEqual(text, "blue square")
        self.assertEqual(emitted, ["blue ", "square"])
        self.assertEqual(stats.backend, "mlx-vlm")
        self.assertEqual(stats.generated_tokens, 4)
        self.assertEqual(stats.prompt_tokens, 20)
        self.assertEqual(stats.prompt_tokens_per_second, 80.0)
        self.assertEqual(stats.generation_tokens_per_second, 40.0)
        self.assertEqual(stats.image_count, 1)
        self.assertFalse(stats.prompt_cache_enabled)
        self.assertEqual(stats.prompt_cache_prefix_tokens, 0)
        self.assertEqual(stats.cold_vision["mode"], "off")
        self.assertFalse(stats.cold_vision["enabled"])
        self.assertIsNone(stats.mean_token_logprob)
        self.assertIsNone(stats.minimum_token_logprob)

    def test_reports_selected_token_logprob_without_counting_final_row_twice(self):
        def confidence_stream(model, processor, prompt, **kwargs):
            first = [FakeScalar(-9.0), FakeScalar(-0.2)]
            second = [FakeScalar(-0.7), FakeScalar(-8.0)]
            yield FakeGenerationRow("A", generation_tokens=1, token=1, logprobs=first)
            yield FakeGenerationRow("B", generation_tokens=2, token=0, logprobs=second)
            yield FakeGenerationRow("", generation_tokens=2, token=0, logprobs=second)

        accelerator = MLXVLMAccelerator(
            FakeModel(),
            object(),
            model_name="fake-vlm",
            asset_cache_dir=self.root / "confidence-assets",
            stream_generate_fn=confidence_stream,
            apply_chat_template_fn=lambda *args, **kwargs: "templated prompt",
        )
        try:
            text, stats = accelerator.generate("test", max_tokens=2)
        finally:
            accelerator.close()

        self.assertEqual(text, "AB")
        self.assertAlmostEqual(stats.mean_token_logprob, -0.45)
        self.assertAlmostEqual(stats.minimum_token_logprob, -0.7)

    def test_final_token_decode_repairs_a_missing_stream_prefix(self):
        def truncated_stream(model, processor, prompt, **kwargs):
            yield FakeGenerationRow("", generation_tokens=1, token=10)
            yield FakeGenerationRow("’re ", generation_tokens=2, token=11)
            yield FakeGenerationRow("welcome!", generation_tokens=3, token=12)
            yield FakeGenerationRow("", generation_tokens=3, token=12)

        accelerator = MLXVLMAccelerator(
            FakeModel(),
            FakeProcessor("You’re welcome!"),
            model_name="fake-vlm",
            asset_cache_dir=self.root / "repair-assets",
            stream_generate_fn=truncated_stream,
            apply_chat_template_fn=lambda *args, **kwargs: "templated prompt",
        )
        emitted = []
        try:
            text, stats = accelerator.generate(
                "Thanks.",
                max_tokens=3,
                on_text=emitted.append,
            )
        finally:
            accelerator.close()

        self.assertEqual("".join(emitted), "’re welcome!")
        self.assertEqual(text, "You’re welcome!")
        self.assertEqual(stats.generated_tokens, 3)

    def test_cold_vision_resize_reaches_native_stream_and_stats(self):
        decision = ColdVisionDecision(
            mode="adaptive",
            enabled=True,
            target_max_edge=512,
            resize_shape=(512, 512),
            source_max_edge=1024,
            image_entropy=0.8,
            image_edge_density=0.2,
            question_class="text-detail",
            reason="test decision",
        )
        messages = [
            {"role": "user", "content": "Old chart question"},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "Read the current label", "images": [str(self.image)]},
        ]

        with patch(
            "machboost.adapters.mlx_vlm.choose_cold_vision",
            return_value=decision,
        ) as choose:
            _, stats = self.accelerator.generate_chat(
                messages,
                max_tokens=8,
                use_vision_cache=False,
                cold_vision_mode="adaptive",
            )

        choose.assert_called_once_with(
            "Read the current label",
            [str(self.image.resolve())],
            mode="adaptive",
            max_edge=None,
        )
        self.assertEqual(self.stream.calls[-1]["kwargs"]["resize_shape"], (512, 512))
        self.assertTrue(stats.cold_vision["enabled"])
        self.assertEqual(stats.cold_vision["target_max_edge"], 512)

    def test_post_fusion_mode_disables_request_cache_and_reports_stats(self):
        post_fusion = SimpleNamespace(
            mode="adaptive",
            info=lambda: {
                "mode": "adaptive",
                "enabled": True,
                "requested_retention_ratio": 0.35,
                "prune_after_layer": 3,
                "original_sequence_tokens": 700,
                "retained_sequence_tokens": 300,
                "original_visual_tokens": 640,
                "retained_visual_tokens": 224,
                "actual_visual_retention_ratio": 0.35,
            },
        )

        with patch(
            "machboost.adapters.mlx_vlm.configure_post_fusion_vision",
            return_value=post_fusion,
        ) as configure:
            _, stats = self.accelerator.generate(
                "Read the image.",
                images=[str(self.image)],
                max_tokens=8,
                vision_token_mode="adaptive",
                vision_token_ratio=0.35,
            )

        configure.assert_called_once_with(
            self.accelerator.model,
            mode="adaptive",
            retain_ratio=0.35,
            prune_after_layer=3,
            token_bucket=0,
            policy=ANY,
        )
        self.assertFalse(stats.visual_cache_enabled)
        self.assertFalse(stats.prompt_cache_enabled)
        self.assertEqual(stats.post_fusion_vision["retained_visual_tokens"], 224)
        self.assertNotIn("vision_cache", self.stream.calls[-1]["kwargs"])

    def test_auto_post_fusion_resolves_document_policy(self):
        post_fusion = SimpleNamespace(
            mode="adaptive",
            info=lambda: {
                "mode": "adaptive",
                "enabled": True,
                "requested_retention_ratio": 0.50,
                "prune_after_layer": 6,
                "token_bucket": 32,
            },
        )
        signals = VisionImageSignals(
            count=1,
            max_edge=1200,
            entropy=0.5,
            edge_density=0.1,
        )

        with (
            patch("machboost.vision_auto.inspect_vision_images", return_value=signals),
            patch(
                "machboost.adapters.mlx_vlm.configure_post_fusion_vision",
                return_value=post_fusion,
            ) as configure,
        ):
            self.accelerator.generate(
                "Read the invoice total.",
                images=[str(self.image)],
                max_tokens=8,
                vision_token_mode="auto",
            )

        options = configure.call_args.kwargs
        self.assertEqual(options["mode"], "adaptive")
        self.assertEqual(options["retain_ratio"], 0.50)
        self.assertEqual(options["prune_after_layer"], 6)
        self.assertEqual(options["token_bucket"], 32)
        self.assertEqual(options["policy"]["requested_mode"], "auto")
        self.assertEqual(options["policy"]["workload"], "document-text")

    def test_chat_template_receives_history_and_image_count(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up", "images": [str(self.image)]},
        ]

        self.accelerator.generate_chat(messages, max_tokens=8)

        prompt, options = self.templates[0]
        self.assertEqual(prompt[2]["content"], "Follow-up")
        self.assertEqual(options["num_images"], 1)
        self.assertFalse(options["enable_thinking"])

    def test_chat_template_can_enable_thinking(self):
        message = {"role": "user", "content": "Reason about this."}

        self.accelerator.generate_chat(
            [message],
            max_tokens=8,
            enable_thinking=True,
        )

        _, options = self.templates[0]
        self.assertTrue(options["enable_thinking"])

    def test_chat_template_receives_native_tools_and_reasoning_strength(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a repository file",
                    "parameters": {"type": "object"},
                },
            }
        ]

        self.accelerator.generate_chat(
            [{"role": "user", "content": "Inspect the code."}],
            max_tokens=8,
            tools=tools,
            tool_choice="required",
            reasoning_strength="medium",
        )

        _, options = self.templates[0]
        self.assertEqual(options["tools"], tools)
        self.assertEqual(options["tool_choice"], "required")
        self.assertEqual(options["reasoning_strength"], "medium")

    def test_low_reasoning_strength_bounds_native_thinking(self):
        self.accelerator.model.config = {"model_type": "muse_glimmer"}
        self.accelerator.generate_chat(
            [{"role": "user", "content": "Inspect the code."}],
            max_tokens=256,
            enable_thinking=True,
            reasoning_strength="low",
        )

        self.assertEqual(self.stream.calls[0]["kwargs"]["thinking_budget"], 64)
        self.assertEqual(self.stream.calls[0]["kwargs"]["thinking_start_token"], "to=self")
        self.assertEqual(self.stream.calls[0]["kwargs"]["thinking_end_token"], "<|eom|>")

    def test_muse_defaults_to_bounded_low_reasoning(self):
        self.accelerator.model.config = {"model_type": "muse_glimmer"}
        self.accelerator.generate_chat(
            [{"role": "user", "content": "Say hello."}],
            max_tokens=128,
        )

        _, template_options = self.templates[0]
        self.assertTrue(template_options["enable_thinking"])
        self.assertEqual(template_options["reasoning_strength"], "low")
        self.assertEqual(self.stream.calls[0]["kwargs"]["thinking_budget"], 64)

    def test_muse_completion_also_uses_bounded_low_reasoning(self):
        self.accelerator.model.config = {"model_type": "muse_glimmer"}
        self.accelerator.generate("Say hello.", max_tokens=128)

        _, template_options = self.templates[0]
        self.assertTrue(template_options["enable_thinking"])
        self.assertEqual(template_options["reasoning_strength"], "low")
        self.assertEqual(self.stream.calls[0]["kwargs"]["thinking_budget"], 64)

    def test_disabled_thinking_does_not_apply_reasoning_budget(self):
        self.accelerator.generate_chat(
            [{"role": "user", "content": "Inspect the code."}],
            max_tokens=256,
            reasoning_strength="low",
        )

        self.assertNotIn("thinking_budget", self.stream.calls[0]["kwargs"])

    def test_streams_reasoning_separately_across_marker_boundaries(self):
        self.accelerator.model.config = {
            "model_type": "muse_glimmer",
            "thinking_start_token": "to=self<|message|>",
            "thinking_end_token": "<|eom|>",
        }

        def reasoning_stream(model, processor, prompt, **kwargs):
            yield FakeGenerationRow("<|start|>assistant to=se")
            yield FakeGenerationRow("lf<|message|>Inspecting")
            yield FakeGenerationRow(" inputs<|eo")
            yield FakeGenerationRow("m|><|start|>assistant to=user<|message|>Done")

        self.accelerator._stream_generate = reasoning_stream
        content = []
        thinking = []

        text, stats = self.accelerator.generate_chat(
            [{"role": "user", "content": "Work it out."}],
            max_tokens=8,
            on_text=content.append,
            on_thinking=thinking.append,
            enable_thinking=True,
        )

        self.assertEqual("".join(thinking), "Inspecting inputs")
        self.assertEqual(stats.thinking, "Inspecting inputs")
        self.assertEqual(text, "Done")
        self.assertEqual("".join(content), text)

    def test_removes_initial_user_prompt_echo_from_reasoning(self):
        self.accelerator.model.config = {
            "model_type": "muse_glimmer",
            "thinking_start_token": "to=self<|message|>",
            "thinking_end_token": "<|eom|>",
        }

        def reasoning_stream(model, processor, prompt, **kwargs):
            yield FakeGenerationRow("<|start|>assistant to=self<|message|>Inspect")
            yield FakeGenerationRow(" this repository briefly.\n")
            yield FakeGenerationRow("\nI should list the top-level files.")
            yield FakeGenerationRow("<|eom|><|start|>assistant to=user<|message|>  Done")

        self.accelerator._stream_generate = reasoning_stream
        content = []
        thinking = []

        text, stats = self.accelerator.generate_chat(
            [{"role": "user", "content": "Inspect this repository briefly."}],
            max_tokens=32,
            on_text=content.append,
            on_thinking=thinking.append,
            enable_thinking=True,
        )

        self.assertEqual("".join(thinking), "I should list the top-level files.")
        self.assertEqual(stats.thinking, "I should list the top-level files.")
        self.assertEqual(text, "Done")
        self.assertEqual("".join(content), "Done")

    def test_removes_prompt_suffix_echo_without_hiding_following_reasoning(self):
        self.accelerator.model.config = {
            "model_type": "muse_glimmer",
            "thinking_start_token": "to=self<|message|>",
            "thinking_end_token": "<|eom|>",
        }

        def reasoning_stream(model, processor, prompt, **kwargs):
            yield FakeGenerationRow(
                "<|start|>assistant to=self<|message|>Reply briefly.\n\nReply"
            )
            yield FakeGenerationRow(" briefly. Reply")
            yield FakeGenerationRow(" briefly. So the answer is 4.")
            yield FakeGenerationRow("<|eom|><|start|>assistant to=user<|message|>4")

        self.accelerator._stream_generate = reasoning_stream
        thinking = []

        text, stats = self.accelerator.generate_chat(
            [{"role": "user", "content": "What is 2 + 2? Reply briefly."}],
            max_tokens=32,
            on_thinking=thinking.append,
            enable_thinking=True,
        )

        self.assertEqual("".join(thinking), "So the answer is 4.")
        self.assertEqual(stats.thinking, "So the answer is 4.")
        self.assertEqual(text, "4")

    def test_preserves_non_echo_initial_reasoning(self):
        self.accelerator.model.config = {
            "model_type": "muse_glimmer",
            "thinking_start_token": "to=self<|message|>",
            "thinking_end_token": "<|eom|>",
        }

        def reasoning_stream(model, processor, prompt, **kwargs):
            yield FakeGenerationRow("<|start|>assistant to=self<|message|>I should inspect")
            yield FakeGenerationRow(" the repository first.<|eom|>")
            yield FakeGenerationRow("<|start|>assistant to=user<|message|>Done")

        self.accelerator._stream_generate = reasoning_stream
        thinking = []

        text, _ = self.accelerator.generate_chat(
            [{"role": "user", "content": "Inspect this repository briefly."}],
            max_tokens=32,
            on_thinking=thinking.append,
            enable_thinking=True,
        )

        self.assertEqual("".join(thinking), "I should inspect the repository first.")
        self.assertEqual(text, "Done")

    def test_reset_cache_releases_projected_features(self):
        self.accelerator.generate(
            "Describe the image.", images=[str(self.image)], max_tokens=8
        )
        self.assertEqual(self.accelerator.cache_info()["size"], 1)

        self.accelerator.reset_cache()

        self.assertEqual(self.accelerator.cache_info()["size"], 0)

    def test_prompt_cache_is_scoped_to_exact_image_content(self):
        second_image = self.root / "second.png"
        second_image.write_bytes(b"\x89PNG\r\n\x1a\ndifferent-image")
        generation = SimpleNamespace(PromptCacheState=FakePromptCacheState)

        with patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ):
            first = self.accelerator._prompt_cache_for([str(self.image)])
            repeated = self.accelerator._prompt_cache_for([str(self.image)])
            different = self.accelerator._prompt_cache_for([str(second_image)])
            reduced = self.accelerator._prompt_cache_for(
                _resolution_scoped_images([str(self.image)], (512, 512))
            )

        self.assertIs(first, repeated)
        self.assertIsNot(first, different)
        self.assertIsNot(first, reduced)
        self.assertEqual(len(self.accelerator._prompt_caches), 3)

    def test_qwen_chat_keeps_image_tokens_on_first_user_turn(self):
        self.accelerator.model.config = {"model_type": "qwen2_5_vl"}
        self.accelerator._apply_chat_template.__module__ = "mlx_vlm.prompt_utils"
        formatted = []

        def get_message_json(model_type, content, role, **options):
            return {"role": role, "content": content, **options}

        template_options = {}

        def get_chat_template(processor, messages, add_generation_prompt, **options):
            formatted.extend(messages)
            template_options.update(options)
            return "rendered"

        prompt_utils = SimpleNamespace(
            get_message_json=get_message_json,
            get_chat_template=get_chat_template,
        )
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]

        with patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=prompt_utils,
        ):
            prompt = self.accelerator._format_chat_prompt(messages, image_count=1)

        self.assertEqual(prompt, "rendered")
        self.assertFalse(formatted[1]["skip_image_token"])
        self.assertTrue(formatted[3]["skip_image_token"])
        self.assertFalse(template_options["enable_thinking"])

    def test_qwen35_uses_projected_tensor_from_vision_tuple(self):
        expected = object()

        def vision_tower(pixel_values, grid, output_hidden_states):
            return expected, [object()]

        self.accelerator.model.vision_tower = vision_tower

        features = self.accelerator._encode_vision_features(
            object(),
            {"image_grid_thw": object()},
            "qwen3_5",
        )

        self.assertIs(features, expected)

    def test_qwen3vl_skips_incomplete_projected_feature_cache(self):
        self.accelerator.model.config = {"model_type": "qwen3_vl"}
        self.accelerator._stream_generate.__module__ = "mlx_vlm.generate"

        mlx_package = ModuleType("mlx")
        mlx_package.__path__ = []
        mlx_core = ModuleType("mlx.core")
        mlx_core.eval = lambda value: None
        mlx_package.core = mlx_core
        mlx_vlm_package = ModuleType("mlx_vlm")
        mlx_vlm_package.__path__ = []
        mlx_vlm_utils = ModuleType("mlx_vlm.utils")
        pixel_values = object()
        mlx_vlm_utils.prepare_inputs = lambda *args, **kwargs: {
            "input_ids": FakeArray([1, 2, 3]),
            "pixel_values": pixel_values,
            "attention_mask": object(),
            "image_grid_thw": object(),
        }
        generation = SimpleNamespace(PromptCacheState=FakePromptCacheState)

        with patch.dict(
            sys.modules,
            {
                "mlx": mlx_package,
                "mlx.core": mlx_core,
                "mlx_vlm": mlx_vlm_package,
                "mlx_vlm.utils": mlx_vlm_utils,
            },
        ), patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ):
            prepared = self.accelerator._prepare_cached_vision(
                "prompt", [str(self.image)]
            )

        self.assertIsNotNone(prepared)
        _, options, _ = prepared
        self.assertIs(options["pixel_values"], pixel_values)
        self.assertNotIn("cached_image_features", options)
        self.assertEqual(self.accelerator.cache_info()["size"], 0)

    def test_qwen35_uses_whole_state_checkpoint_for_prefix_reuse(self):
        self.accelerator.model.config = {"model_type": "qwen3_5"}
        self.accelerator._stream_generate.__module__ = "mlx_vlm.generate"
        self.accelerator.vision_cache.put([str(self.image)], "cached-features")

        mlx_package = ModuleType("mlx")
        mlx_package.__path__ = []
        mlx_core = ModuleType("mlx.core")
        mlx_core.eval = lambda value: None
        mlx_package.core = mlx_core
        mlx_vlm_package = ModuleType("mlx_vlm")
        mlx_vlm_package.__path__ = []
        mlx_vlm_utils = ModuleType("mlx_vlm.utils")
        mlx_vlm_utils.prepare_inputs = lambda *args, **kwargs: {
            "input_ids": FakeArray([1, 2, 3]),
            "pixel_values": object(),
            "attention_mask": object(),
            "image_grid_thw": object(),
        }

        def make_state():
            state = FakePromptCacheState([1, 2])
            state.cache = object()
            return state

        generation = SimpleNamespace(PromptCacheState=make_state)
        apc_manager = object()

        with patch.dict(
            sys.modules,
            {
                "mlx": mlx_package,
                "mlx.core": mlx_core,
                "mlx_vlm": mlx_vlm_package,
                "mlx_vlm.utils": mlx_vlm_utils,
            },
        ), patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ), patch.object(
            self.accelerator,
            "_get_apc_manager",
            return_value=apc_manager,
        ):
            prepared = self.accelerator._prepare_cached_vision(
                "prompt", [str(self.image)]
            )

        self.assertIsNotNone(prepared)
        _, options, prefix_tokens = prepared
        self.assertEqual(prefix_tokens, 2)
        self.assertIs(options["apc_manager"], apc_manager)
        self.assertIsNone(options["prompt_cache_state"].cache)
        self.assertEqual(options["cached_image_features"], "cached-features")

    def test_apc_checkpoint_match_is_included_in_prefix_metrics(self):
        class FakeAPCManager:
            def __init__(self):
                self.snapshots = iter((0, 776))

            def stats_snapshot(self):
                return {"matched_tokens": next(self.snapshots)}

        apc_manager = FakeAPCManager()
        prepared = (None, {"apc_manager": apc_manager}, 0)

        with patch.object(
            self.accelerator,
            "_prepare_cached_vision",
            return_value=prepared,
        ):
            _, stats = self.accelerator.generate(
                "Describe the image.",
                images=[str(self.image)],
                max_tokens=8,
            )

        self.assertTrue(stats.prompt_cache_enabled)
        self.assertEqual(stats.prompt_cache_prefix_tokens, 776)

    def test_text_only_prompts_use_cross_request_prefix_cache(self):
        class FakeAPCManager:
            def __init__(self):
                self.snapshots = iter((0, 512))

            def stats_snapshot(self):
                return {"matched_tokens": next(self.snapshots)}

        calls = []

        def mlx_stream(model, processor, prompt, **kwargs):
            calls.append(kwargs)
            yield FakeGenerationRow("cached")

        mlx_stream.__module__ = "mlx_vlm.generate"
        self.accelerator._stream_generate = mlx_stream
        apc_manager = FakeAPCManager()

        with patch.object(
            self.accelerator,
            "_get_apc_manager",
            return_value=apc_manager,
        ), patch.object(self.accelerator, "_bind_thread_local_stream"):
            _, stats = self.accelerator.generate("Repository prompt", max_tokens=8)

        self.assertIs(calls[0]["apc_manager"], apc_manager)
        self.assertTrue(stats.prompt_cache_enabled)
        self.assertEqual(stats.prompt_cache_prefix_tokens, 512)

    def test_text_only_affinity_reuses_exact_conversation_cache(self):
        states = []

        def mlx_stream(model, processor, prompt, **kwargs):
            states.append(kwargs.get("prompt_cache_state"))
            yield FakeGenerationRow("cached")

        mlx_stream.__module__ = "mlx_vlm.generate"
        self.accelerator._stream_generate = mlx_stream
        generation = SimpleNamespace(PromptCacheState=FakePromptCacheState)

        with patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ), patch.object(
            self.accelerator,
            "_get_apc_manager",
            return_value=None,
        ), patch.object(self.accelerator, "_bind_thread_local_stream"):
            _, first = self.accelerator.generate(
                "First prompt", max_tokens=8, cache_key="conversation-a"
            )
            self.accelerator.generate(
                "First prompt plus a tool result",
                max_tokens=8,
                cache_key="conversation-a",
            )
            self.accelerator.generate(
                "Unrelated prompt", max_tokens=8, cache_key="conversation-b"
            )

        self.assertTrue(first.prompt_cache_enabled)
        self.assertIsNotNone(states[0])
        self.assertIs(states[0], states[1])
        self.assertIsNot(states[0], states[2])

    def test_text_only_apc_is_scoped_to_the_request_affinity(self):
        calls = []

        class FakeAPCManager:
            def stats_snapshot(self):
                return {"matched_tokens": 0}

        def mlx_stream(model, processor, prompt, **kwargs):
            calls.append(kwargs)
            yield FakeGenerationRow("cached")

        mlx_stream.__module__ = "mlx_vlm.generate"
        self.accelerator._stream_generate = mlx_stream
        generation = SimpleNamespace(PromptCacheState=FakePromptCacheState)

        with patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ), patch.object(
            self.accelerator,
            "_get_apc_manager",
            return_value=FakeAPCManager(),
        ), patch.object(self.accelerator, "_bind_thread_local_stream"):
            self.accelerator.generate(
                "Repository prompt",
                max_tokens=8,
                cache_key="conversation-a",
            )

        self.assertEqual(calls[0]["apc_tenant"], "conversation-a")

    def test_qwen_partial_prefix_drops_untrimmed_attention_mask(self):
        self.accelerator.model.config = {"model_type": "qwen2_5_vl"}
        self.accelerator._stream_generate.__module__ = "mlx_vlm.generate"
        self.accelerator.vision_cache.put([str(self.image)], "cached-features")

        mlx_package = ModuleType("mlx")
        mlx_package.__path__ = []
        mlx_core = ModuleType("mlx.core")
        mlx_core.eval = lambda value: None
        mlx_package.core = mlx_core
        mlx_vlm_package = ModuleType("mlx_vlm")
        mlx_vlm_package.__path__ = []
        mlx_vlm_utils = ModuleType("mlx_vlm.utils")
        full_mask = object()
        mlx_vlm_utils.prepare_inputs = lambda *args, **kwargs: {
            "input_ids": FakeArray([1, 2, 3]),
            "pixel_values": object(),
            "attention_mask": full_mask,
            "image_grid_thw": object(),
        }
        generation = SimpleNamespace(
            PromptCacheState=lambda: FakePromptCacheState([1, 2])
        )

        with patch.dict(
            sys.modules,
            {
                "mlx": mlx_package,
                "mlx.core": mlx_core,
                "mlx_vlm": mlx_vlm_package,
                "mlx_vlm.utils": mlx_vlm_utils,
            },
        ), patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ):
            prepared = self.accelerator._prepare_cached_vision(
                "prompt", [str(self.image)]
            )

        self.assertIsNotNone(prepared)
        _, options, prefix_tokens = prepared
        self.assertEqual(prefix_tokens, 2)
        self.assertIsNone(options["mask"])

    def test_reset_cache_releases_prompt_states(self):
        generation = SimpleNamespace(PromptCacheState=FakePromptCacheState)
        with patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=generation,
        ):
            self.accelerator._prompt_cache_for([str(self.image)])

        self.accelerator.reset_cache()

        self.assertEqual(len(self.accelerator._prompt_caches), 0)

    def test_apc_pool_size_can_be_tuned_for_machine_memory(self):
        observed = {}

        class FakeAPCManager:
            def __init__(self, **kwargs):
                observed.update(kwargs)

            def clear(self):
                pass

        with patch.dict(
            "os.environ",
            {"MACHBOOST_MLX_APC_BLOCKS": "1024"},
        ), patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=SimpleNamespace(APCManager=FakeAPCManager),
        ):
            self.accelerator._get_apc_manager()

        self.assertEqual(observed, {"num_blocks": 1024, "block_size": 16})

    def test_apc_disk_cache_uses_model_namespace_and_bounded_storage(self):
        observed = {}

        class FakeDiskBlockStore:
            def __init__(self, root, **kwargs):
                observed["disk"] = self
                observed["disk_root"] = root
                observed["disk_options"] = kwargs

        class FakeAPCManager:
            def __init__(self, **kwargs):
                observed["manager_options"] = kwargs

            def clear(self):
                pass

        disk_root = self.root / "persistent-apc"
        snapshot_config = (
            self.root
            / "models--mlx-community--fake-vlm"
            / "snapshots"
            / "0123456789abcdef0123456789abcdef"
            / "config.json"
        )

        def import_module(name):
            if name == "huggingface_hub":
                return SimpleNamespace(
                    try_to_load_from_cache=lambda *_args, **_kwargs: snapshot_config
                )
            return SimpleNamespace(
                APCManager=FakeAPCManager,
                DiskBlockStore=FakeDiskBlockStore,
            )

        with patch.dict(
            "os.environ",
            {
                "MACHBOOST_MLX_APC_DISK": "1",
                "MACHBOOST_MLX_APC_DISK_PATH": str(disk_root),
                "MACHBOOST_MLX_APC_DISK_GB": "1.5",
            },
        ), patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            side_effect=import_module,
        ):
            self.accelerator._get_apc_manager()

        self.assertEqual(observed["disk_root"], disk_root)
        self.assertEqual(
            observed["disk_options"]["namespace"],
            "0123456789abcdef-fake-vlm",
        )
        self.assertEqual(observed["disk_options"]["num_workers"], 1)
        self.assertEqual(
            observed["disk_options"]["max_bytes"],
            int(1.5 * (1 << 30)),
        )
        self.assertIs(
            observed["manager_options"]["disk"],
            observed["disk"],
        )

    def test_apc_disk_cache_can_be_disabled(self):
        observed = {}

        class FakeAPCManager:
            def __init__(self, **kwargs):
                observed.update(kwargs)

            def clear(self):
                pass

        class UnexpectedDiskBlockStore:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("disk cache should be disabled")

        with patch.dict(
            "os.environ",
            {"MACHBOOST_MLX_APC_DISK": "0"},
        ), patch(
            "machboost.adapters.mlx_vlm.importlib.import_module",
            return_value=SimpleNamespace(
                APCManager=FakeAPCManager,
                DiskBlockStore=UnexpectedDiskBlockStore,
            ),
        ):
            self.accelerator._get_apc_manager()

        self.assertNotIn("disk", observed)

    def test_close_stops_worker_and_rejects_future_generation(self):
        closed = []

        class FakeAPCManager:
            def clear(self):
                pass

            def close(self):
                closed.append(True)

        self.accelerator._apc_manager = FakeAPCManager()
        self.accelerator.generate(
            "Describe the image.", images=[str(self.image)], max_tokens=8
        )

        self.accelerator.close()

        self.assertEqual(closed, [True])

        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.accelerator.generate(
                "Describe the image.", images=[str(self.image)], max_tokens=8
            )


if __name__ == "__main__":
    unittest.main()
