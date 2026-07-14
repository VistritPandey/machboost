from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from machboost.adapters.mlx_vlm import MLXVLMAccelerator


@dataclass
class FakeGenerationRow:
    text: str
    generation_tokens: int = 4
    prompt_tokens: int = 20
    prompt_tps: float = 80.0
    generation_tps: float = 40.0
    peak_memory: float = 1.5


class FakeModel:
    config = {"model_type": "fake_vlm"}


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

        self.assertIs(first, repeated)
        self.assertIsNot(first, different)
        self.assertEqual(len(self.accelerator._prompt_caches), 2)

    def test_qwen_chat_keeps_image_tokens_on_first_user_turn(self):
        self.accelerator.model.config = {"model_type": "qwen2_5_vl"}
        self.accelerator._apply_chat_template.__module__ = "mlx_vlm.prompt_utils"
        formatted = []

        def get_message_json(model_type, content, role, **options):
            return {"role": role, "content": content, **options}

        def get_chat_template(processor, messages, add_generation_prompt):
            formatted.extend(messages)
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

    def test_close_stops_worker_and_rejects_future_generation(self):
        self.accelerator.generate(
            "Describe the image.", images=[str(self.image)], max_tokens=8
        )

        self.accelerator.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.accelerator.generate(
                "Describe the image.", images=[str(self.image)], max_tokens=8
            )


if __name__ == "__main__":
    unittest.main()
