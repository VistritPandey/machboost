from __future__ import annotations

import unittest
from types import SimpleNamespace

from machboost.vision_tokens import (
    PostFusionVisionModel,
    _contiguous_segments,
    configure_post_fusion_vision,
)


class FakeInnerModel:
    embed_tokens = object()
    layers = []
    norm = object()

    def _deepstack_process(self):
        return None


def fake_model(model_type: str = "qwen3_vl") -> SimpleNamespace:
    return SimpleNamespace(
        config={"model_type": model_type},
        language_model=SimpleNamespace(model=FakeInnerModel()),
    )


class PostFusionVisionTests(unittest.TestCase):
    def test_off_mode_does_not_wrap_an_unmodified_model(self) -> None:
        model = fake_model()

        wrapper = configure_post_fusion_vision(model, mode="off")

        self.assertIsNone(wrapper)
        self.assertIsInstance(model.language_model.model, FakeInnerModel)

    def test_configure_wraps_qwen3vl_once_and_updates_mode(self) -> None:
        model = fake_model()

        first = configure_post_fusion_vision(
            model,
            mode="adaptive",
            retain_ratio=0.35,
        )
        repeated = configure_post_fusion_vision(
            model,
            mode="merge",
            retain_ratio=0.25,
        )

        self.assertIs(first, repeated)
        self.assertIs(model.language_model.model, first)
        self.assertEqual(repeated.mode, "merge")
        self.assertEqual(repeated.retain_ratio, 0.25)
        self.assertFalse(repeated.info()["enabled"])

    def test_unsupported_model_rejects_compression(self) -> None:
        with self.assertRaisesRegex(ValueError, "Qwen3-VL only"):
            configure_post_fusion_vision(fake_model("gemma3"), mode="adaptive")

    def test_configuration_validates_mode_ratio_and_layer(self) -> None:
        wrapper = PostFusionVisionModel(FakeInnerModel())

        with self.assertRaisesRegex(ValueError, "mode"):
            wrapper.configure(mode="turbo", retain_ratio=0.35)
        with self.assertRaisesRegex(ValueError, "ratio"):
            wrapper.configure(mode="merge", retain_ratio=0.01)
        with self.assertRaisesRegex(ValueError, "layer"):
            wrapper.configure(
                mode="merge",
                retain_ratio=0.25,
                prune_after_layer=0,
            )

    def test_info_reports_actual_retention(self) -> None:
        wrapper = PostFusionVisionModel(FakeInnerModel())
        wrapper.configure(mode="adaptive", retain_ratio=0.35)
        wrapper.original_sequence_tokens = 700
        wrapper.retained_sequence_tokens = 300
        wrapper.original_visual_tokens = 640
        wrapper.retained_visual_tokens = 224

        info = wrapper.info()

        self.assertTrue(info["enabled"])
        self.assertEqual(info["actual_visual_retention_ratio"], 0.35)
        self.assertEqual(info["retained_sequence_tokens"], 300)

    def test_contiguous_segments_separate_multiple_visual_spans(self) -> None:
        self.assertEqual(
            _contiguous_segments([3, 4, 5, 9, 10, 20]),
            [0, 0, 0, 1, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
