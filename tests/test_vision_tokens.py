from __future__ import annotations

import unittest
from types import SimpleNamespace

from machboost.vision_tokens import (
    PostFusionVisionModel,
    _contiguous_segments,
    bucket_token_target,
    configure_post_fusion_vision,
    spatial_group_factor,
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
            mode="random",
            retain_ratio=0.25,
            prune_after_layer=6,
            token_bucket=32,
            policy={"source": "test"},
        )

        self.assertIs(first, repeated)
        self.assertIs(model.language_model.model, first)
        self.assertEqual(repeated.mode, "random")
        self.assertEqual(repeated.retain_ratio, 0.25)
        self.assertEqual(repeated.prune_after_layer, 6)
        self.assertEqual(repeated.token_bucket, 32)
        self.assertEqual(repeated.policy, {"source": "test"})
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
        with self.assertRaisesRegex(ValueError, "bucket"):
            wrapper.configure(
                mode="merge",
                retain_ratio=0.25,
                token_bucket=-1,
            )

    def test_info_reports_actual_retention(self) -> None:
        wrapper = PostFusionVisionModel(FakeInnerModel())
        wrapper.configure(mode="adaptive", retain_ratio=0.35)
        wrapper.original_sequence_tokens = 700
        wrapper.retained_sequence_tokens = 300
        wrapper.original_visual_tokens = 640
        wrapper.retained_visual_tokens = 224
        wrapper.target_visual_tokens = 224
        wrapper.applied_after_layer = 6

        info = wrapper.info()

        self.assertTrue(info["enabled"])
        self.assertEqual(info["actual_visual_retention_ratio"], 0.35)
        self.assertEqual(info["retained_sequence_tokens"], 300)
        self.assertEqual(info["target_visual_tokens"], 224)
        self.assertEqual(info["applied_after_layer"], 6)

    def test_contiguous_segments_separate_multiple_visual_spans(self) -> None:
        self.assertEqual(
            _contiguous_segments([3, 4, 5, 9, 10, 20]),
            [0, 0, 0, 1, 1, 2],
        )

    def test_bucket_target_rounds_up_and_never_exceeds_visual_count(self) -> None:
        self.assertEqual(bucket_token_target(672, 0.35), 236)
        self.assertEqual(bucket_token_target(672, 0.35, 32), 256)
        self.assertEqual(bucket_token_target(100, 0.95, 32), 96)
        self.assertEqual(bucket_token_target(0, 0.35, 32), 0)

    def test_half_retention_still_groups_neighboring_visual_tokens(self) -> None:
        self.assertEqual(spatial_group_factor(1.0), 1)
        self.assertEqual(spatial_group_factor(0.5), 2)
        self.assertEqual(spatial_group_factor(0.35), 2)


if __name__ == "__main__":
    unittest.main()
