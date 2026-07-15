from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from machboost.vision_policy import choose_cold_vision


class ColdVisionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is not installed")

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        simple = Image.new("RGB", (1024, 768), "white")
        draw = ImageDraw.Draw(simple)
        draw.text((100, 100), "Status: READY", fill="black")
        self.simple = self.root / "simple.png"
        simple.save(self.simple)

        dense = Image.new("RGB", (1024, 768), "white")
        draw = ImageDraw.Draw(dense)
        for y in range(0, 768, 8):
            for x in range(0, 1024, 16):
                shade = (x * 17 + y * 29) % 256
                draw.rectangle((x, y, x + 7, y + 3), fill=(shade, 255 - shade, shade // 2))
        self.dense = self.root / "dense.png"
        dense.save(self.dense)

        small = Image.new("RGB", (320, 240), "white")
        self.small = self.root / "small.png"
        small.save(self.small)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_adaptive_simple_layout_uses_compact_budget(self) -> None:
        decision = choose_cold_vision(
            "Return only the status shown in the image.",
            [str(self.simple)],
            mode="adaptive",
        )

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.target_max_edge, 336)
        self.assertEqual(decision.resize_shape, (336, 336))
        self.assertEqual(decision.question_class, "general")

    def test_adaptive_dense_text_question_uses_medium_budget(self) -> None:
        decision = choose_cold_vision(
            "What does the small text spell?",
            [str(self.dense)],
            mode="adaptive",
        )

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.target_max_edge, 512)
        self.assertEqual(decision.question_class, "text-detail")

    def test_adaptive_chart_question_keeps_larger_budget(self) -> None:
        decision = choose_cold_vision(
            "What is the difference between the highest and lowest bar in the chart?",
            [str(self.dense)],
            mode="adaptive",
        )

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.target_max_edge, 672)
        self.assertEqual(decision.question_class, "structured-detail")

    def test_policy_never_upscales_small_images(self) -> None:
        decision = choose_cold_vision(
            "Read the label.",
            [str(self.small)],
            mode="quality",
        )

        self.assertFalse(decision.enabled)
        self.assertIsNone(decision.resize_shape)
        self.assertIn("already within", decision.reason)

    def test_explicit_budget_overrides_mode_budget(self) -> None:
        decision = choose_cold_vision(
            "Describe the image.",
            [str(self.dense)],
            mode="balanced",
            max_edge=420,
        )

        self.assertEqual(decision.target_max_edge, 420)
        self.assertEqual(decision.resize_shape, (420, 420))

    def test_off_mode_does_not_open_images(self) -> None:
        decision = choose_cold_vision(
            "Describe the image.",
            [str(self.root / "missing.png")],
            mode="off",
        )

        self.assertFalse(decision.enabled)
        self.assertIsNone(decision.resize_shape)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cold vision mode"):
            choose_cold_vision("Describe it.", [str(self.simple)], mode="turbo")

    def test_tiny_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 56"):
            choose_cold_vision(
                "Describe it.",
                [str(self.simple)],
                mode="fast",
                max_edge=32,
            )


if __name__ == "__main__":
    unittest.main()
