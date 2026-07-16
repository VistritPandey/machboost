from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from machboost.video import (
    TemporalVideoSampler,
    frame_change_scores,
    select_temporal_frames,
    select_uniform_frames,
)


class VideoTests(unittest.TestCase):
    def test_temporal_selection_keeps_changes_and_boundaries(self) -> None:
        selected = select_temporal_frames(
            (1.0, 0.01, 0.8, 0.02, 0.7, 0.01),
            threshold=0.1,
            max_frames=8,
        )

        self.assertEqual(selected, (0, 2, 4, 5))

    def test_temporal_selection_caps_by_change_strength(self) -> None:
        selected = select_temporal_frames(
            (1.0, 0.4, 0.9, 0.8, 0.1),
            threshold=0.0,
            max_frames=3,
        )

        self.assertEqual(selected, (0, 2, 4))

    def test_single_frame_budget_keeps_first_frame(self) -> None:
        self.assertEqual(
            select_temporal_frames((1.0, 0.8, 0.7), threshold=0.0, max_frames=1),
            (0,),
        )

    def test_uniform_selection_spans_the_full_video(self) -> None:
        self.assertEqual(select_uniform_frames(10, max_frames=4), (0, 3, 6, 9))
        self.assertEqual(select_uniform_frames(3, max_frames=10), (0, 1, 2))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is optional")
    def test_change_score_detects_equal_luminance_color_transitions(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            red = Path(tmp) / "red.png"
            green = Path(tmp) / "green.png"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(red)
            Image.new("RGB", (32, 32), (0, 130, 0)).save(green)

            scores = frame_change_scores((red, green))

        self.assertGreater(scores[1], 0.1)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is optional")
    def test_sampler_reports_selected_frame_telemetry(self) -> None:
        from PIL import Image

        class FixtureSampler(TemporalVideoSampler):
            def __init__(self, frames, cache_dir):
                super().__init__(cache_dir=cache_dir, ffmpeg="true")
                self.frames = frames

            def _extract_frames(self, source, extraction_dir, fps):
                return self.frames, True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip.mp4"
            source.write_bytes(b"fixture")
            frames = []
            for index, color in enumerate(("black", "black", "white", "white")):
                frame = root / f"frame-{index}.png"
                Image.new("RGB", (32, 32), color).save(frame)
                frames.append(frame)

            result = FixtureSampler(frames, root / "cache").sample(
                source,
                fps=2.0,
                change_threshold=0.1,
                max_frames=4,
            )

        self.assertEqual(result.sampled_frames, 4)
        self.assertEqual(result.strategy, "temporal-change")
        self.assertEqual(result.selected_frames, 3)
        self.assertEqual([frame.sample_index for frame in result.frames], [0, 2, 3])
        self.assertEqual(result.frames[1].timestamp_seconds, 1.0)
        self.assertEqual(result.reduction_rate, 0.25)
        self.assertTrue(result.extraction_cache_hit)


if __name__ == "__main__":
    unittest.main()
