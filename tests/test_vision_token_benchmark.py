from __future__ import annotations

import unittest

from scripts.benchmark_vision_tokens import (
    DEFAULT_PROFILES,
    _add_pair_metrics,
    parse_profile,
    parse_profiles,
)


class VisionTokenBenchmarkTests(unittest.TestCase):
    def test_default_ablation_includes_controls_and_auto(self) -> None:
        profiles = parse_profiles(DEFAULT_PROFILES)

        self.assertEqual(len(profiles), 6)
        self.assertEqual(profiles[0].mode, "random")
        self.assertEqual(profiles[-1].mode, "auto")
        self.assertIn("adaptive-r0.5-l6-b32", {profile.slug for profile in profiles})

    def test_profile_parser_supports_partial_and_full_shapes(self) -> None:
        automatic = parse_profile("auto")
        explicit = parse_profile("merge:0.45:6:32")

        self.assertEqual(automatic.slug, "auto-r0.35-lauto-bauto")
        self.assertEqual(explicit.retain_ratio, 0.45)
        self.assertEqual(explicit.prune_after_layer, 6)
        self.assertEqual(explicit.token_bucket, 32)

    def test_profile_parser_rejects_baseline_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate mode"):
            parse_profile("off")
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_profiles("auto,auto")

    def test_pair_metrics_use_shared_baseline(self) -> None:
        baseline = {"client_total_seconds": 4.0, "output": "Blue square"}
        accelerated = {"client_total_seconds": 2.0, "output": "blue-square"}

        _add_pair_metrics(baseline, accelerated)

        self.assertEqual(accelerated["paired_total_speedup"], 2.0)
        self.assertFalse(accelerated["paired_literal_output_equal"])
        self.assertTrue(accelerated["paired_normalized_output_equal"])


if __name__ == "__main__":
    unittest.main()
