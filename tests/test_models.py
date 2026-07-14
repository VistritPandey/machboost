import unittest
from unittest.mock import patch

from machboost.models import MODEL_ALIASES, alias_rows, model_targets, resolve_model


class ModelCatalogTests(unittest.TestCase):
    def test_short_alias_prefers_mlx_when_available(self):
        with patch("machboost.models.native_mlx_available", return_value=True):
            resolution = resolve_model("qwen2.5:3b")

        self.assertEqual(resolution.backend, "mlx")
        self.assertEqual(resolution.model, "mlx-community/Qwen2.5-3B-Instruct-4bit")
        self.assertEqual(resolution.alias, "qwen2.5:3b")

    def test_short_alias_falls_back_to_hugging_face(self):
        with patch("machboost.models.native_mlx_available", return_value=False):
            resolution = resolve_model("qwen2.5:3b")

        self.assertEqual(resolution.backend, "hf")
        self.assertEqual(resolution.model, "Qwen/Qwen2.5-3B-Instruct")

    def test_explicit_backend_overrides_platform_selection(self):
        with patch("machboost.models.native_mlx_available", return_value=True):
            resolution = resolve_model("qwen2.5-coder:3b", backend="hf")

        self.assertEqual(resolution.backend, "hf")
        self.assertEqual(resolution.model, "Qwen/Qwen2.5-Coder-3B-Instruct")

    def test_full_repo_id_passes_through(self):
        resolution = resolve_model("mlx-community/custom-model", backend="auto")

        self.assertEqual(resolution.model, "mlx-community/custom-model")
        self.assertEqual(resolution.backend, "mlx")
        self.assertIsNone(resolution.alias)

    def test_catalog_rows_are_stable_and_sorted(self):
        rows = alias_rows()

        self.assertEqual([row["name"] for row in rows], sorted(MODEL_ALIASES))
        self.assertTrue(all(row["mlx"] or row["hf"] for row in rows))

    def test_alias_targets_include_both_native_backends(self):
        self.assertEqual(
            model_targets("qwen2.5:3b"),
            {
                "mlx-community/Qwen2.5-3B-Instruct-4bit",
                "Qwen/Qwen2.5-3B-Instruct",
            },
        )

    def test_vision_alias_prefers_mlx_vlm_when_available(self):
        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            resolution = resolve_model("qwen2.5-vl:3b")

        self.assertEqual(resolution.backend, "mlx-vlm")
        self.assertEqual(resolution.model, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
        self.assertEqual(resolution.alias, "qwen2.5-vl:3b")

    def test_vision_alias_falls_back_to_hf_vlm(self):
        with patch("machboost.models.native_mlx_vlm_available", return_value=False):
            resolution = resolve_model("qwen2.5-vl:3b")

        self.assertEqual(resolution.backend, "hf-vlm")
        self.assertEqual(resolution.model, "Qwen/Qwen2.5-VL-3B-Instruct")

    def test_explicit_mlx_backend_is_normalized_for_vision_alias(self):
        resolution = resolve_model("qwen2-vl:2b", backend="mlx")

        self.assertEqual(resolution.backend, "mlx-vlm")
        self.assertEqual(resolution.model, "mlx-community/Qwen2-VL-2B-Instruct-4bit")

    def test_full_vision_repo_selects_vlm_backend(self):
        mlx_resolution = resolve_model("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
        hf_resolution = resolve_model("Qwen/Qwen2.5-VL-3B-Instruct")

        self.assertEqual(mlx_resolution.backend, "mlx-vlm")
        self.assertEqual(hf_resolution.backend, "hf-vlm")

    def test_vision_alias_targets_include_mlx_and_hf_variants(self):
        self.assertEqual(
            model_targets("qwen2.5-vl:3b"),
            {
                "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                "Qwen/Qwen2.5-VL-3B-Instruct",
            },
        )


if __name__ == "__main__":
    unittest.main()
