import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from machboost.models import (
    DFLASH_ALIASES,
    MODEL_ALIASES,
    alias_rows,
    cached_repo_path,
    catalog_rows,
    model_targets,
    model_repositories,
    preflight_model,
    resolve_model,
)


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

    def test_explicit_dflash_backend_resolves_mlx_target_for_hybrid_alias(self):
        resolution = resolve_model("qwen3.5:9b", backend="dflash")

        self.assertEqual(resolution.backend, "dflash")
        self.assertEqual(resolution.model, "mlx-community/Qwen3.5-9B-MLX-bf16")
        self.assertEqual(resolution.alias, "qwen3.5:9b")

    def test_explicit_dflash_backend_rejects_unsupported_alias(self):
        with self.assertRaisesRegex(ValueError, "supported aliases"):
            resolve_model("llama3.2:3b", backend="dflash")

    def test_dflash_alias_selects_accelerated_backend_without_custom_options(self):
        resolution = resolve_model("qwen3.5:4b-dflash")

        self.assertEqual(resolution.backend, "dflash")
        self.assertEqual(resolution.model, "mlx-community/Qwen3.5-4B-MLX-bf16")
        self.assertEqual(resolution.alias, "qwen3.5:4b-dflash")

    def test_dflash_alias_rejects_incompatible_backend_override(self):
        with self.assertRaisesRegex(ValueError, "requires the DFlash backend"):
            resolve_model("qwen3.5:4b-dflash", backend="mlx")

    def test_dflash_alias_lists_target_and_draft_repositories(self):
        self.assertEqual(
            model_repositories("qwen3.5:9b-dflash"),
            (
                "mlx-community/Qwen3.5-9B-MLX-bf16",
                "z-lab/Qwen3.5-9B-DFlash",
            ),
        )

    def test_catalog_rows_are_stable_and_sorted(self):
        rows = alias_rows()

        expected_names = sorted((*MODEL_ALIASES, *DFLASH_ALIASES))
        self.assertEqual([row["name"] for row in rows], expected_names)
        self.assertTrue(all(row["mlx"] or row["hf"] for row in rows))

    def test_desktop_catalog_includes_capabilities_and_resource_guidance(self):
        with (
            patch("machboost.models.cached_repo_path", return_value=None),
            patch("machboost.models.backend_available", return_value=True),
        ):
            rows = catalog_rows()

        llama = next(row for row in rows if row["name"] == "llama3.2:3b")
        vision = next(row for row in rows if row["name"] == "qwen3-vl:4b")
        self.assertEqual(llama["capabilities"], ["chat", "completion"])
        self.assertEqual(vision["capabilities"], ["chat", "vision"])
        self.assertGreater(llama["download_size_gb"], 0)
        self.assertTrue(vision["recommended"])
        self.assertFalse(vision["cached"])

    def test_desktop_catalog_requires_both_dflash_repositories_in_cache(self):
        target = Path("/tmp/dflash-target")
        draft = Path("/tmp/dflash-draft")

        def cached(model):
            if model == "mlx-community/Qwen3.5-4B-MLX-bf16":
                return target
            if model == "z-lab/Qwen3.5-4B-DFlash":
                return draft
            return None

        with (
            patch("machboost.models.cached_repo_path", side_effect=cached),
            patch("machboost.models.backend_available", return_value=True),
            patch("machboost.models._directory_size_gb", return_value=1.0),
        ):
            rows = catalog_rows(include_cached_repositories=False)

        accelerated = next(row for row in rows if row["name"] == "qwen3.5:4b-dflash")
        experimental = next(row for row in rows if row["name"] == "qwen3.5:9b-dflash")
        self.assertEqual(accelerated["backend"], "dflash")
        self.assertEqual(accelerated["draft_repository"], "z-lab/Qwen3.5-4B-DFlash")
        self.assertTrue(accelerated["cached"])
        self.assertEqual(accelerated["disk_size_gb"], 2.0)
        self.assertFalse(accelerated["experimental"])
        self.assertEqual(accelerated["validation_status"], "passed_bounded_suite")
        self.assertTrue(experimental["experimental"])
        self.assertEqual(experimental["validation_status"], "divergence_observed")

    def test_desktop_catalog_discovers_compatible_custom_mlx_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(
                directory,
                "models--acme--Custom-3B-MLX-4bit",
                "snapshots",
                "revision",
            )
            snapshot.mkdir(parents=True)
            Path(snapshot, "config.json").write_text(
                json.dumps({"model_type": "qwen2"}),
                encoding="utf-8",
            )
            Path(snapshot, "weights.safetensors").write_bytes(b"weights")
            with (
                patch("machboost.models.cached_repo_path", return_value=None),
                patch("machboost.models.backend_available", return_value=True),
                patch("machboost.models._validate_cached_mlx_architecture") as validate,
            ):
                rows = catalog_rows(cache_dirs=[Path(directory)])

        custom = next(
            row for row in rows if row["name"] == "acme/Custom-3B-MLX-4bit"
        )
        self.assertTrue(custom["cached"])
        self.assertEqual(custom["backend"], "mlx")
        self.assertEqual(custom["support"], "ready")
        self.assertFalse(custom["tested"])
        self.assertGreater(custom["disk_size_gb"], 0)
        validate.assert_called_once()

    def test_desktop_catalog_reports_unsupported_custom_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(
                directory,
                "models--acme--Future-MLX-4bit",
                "snapshots",
                "revision",
            )
            snapshot.mkdir(parents=True)
            Path(snapshot, "config.json").write_text(
                json.dumps({"model_type": "future_model"}),
                encoding="utf-8",
            )
            with (
                patch("machboost.models.cached_repo_path", return_value=None),
                patch("machboost.models.backend_available", return_value=True),
                patch(
                    "machboost.models._validate_cached_mlx_architecture",
                    side_effect=ValueError("future_model is not supported"),
                ),
            ):
                rows = catalog_rows(cache_dirs=[Path(directory)])

        custom = next(
            row for row in rows if row["name"] == "acme/Future-MLX-4bit"
        )
        self.assertEqual(custom["support"], "unsupported")
        self.assertIn("not supported", custom["support_reason"])

    def test_preflight_reports_supported_local_mlx_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps({"model_type": "qwen2"}),
                encoding="utf-8",
            )
            with (
                patch("machboost.models.backend_available", return_value=True),
                patch("machboost.models._validate_mlx_architecture") as validate,
            ):
                result = preflight_model(directory, "mlx")

        self.assertTrue(result["cached"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["model_type"], "qwen2")
        validate.assert_called_once()

    def test_preflight_exposes_unsupported_architecture_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps({"model_type": "future_model"}),
                encoding="utf-8",
            )
            with (
                patch("machboost.models.backend_available", return_value=True),
                patch(
                    "machboost.models._validate_mlx_architecture",
                    side_effect=ValueError("Model type future_model not supported"),
                ),
            ):
                result = preflight_model(directory, "mlx")

        self.assertFalse(result["supported"])
        self.assertIn("not supported", result["reason"])

    def test_cached_repo_path_returns_snapshot_with_config(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory, "snapshots", "revision")
            snapshot.mkdir(parents=True)
            Path(snapshot, "config.json").write_text("{}", encoding="utf-8")
            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = lambda **_: str(snapshot)
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                result = cached_repo_path("organization/model")

        self.assertEqual(result, snapshot.resolve())

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

    def test_sub_10b_qwen_vision_aliases_use_mlx_vlm(self):
        expected = {
            "qwen3-vl:2b": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "qwen3-vl:4b": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            "qwen3-vl:8b": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "qwen3.5:0.8b": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "qwen3.5:4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "qwen3.5:9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
        }

        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            for alias, model in expected.items():
                with self.subTest(alias=alias):
                    resolution = resolve_model(alias)
                    self.assertEqual(resolution.backend, "mlx-vlm")
                    self.assertEqual(resolution.model, model)


if __name__ == "__main__":
    unittest.main()
