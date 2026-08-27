from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from machboost.claude_desktop import (
    CLAUDE_DESKTOP_ROUTES,
    PROFILE_ID,
    ClaudeDesktopProfileManager,
    claude_desktop_models,
    estimate_anthropic_input_tokens,
    normalize_gateway_endpoint,
    resolve_claude_desktop_model,
    save_model_mappings,
)


class ClaudeDesktopGatewayTests(unittest.TestCase):
    def test_models_use_claude_routes_and_machboost_display_names(self):
        models = claude_desktop_models(
            ["mlx-community/Muse-Glimmer-30B-4bit", "team/code-model"],
            configured={},
        )

        self.assertEqual(models[0]["id"], CLAUDE_DESKTOP_ROUTES[0].id)
        self.assertEqual(
            models[0]["display_name"],
            "mlx-community/Muse-Glimmer-30B-4bit",
        )
        self.assertEqual(models[0]["type"], "model")
        self.assertEqual(models[0]["anthropic_family_tier"], "fable")
        self.assertEqual(models[1]["display_name"], "team/code-model")

    def test_configured_routes_are_stable_and_missing_models_fall_back(self):
        configured = {
            CLAUDE_DESKTOP_ROUTES[0].id: "missing/model",
            CLAUDE_DESKTOP_ROUTES[1].id: "available/model",
        }
        models = claude_desktop_models(
            ["available/model", "fallback/model"],
            configured=configured,
        )

        by_id = {model["id"]: model["display_name"] for model in models}
        self.assertEqual(by_id[CLAUDE_DESKTOP_ROUTES[1].id], "available/model")
        self.assertEqual(by_id[CLAUDE_DESKTOP_ROUTES[0].id], "fallback/model")

    def test_route_resolution_keeps_direct_model_ids_unchanged(self):
        available = ["mlx-community/model-a"]
        route = CLAUDE_DESKTOP_ROUTES[0].id

        self.assertEqual(
            resolve_claude_desktop_model(route, available, configured={}),
            "mlx-community/model-a",
        )
        self.assertEqual(
            resolve_claude_desktop_model("custom/model", available, configured={}),
            "custom/model",
        )

    def test_token_estimate_counts_text_tools_and_images(self):
        text_only = estimate_anthropic_input_tokens(
            {"messages": [{"role": "user", "content": "hello world"}]}
        )
        multimodal = estimate_anthropic_input_tokens(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello world"},
                            {"type": "image", "source": {"type": "base64", "data": "abc"}},
                        ],
                    }
                ],
                "tools": [{"name": "read_file", "description": "Read a file"}],
            }
        )

        self.assertGreater(text_only, 0)
        self.assertGreater(multimodal, text_only + 1_500)

    def test_model_mapping_file_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            mappings = save_model_mappings(["model-a", "model-b"], path)

            self.assertEqual(mappings[CLAUDE_DESKTOP_ROUTES[0].id], "model-a")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_gateway_endpoint_rejects_api_paths_and_credentials(self):
        self.assertEqual(
            normalize_gateway_endpoint("192.168.1.20:11435"),
            "http://192.168.1.20:11435",
        )
        for value in ("http://user:pass@host:11435", "http://host:11435/v1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_gateway_endpoint(value)


class ClaudeDesktopProfileTests(unittest.TestCase):
    def test_restart_uses_app_path_then_bundle_identifier_fallback(self):
        manager = ClaudeDesktopProfileManager()
        app = Path("/Applications/Claude.app")
        results = [
            Mock(returncode=0),
            Mock(returncode=1),
            Mock(returncode=1, stderr="path launch failed"),
            Mock(returncode=0, stderr=""),
        ]

        with (
            patch.object(manager, "installed_application", return_value=app),
            patch("machboost.claude_desktop.subprocess.run", side_effect=results) as run,
            patch("machboost.claude_desktop.time.sleep"),
        ):
            manager.restart_application()

        self.assertEqual(run.call_args_list[2].args[0], ["/usr/bin/open", str(app)])
        self.assertEqual(
            run.call_args_list[3].args[0],
            ["/usr/bin/open", "-b", "com.anthropic.claudefordesktop"],
        )

    def test_configure_and_restore_preserve_previous_provider(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "machboost.claude_desktop.platform.system", return_value="Darwin"
        ):
            root = Path(directory)
            support = root / "Library" / "Application Support"
            normal = support / "Claude" / "claude_desktop_config.json"
            third_party = support / "Claude-3p"
            third_config = third_party / "claude_desktop_config.json"
            meta = third_party / "configLibrary" / "_meta.json"
            previous_profile = third_party / "configLibrary" / "ollama.json"
            for path, value in (
                (normal, {"deploymentMode": "3p", "keep": True}),
                (third_config, {"deploymentMode": "3p"}),
                (
                    meta,
                    {
                        "appliedId": "ollama",
                        "entries": [{"id": "ollama", "name": "Ollama"}],
                    },
                ),
                (previous_profile, {"inferenceGatewayBaseUrl": "http://127.0.0.1:11435"}),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            manager = ClaudeDesktopProfileManager(
                home=root,
                application_support=support,
                state_path=root / ".machboost" / "profile-state.json",
            )
            status = manager.configure("http://192.168.1.5:11435", "team-key")
            machboost_profile = third_party / "configLibrary" / f"{PROFILE_ID}.json"

            self.assertTrue(status["connected"])
            self.assertEqual(status["endpoint"], "http://192.168.1.5:11435")
            self.assertEqual(json.loads(meta.read_text())["appliedId"], PROFILE_ID)
            self.assertEqual(
                json.loads(machboost_profile.read_text())["inferenceGatewayApiKey"],
                "team-key",
            )

            restored = manager.restore()

            self.assertFalse(restored["connected"])
            self.assertEqual(json.loads(meta.read_text())["appliedId"], "ollama")
            self.assertEqual(json.loads(normal.read_text())["deploymentMode"], "3p")
            self.assertTrue(json.loads(normal.read_text())["keep"])
            self.assertFalse(machboost_profile.exists())
            self.assertTrue(previous_profile.exists())


if __name__ == "__main__":
    unittest.main()
