import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from machboost.connections import (
    ConnectionStore,
    connection_token_environment_name,
    is_loopback_endpoint,
    normalize_endpoint,
)


class ConnectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "connections.json"
        self.store = ConnectionStore(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    @patch("machboost.connections.set_connection_secret")
    def test_save_select_and_remove_profile_without_persisting_token(self, save_secret):
        profile = self.store.save(
            "studio",
            "192.168.1.44:11435/v1",
            api_token="mbk_secret",
        )

        self.assertEqual(profile.endpoint, "http://192.168.1.44:11435")
        self.assertEqual(self.store.active(), profile)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("mbk_secret", self.path.read_text(encoding="utf-8"))
        save_secret.assert_called_once_with(profile.id, "mbk_secret")

        self.assertIsNone(self.store.select("local"))
        self.assertIsNone(self.store.active())
        self.assertEqual(self.store.select("studio"), profile)

        with patch("machboost.connections.delete_connection_secret") as delete_secret:
            self.assertEqual(self.store.remove("studio"), profile)
            delete_secret.assert_called_once_with(profile.id)
        self.assertEqual(self.store.list(), [])

    def test_profile_file_has_stable_machine_readable_schema(self):
        with patch("machboost.connections.set_connection_secret"):
            profile = self.store.save("host-a", "https://inference.example.com")
        payload = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "machboost.connections.v2")
        self.assertEqual(payload["active"], profile.id)
        self.assertEqual(payload["profiles"][0]["name"], "host-a")

    def test_auto_mode_uses_every_saved_host_without_changing_fixed_mode(self):
        with patch("machboost.connections.set_connection_secret"):
            first = self.store.save("studio", "192.168.1.10:11435")
            self.store.save("laptop", "192.168.1.11:11435")

        self.assertEqual(self.store.mode(), "fixed")
        self.assertNotEqual(self.store.active(), first)
        self.assertIsNone(self.store.select("auto"))
        self.assertEqual(self.store.mode(), "auto")
        self.assertIsNone(self.store.active())

        selected = self.store.select("studio")
        self.assertEqual(selected, first)
        self.assertEqual(self.store.mode(), "fixed")

    def test_connection_token_supports_cross_platform_environment_secret(self):
        with patch("machboost.connections.set_connection_secret"):
            profile = self.store.save("build-studio", "192.168.1.10:11435")
        environment_name = connection_token_environment_name(profile.name)

        with patch.dict("os.environ", {environment_name: "mbk_environment"}):
            self.assertEqual(self.store.token(profile), "mbk_environment")

    def test_endpoint_validation_rejects_self_and_api_subpaths(self):
        self.assertTrue(is_loopback_endpoint("http://127.0.0.1:11435"))
        self.assertEqual(
            normalize_endpoint("https://host.example.com:11435/v1"),
            "https://host.example.com:11435",
        )
        with self.assertRaisesRegex(ValueError, "local server"):
            self.store.save("self", "localhost:11435")
        with self.assertRaisesRegex(ValueError, "server root"):
            normalize_endpoint("https://host.example.com/api/chat")


if __name__ == "__main__":
    unittest.main()
