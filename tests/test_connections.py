import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from machboost.connections import ConnectionStore, is_loopback_endpoint, normalize_endpoint


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

        self.assertEqual(payload["schema"], "machboost.connections.v1")
        self.assertEqual(payload["active"], profile.id)
        self.assertEqual(payload["profiles"][0]["name"], "host-a")

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
