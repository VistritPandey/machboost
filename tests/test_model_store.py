import tempfile
import unittest
from pathlib import Path

from machboost.model_store import ModelStore, apply_stored_model


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ModelStore(
            Path(self.temporary.name) / "models.sqlite3", clock=lambda: 100.0
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_create_resolve_update_copy_and_delete(self):
        created = self.store.create(
            "company-coder:latest",
            "qwen2.5-coder:7b",
            system="Follow repository conventions.",
            template="{{ .Prompt }}",
            options={"num_ctx": 8192, "temperature": 0},
        )
        resolved, stored = self.store.resolve("company-coder:latest")
        copied = self.store.copy("company-coder:latest", "company-coder:backup")

        self.assertEqual(resolved, "qwen2.5-coder:7b")
        self.assertEqual(stored, created)
        self.assertEqual(copied.options["num_ctx"], 8192)
        self.assertEqual(len(self.store.list()), 2)
        self.assertTrue(self.store.delete("company-coder:backup"))
        self.assertFalse(self.store.delete("company-coder:backup"))

    def test_request_options_override_stored_defaults(self):
        model = self.store.create(
            "coder",
            "source",
            system="stored system",
            options={"temperature": 0.2, "num_ctx": 4096},
        )
        merged = apply_stored_model(model, {"temperature": 0.8, "_system": "request"})

        self.assertEqual(merged["temperature"], 0.8)
        self.assertEqual(merged["num_ctx"], 4096)
        self.assertEqual(merged["_system"], "request")

    def test_invalid_names_and_non_json_options_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create("bad name", "source")
        with self.assertRaises(ValueError):
            self.store.create("name", "")
        with self.assertRaises(TypeError):
            self.store.create("name", "source", options={"bad": object()})


if __name__ == "__main__":
    unittest.main()
