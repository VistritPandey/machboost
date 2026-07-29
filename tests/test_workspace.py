import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from machboost.workspace import WorkspaceError, WorkspaceStore


class WorkspaceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.store = WorkspaceStore(self.root / "indexes")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_register_index_and_query_repository(self) -> None:
        (self.repo / "auth.py").write_text(
            "def authenticate_user(token):\n"
            "    payload = decode_token(token)\n"
            "    return lookup_user(payload['sub'])\n",
            encoding="utf-8",
        )
        (self.repo / "routes.swift").write_text(
            "struct LoginRoute {\n"
            "    func handleLogin() async {}\n"
            "}\n",
            encoding="utf-8",
        )

        workspace = self.store.register(self.repo, name="Authentication Service")
        report = self.store.index(workspace.id)
        result = self.store.query(workspace.id, "Where is authenticate_user handled?")

        self.assertEqual(workspace.id, self.store.register(self.repo).id)
        self.assertEqual(report.workspace.name, "Authentication Service")
        self.assertEqual(report.workspace.file_count, 2)
        self.assertEqual(report.indexed_files, 2)
        self.assertEqual(result.hits[0].path, "auth.py")
        self.assertIn("authenticate_user", result.hits[0].symbols)
        self.assertIn("auth.py:1-3", result.context)
        self.assertIn("Cite evidence as path:start-end", result.context)

    def test_git_index_respects_ignores_and_skips_likely_secrets(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / ".gitignore").write_text(
            "ignored/\n*.generated\n",
            encoding="utf-8",
        )
        (self.repo / "visible.py").write_text("VISIBLE_TOKEN = 1\n", encoding="utf-8")
        (self.repo / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
        (self.repo / "private.pem").write_text("secret-key\n", encoding="utf-8")
        (self.repo / "ignored").mkdir()
        (self.repo / "ignored" / "hidden.py").write_text(
            "HIDDEN_TOKEN = 1\n",
            encoding="utf-8",
        )
        (self.repo / "output.generated").write_text(
            "GENERATED_TOKEN = 1\n",
            encoding="utf-8",
        )
        (self.repo / "binary.dat").write_bytes(b"\0\1\2")

        workspace = self.store.register(self.repo)
        report = self.store.index(workspace.id)

        self.assertEqual(report.workspace.file_count, 2)
        self.assertEqual(
            {hit.path for hit in self.store.query(workspace.id, "VISIBLE_TOKEN").hits},
            {"visible.py"},
        )
        self.assertEqual(self.store.query(workspace.id, "secret API_KEY").hits, ())
        self.assertEqual(self.store.query(workspace.id, "HIDDEN_TOKEN").hits, ())
        self.assertEqual(self.store.query(workspace.id, "GENERATED_TOKEN").hits, ())

    def test_reindex_updates_changed_files_and_removes_deleted_files(self) -> None:
        source = self.repo / "service.py"
        removed = self.repo / "old.py"
        source.write_text("def old_name():\n    return 1\n", encoding="utf-8")
        removed.write_text("REMOVED_SENTINEL = True\n", encoding="utf-8")
        workspace = self.store.register(self.repo)
        first = self.store.index(workspace.id)

        source.write_text(
            "def replacement_handler():\n    return 2\n",
            encoding="utf-8",
        )
        removed.unlink()
        second = self.store.index(workspace.id)

        self.assertEqual(first.indexed_files, 2)
        self.assertEqual(second.indexed_files, 1)
        self.assertEqual(second.removed_files, 1)
        self.assertEqual(second.workspace.file_count, 1)
        self.assertEqual(
            self.store.query(workspace.id, "replacement_handler").hits[0].path,
            "service.py",
        )
        self.assertEqual(
            self.store.query(workspace.id, "REMOVED_SENTINEL").hits,
            (),
        )

    def test_query_limits_context_and_chunks_per_file(self) -> None:
        repeated = "\n".join(
            f"def shared_symbol_{index}():\n    return 'needle value {index}'"
            for index in range(400)
        )
        (self.repo / "large.py").write_text(repeated, encoding="utf-8")
        (self.repo / "small.py").write_text(
            "def needle_entrypoint():\n    return 'needle value'\n",
            encoding="utf-8",
        )
        workspace = self.store.register(self.repo)
        self.store.index(workspace.id)

        result = self.store.query(
            workspace.id,
            "needle value entrypoint",
            top_k=8,
            max_chars=1_200,
        )

        self.assertLessEqual(len(result.context), 1_200)
        self.assertLessEqual(len(result.hits), 8)
        paths = [hit.path for hit in result.hits]
        self.assertLessEqual(paths.count("large.py"), 3)
        self.assertTrue(result.truncated)

    def test_repository_map_is_stable_across_different_queries(self) -> None:
        (self.repo / "auth.py").write_text(
            "def authenticate_user(token):\n    return token\n",
            encoding="utf-8",
        )
        (self.repo / "billing.py").write_text(
            "def capture_payment(invoice):\n    return invoice\n",
            encoding="utf-8",
        )
        workspace = self.store.register(self.repo)
        self.store.index(workspace.id)

        auth = self.store.query(workspace.id, "authenticate user")
        billing = self.store.query(workspace.id, "capture payment")
        auth_prefix = auth.context.split("\n\n## ", maxsplit=1)[0]
        billing_prefix = billing.context.split("\n\n## ", maxsplit=1)[0]

        self.assertEqual(auth_prefix, billing_prefix)
        self.assertIn("auth.py: authenticate_user", auth_prefix)
        self.assertIn("billing.py: capture_payment", auth_prefix)
        self.assertNotEqual(auth.hits[0].path, billing.hits[0].path)

    def test_metadata_schema_round_trips_as_json(self) -> None:
        (self.repo / "main.go").write_text(
            "package main\nfunc main() {}\n",
            encoding="utf-8",
        )
        workspace = self.store.register(self.repo)
        report = self.store.index(workspace.id)

        payload = json.loads(json.dumps(report.to_dict()))
        restored = self.store.get(workspace.id)

        self.assertEqual(
            set(payload),
            {
                "workspace",
                "scanned_files",
                "indexed_files",
                "unchanged_files",
                "removed_files",
                "skipped_files",
            },
        )
        self.assertEqual(restored.to_dict(), payload["workspace"])
        self.assertEqual(payload["workspace"]["languages"], [{"name": "Go", "files": 1}])

    def test_remove_deletes_only_the_index(self) -> None:
        source = self.repo / "main.py"
        source.write_text("print('still here')\n", encoding="utf-8")
        workspace = self.store.register(self.repo)
        self.store.index(workspace.id)

        self.assertTrue(self.store.remove(workspace.id))
        self.assertTrue(source.exists())
        self.assertFalse(self.store.remove(workspace.id))
        with self.assertRaises(WorkspaceError):
            self.store.get(workspace.id)


if __name__ == "__main__":
    unittest.main()
