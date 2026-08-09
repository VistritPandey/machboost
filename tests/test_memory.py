import tempfile
import unittest
from pathlib import Path

from machboost.memory import CacheNamespace, TeamMemoryStore, exchange_memory


class MutableClock:
    def __init__(self, value=1_000.0):
        self.value = value

    def __call__(self):
        return self.value


class TeamMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.clock = MutableClock()
        self.store = TeamMemoryStore(
            Path(self.tempdir.name) / "team.sqlite3",
            clock=self.clock,
            max_storage_bytes=8_192,
        )
        self.private = CacheNamespace("acme", "repo", "r1", "private", "alice")
        self.team = CacheNamespace("acme", "repo", "r1", "team")

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def put(self, namespace=None, **overrides):
        values = {
            "namespace": (namespace or self.private).key,
            "workspace_id": "repo",
            "scope": "private",
            "principal_id": "alice",
            "kind": "fix",
            "title": "Fix parser crash",
            "content": "The parser must retain UTF-8 input.",
            "query_text": "parser crash unicode",
            "revision": "r1",
            "dependencies": {"src/parser.py": "digest-1"},
            "evidence": ["tests/test_parser.py"],
        }
        values.update(overrides)
        return self.store.put(**values)

    def test_namespace_keys_isolate_scope_principal_and_revision(self):
        self.assertNotEqual(self.private.key, self.team.key)
        self.assertNotEqual(
            self.private.key,
            CacheNamespace("acme", "repo", "r1", "private", "bob").key,
        )
        self.assertNotEqual(
            self.private.key,
            CacheNamespace("acme", "repo", "r2", "private", "alice").key,
        )
        with self.assertRaises(ValueError):
            CacheNamespace("acme", "repo", "r1", "private")

    def test_put_deduplicates_and_redacts_secrets(self):
        first = self.put(content="Use api_key=abc and Authorization: Bearer xyz")
        second = self.put(content="Use api_key=abc and Authorization: Bearer xyz")

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.list(admin=True)), 1)
        self.assertNotIn("abc", second.content)
        self.assertNotIn("xyz", second.content)
        self.assertIn("[REDACTED]", second.content)

    def test_private_and_team_visibility(self):
        private = self.put()
        team = self.put(
            namespace=self.team,
            scope="team",
            principal_id=None,
            title="Shared build fix",
            content="Regenerate the lockfile.",
            query_text="build lockfile",
            dependencies={},
        )

        self.assertEqual(
            [record.id for record in self.store.list(principal_id="bob")], [team.id]
        )
        with self.assertRaises(KeyError):
            self.store.get(private.id, principal_id="bob")
        self.assertEqual(self.store.get(team.id, principal_id="bob").id, team.id)

    def test_dependency_digest_and_revision_invalidate_results(self):
        self.put()
        valid = self.store.search(
            namespace=self.private.key,
            workspace_id="repo",
            query="parser crash",
            revision="r2",
            dependency_digests={"src/parser.py": "digest-1"},
            principal_id="alice",
        )
        stale = self.store.search(
            namespace=self.private.key,
            workspace_id="repo",
            query="parser crash",
            revision="r2",
            dependency_digests={"src/parser.py": "digest-2"},
            principal_id="alice",
        )

        self.assertEqual(len(valid.records), 1)
        self.assertEqual(stale.records, ())
        self.assertEqual(stale.stale_rejected, 1)

    def test_validation_confidence_and_pinning_affect_rank(self):
        self.put(title="Weak parser note", confidence=0.1, dependencies={})
        strong = self.put(
            title="Validated parser procedure",
            content="Run the parser regression suite.",
            confidence=0.95,
            validated_by=["ci", "review"],
            pinned=True,
            dependencies={},
        )
        result = self.store.search(
            namespace=self.private.key,
            workspace_id="repo",
            query="parser",
            revision="r1",
            principal_id="alice",
        )

        self.assertEqual(result.records[0].id, strong.id)

    def test_context_budget_is_enforced(self):
        self.put(content="parser " * 300, dependencies={})
        result = self.store.search(
            namespace=self.private.key,
            workspace_id="repo",
            query="parser",
            revision="r1",
            principal_id="alice",
            max_chars=220,
        )

        self.assertLessEqual(len(result.context), 220)
        self.assertTrue(result.truncated)

    def test_expired_memory_and_exact_cache_are_removed(self):
        self.put(ttl_seconds=10)
        self.store.put_exact(
            namespace=self.private.key,
            workspace_id="repo",
            revision="r1",
            model="model",
            request={"prompt": "hello"},
            response={"response": "hi"},
            ttl_seconds=10,
        )
        self.clock.value += 11

        self.assertEqual(self.store.prune_expired(), 2)
        self.assertEqual(self.store.list(admin=True), [])
        self.assertIsNone(
            self.store.get_exact(
                namespace=self.private.key,
                workspace_id="repo",
                revision="r1",
                model="model",
                request={"prompt": "hello"},
            )
        )

    def test_exact_cache_is_revision_scoped_and_tracks_savings(self):
        request = {"prompt": "hello", "temperature": 0}
        self.store.put_exact(
            namespace=self.private.key,
            workspace_id="repo",
            revision="r1",
            model="model",
            request=request,
            response={"response": "hi"},
            prompt_tokens=12,
            completion_tokens=4,
            cost_usd=0.002,
        )

        hit = self.store.get_exact(
            namespace=self.private.key,
            workspace_id="repo",
            revision="r1",
            model="model",
            request=request,
        )
        miss = self.store.get_exact(
            namespace=self.private.key,
            workspace_id="repo",
            revision="r2",
            model="model",
            request=request,
        )
        totals = self.store.metrics()["totals"]

        self.assertEqual(hit["response"], {"response": "hi"})
        self.assertIsNone(miss)
        self.assertEqual(totals["avoided_prompt_tokens"], 12)
        self.assertEqual(totals["avoided_completion_tokens"], 4)
        self.assertEqual(totals["avoided_cost_microusd"], 2_000)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.put(kind="opinion")
        with self.assertRaises(ValueError):
            self.put(confidence=1.1)
        with self.assertRaises(ValueError):
            self.put(ttl_seconds=0)
        with self.assertRaises(ValueError):
            self.put(scope="private", principal_id=None)


class ExchangeMemoryTests(unittest.TestCase):
    def test_exchange_is_typed_bounded_and_redacted(self):
        result = exchange_memory(
            user_text="Fix the login error with token=private-value",
            assistant_text="Updated auth.py and tests passed. " * 100,
            evidence=["auth.py", "auth.py"],
            validated_by=["pytest"],
            max_chars=400,
        )

        self.assertEqual(result["kind"], "fix")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["evidence"], ["auth.py"])
        self.assertNotIn("private-value", result["content"])
        self.assertLessEqual(len(result["content"]), 400)


if __name__ == "__main__":
    unittest.main()
