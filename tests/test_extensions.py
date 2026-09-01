from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from machboost.extensions import ExtensionStore, MCPConnectorManager


class ExtensionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ExtensionStore(Path(self.temporary.name) / "extensions.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_connector_listing_redacts_environment_and_headers(self) -> None:
        server = self.store.configure_server(
            server_id=None,
            name="Issue tracker",
            transport="http",
            url="https://example.test/mcp",
            headers={"Authorization": "Bearer secret"},
            env={"PRIVATE_TOKEN": "secret"},
        )

        listed = server.to_dict()

        self.assertEqual(listed["header_names"], ["Authorization"])
        self.assertEqual(listed["env_keys"], ["PRIVATE_TOKEN"])
        self.assertNotIn("headers", listed)
        self.assertNotIn("env", listed)

    def test_stdio_connector_requires_a_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "require a command"):
            self.store.configure_server(
                server_id=None,
                name="Files",
                transport="stdio",
            )

    def test_connector_update_preserves_omitted_secrets(self) -> None:
        server = self.store.configure_server(
            server_id=None,
            name="Tracker",
            transport="http",
            url="https://example.test/mcp",
            headers={"Authorization": "Bearer secret"},
        )

        updated = self.store.configure_server(
            server_id=server.id,
            name="Renamed tracker",
            transport="http",
            url="https://example.test/mcp",
            headers=None,
        )

        self.assertEqual(updated.headers, {"Authorization": "Bearer secret"})

    def test_enabled_skills_form_a_stable_instruction_prompt(self) -> None:
        first = self.store.configure_skill(
            skill_id=None,
            name="Concise",
            instructions="Use short answers.",
        )
        disabled = self.store.configure_skill(
            skill_id=None,
            name="Verbose",
            instructions="Explain every detail.",
            enabled=False,
        )

        default_prompt = self.store.skill_prompt()
        selected_prompt = self.store.skill_prompt([disabled.id])

        self.assertIn(first.name, default_prompt or "")
        self.assertNotIn(disabled.name, default_prompt or "")
        self.assertIn(disabled.instructions, selected_prompt or "")


class FakeConnectorManager(MCPConnectorManager):
    async def _list_tools(self, server):
        return [
            {
                "server_id": server.id,
                "server_name": server.name,
                "name": "find_issue",
                "description": "Find an issue by title",
                "input_schema": {"type": "object"},
                "annotations": None,
            },
            {
                "server_id": server.id,
                "server_name": server.name,
                "name": "list_projects",
                "description": "List projects",
                "input_schema": {"type": "object"},
                "annotations": None,
            },
        ]


class MCPConnectorManagerTests(unittest.TestCase):
    def test_search_returns_compact_ranked_tool_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExtensionStore(Path(directory) / "extensions.sqlite3")
            server = store.configure_server(
                server_id=None,
                name="Tracker",
                transport="http",
                url="https://example.test/mcp",
            )
            manager = FakeConnectorManager(store)

            rows = manager.search_tools("find issue")

            self.assertEqual([row["name"] for row in rows], ["find_issue"])
            self.assertEqual(store.server(server.id).tool_count, 2)
            store.close()


@unittest.skipUnless(importlib.util.find_spec("mcp"), "MCP SDK is not installed")
class MCPConnectorIntegrationTests(unittest.TestCase):
    def test_stdio_server_lists_and_calls_a_real_mcp_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExtensionStore(Path(directory) / "extensions.sqlite3")
            try:
                server = store.configure_server(
                    server_id=None,
                    name="Echo fixture",
                    transport="stdio",
                    command=sys.executable,
                    args=[str(Path(__file__).parent / "fixtures" / "mcp_echo_server.py")],
                )
                manager = MCPConnectorManager(store)

                tools = manager.list_tools(server.id)
                result = manager.call_tool(
                    server.id,
                    "echo",
                    {"text": "connected"},
                )

                self.assertEqual([tool["name"] for tool in tools], ["echo"])
                self.assertFalse(result["is_error"])
                self.assertEqual(result["text"], "echo:connected")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
