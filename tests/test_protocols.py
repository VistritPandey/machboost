from __future__ import annotations

import unittest

from machboost.protocols import select_anthropic_tools


def tool(name: str, description: str = "") -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": {}},
    }


class AnthropicToolSelectionTests(unittest.TestCase):
    def test_keeps_core_and_query_relevant_tools(self):
        tools = [tool(f"unrelated_{index}", "Manage an unrelated resource") for index in range(70)]
        tools.extend(
            [
                tool("mcp__filesystem__read_file", "Read a repository file"),
                tool("get_revenue_report", "Fetch sponsorship revenue reports"),
            ]
        )

        selected = select_anthropic_tools(
            tools,
            [{"role": "user", "content": "Read the repo and inspect the revenue report"}],
            limit=8,
        )

        names = {item["name"] for item in selected}
        self.assertEqual(len(selected), 8)
        self.assertIn("mcp__filesystem__read_file", names)
        self.assertIn("get_revenue_report", names)

    def test_keeps_previously_used_and_explicitly_forced_tools(self):
        tools = [tool(f"tool_{index}") for index in range(60)]
        tools.extend([tool("prior_tool"), tool("forced_tool")])
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "prior_tool",
                        "input": {},
                    }
                ],
            },
            {"role": "user", "content": "Continue"},
        ]

        selected = select_anthropic_tools(
            tools,
            messages,
            tool_choice={"type": "tool", "name": "forced_tool"},
            limit=4,
        )

        names = {item["name"] for item in selected}
        self.assertEqual(len(selected), 4)
        self.assertIn("prior_tool", names)
        self.assertIn("forced_tool", names)

    def test_small_tool_sets_are_unchanged(self):
        tools = [tool("read_file"), tool("write_file")]

        self.assertEqual(select_anthropic_tools(tools, [], limit=48), tools)


if __name__ == "__main__":
    unittest.main()
