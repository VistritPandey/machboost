from __future__ import annotations

import unittest

from machboost.protocols import anthropic_messages, select_anthropic_tools


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

    def test_follow_up_queries_keep_the_initial_conversation_tool_prefix(self):
        tools = [tool(f"unrelated_{index}") for index in range(12)]
        tools.extend(
            [
                tool("read_file", "Read repository files"),
                tool("revenue_report", "Fetch revenue data"),
            ]
        )
        initial = [{"role": "user", "content": "Inspect the repository files"}]
        follow_up = [
            *initial,
            {"role": "assistant", "content": "The repository is available."},
            {"role": "user", "content": "Now fetch the revenue report"},
        ]

        initial_names = [
            item["name"] for item in select_anthropic_tools(tools, initial, limit=4)
        ]
        follow_up_names = [
            item["name"] for item in select_anthropic_tools(tools, follow_up, limit=4)
        ]

        self.assertEqual(follow_up_names, initial_names)
        self.assertIn("read_file", initial_names)


class AnthropicMessageTests(unittest.TestCase):
    def test_prior_thinking_is_accepted_without_replaying_private_reasoning(self):
        messages = anthropic_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "I should inspect the repository first.",
                                "signature": "signed-reasoning",
                            },
                            {"type": "text", "text": "I will inspect the repository."},
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "list_files",
                                "input": {"path": "."},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": "README.md\nsrc",
                            }
                        ],
                    },
                ]
            }
        )

        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "I will inspect the repository."}])
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "list_files")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertNotIn("inspect the repository first", str(messages))

    def test_redacted_thinking_block_is_accepted(self):
        messages = anthropic_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "data": "opaque"},
                            {"type": "text", "text": "Done."},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(messages, [{"role": "assistant", "content": [{"type": "text", "text": "Done."}]}])


if __name__ == "__main__":
    unittest.main()
