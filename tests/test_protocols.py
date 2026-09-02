from __future__ import annotations

import unittest

from machboost.protocols import (
    anthropic_cache_affinity,
    anthropic_messages,
    claude_code_session_title,
    compact_claude_code_messages,
    compact_claude_code_tools,
    is_claude_code_request,
    select_anthropic_tools,
)


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
        self.assertLessEqual(len(selected), 8)
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

    def test_irrelevant_large_tools_do_not_fill_the_cap(self):
        tools = [
            tool("Bash", "Run a command"),
            tool("Read", "Read a file"),
            tool("Edit", "Edit a file"),
            tool("Write", "Write a file"),
            tool("Monitor", "M" * 12_000),
            tool("DesignSync", "D" * 12_000),
        ]

        selected = select_anthropic_tools(
            tools,
            [{"role": "user", "content": "Reply with exactly OK"}],
            limit=12,
        )

        self.assertEqual(
            [item["name"] for item in selected],
            ["Bash", "Read", "Edit", "Write"],
        )


class AnthropicCacheAffinityTests(unittest.TestCase):
    def test_separates_agent_and_utility_requests_in_one_claude_session(self):
        base = {
            "model": "claude-opus-5",
            "metadata": {
                "user_id": '{"device_id":"device-1","session_id":"session-42"}'
            },
            "messages": [{"role": "user", "content": "Inspect this repository"}],
        }

        utility = anthropic_cache_affinity(base, [])
        agent = anthropic_cache_affinity(base, [tool("Read")])

        self.assertNotEqual(agent, utility)
        self.assertEqual(
            agent,
            anthropic_cache_affinity(
                {
                    **base,
                    "messages": [
                        *base["messages"],
                        {"role": "assistant", "content": "I will inspect it."},
                    ],
                },
                [tool("Read")],
            ),
        )

    def test_fallback_affinity_is_stable_across_follow_ups(self):
        initial = {
            "model": "local-model",
            "system": "stable system",
            "messages": [{"role": "user", "content": "Initial request"}],
        }
        follow_up = {
            **initial,
            "messages": [
                *initial["messages"],
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Continue"},
            ],
        }

        self.assertEqual(
            anthropic_cache_affinity(initial, [tool("Read")]),
            anthropic_cache_affinity(follow_up, [tool("Read")]),
        )


class ClaudeCodeCompactionTests(unittest.TestCase):
    def test_detects_claude_code_from_harness_without_user_agent(self):
        payload = {
            "system": (
                "You are an interactive agent that helps users with software "
                "engineering tasks."
            )
        }

        self.assertTrue(is_claude_code_request(payload, "Python-urllib/3"))
        self.assertTrue(is_claude_code_request({}, "claude-cli/2.1.255"))
        self.assertFalse(
            is_claude_code_request(
                {"system": "You are a concise assistant."},
                "anthropic-sdk-python/1.0",
            )
        )

    def test_compacts_tool_prose_without_changing_schema_contract(self):
        verbose = "Detailed guidance that repeats client behavior. " * 80
        tools = [
            {
                "name": "Read",
                "description": "Reads a file from the local filesystem. " + verbose,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": verbose,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["text", "binary"],
                            "description": verbose,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
        ]

        compacted = compact_claude_code_tools(tools)

        self.assertEqual([item["name"] for item in compacted], ["Read"])
        schema = compacted[0]["input_schema"]
        self.assertEqual(schema["required"], ["path"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["text", "binary"])
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("repeats client behavior", str(compacted))
        self.assertLess(len(str(compacted)), len(str(tools)) // 4)

    def test_extracts_session_title_without_running_the_model(self):
        payload = {
            "system": (
                "You are naming a coding session. "
                'Return JSON with a single "title" field.'
            ),
            "messages": [
                {
                    "role": "user",
                    "content": "<session>Fix the doubled model search input</session>",
                }
            ],
        }

        self.assertEqual(
            claude_code_session_title(payload),
            "Fix doubled model search",
        )

    def test_ignores_normal_anthropic_requests(self):
        self.assertIsNone(
            claude_code_session_title(
                {
                    "system": "You are a coding assistant.",
                    "messages": [{"role": "user", "content": "Fix search"}],
                }
            )
        )

    def test_compacts_harness_boilerplate_and_drops_unavailable_agent_catalog(self):
        messages = [
            {
                "role": "system",
                "content": """
You are an interactive agent that helps users with software engineering tasks.

# Harness
Long generic harness instructions.

# Environment
 - Primary working directory: /tmp/project
 - Is a git repository: true
 - Platform: darwin
 - Shell: zsh

# Corrections
Long correction instructions.
""",
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Available agent types for the Agent tool:\n- Explore\n\nThe following skills are available for use with the Skill tool:\n- docs",
                    }
                ],
            },
            {"role": "user", "content": "Inspect the repository"},
        ]

        compacted = compact_claude_code_messages(
            messages,
            selected_tool_names={"Bash", "Read", "Edit", "Write"},
        )

        self.assertEqual(len(compacted), 2)
        system = str(compacted[0]["content"])
        self.assertIn("interactive coding agent", system)
        self.assertIn("Primary working directory: /tmp/project", system)
        self.assertNotIn("Long correction instructions", system)
        self.assertEqual(compacted[-1], messages[-1])


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
