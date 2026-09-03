import tempfile
import unittest
from pathlib import Path

from machboost.coding import CodingWorkspace, coding_system_prompt, coding_tools


def call(name, arguments=None, call_id="call_1"):
    return {
        "id": call_id,
        "function": {"name": name, "arguments": arguments or {}},
    }


class CodingWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "def greet():\n    return 'hello'\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_tool_catalog_has_bounded_file_and_shell_tools(self):
        names = [item["function"]["name"] for item in coding_tools()]

        self.assertEqual(
            names,
            [
                "list_files",
                "read_file",
                "search_code",
                "replace_in_file",
                "create_file",
                "delete_file",
                "run_command",
                "git_diff",
            ],
        )

    def test_read_list_and_search_are_allowed_in_manual_mode(self):
        workspace = CodingWorkspace(self.root)

        listed = workspace.execute(call("list_files", {"path": "."}))
        read = workspace.execute(call("read_file", {"path": "src/app.py"}))
        searched = workspace.execute(call("search_code", {"query": "greet"}))

        self.assertEqual(listed.status, "done")
        self.assertIn("src/app.py", listed.content)
        self.assertIn("return 'hello'", read.content)
        self.assertIn("src/app.py:1", searched.content)

    def test_manual_mode_prompts_before_an_edit(self):
        workspace = CodingWorkspace(self.root, permission_mode="manual")
        prompts = []
        tool_call = call(
            "replace_in_file",
            {"path": "src/app.py", "old_text": "hello", "new_text": "hi"},
        )

        denied = workspace.execute(tool_call, confirm=lambda prompt: prompts.append(prompt) or False)
        accepted = workspace.execute(tool_call, confirm=lambda _prompt: True)

        self.assertEqual(denied.status, "denied")
        self.assertEqual(prompts, ["Edit src/app.py"])
        self.assertEqual(accepted.status, "done")
        self.assertEqual(accepted.changed_path, "src/app.py")
        self.assertIn("hi", (self.root / "src" / "app.py").read_text(encoding="utf-8"))

    def test_accept_edits_allows_files_but_prompts_for_shell(self):
        workspace = CodingWorkspace(self.root, permission_mode="accept-edits")

        created = workspace.execute(
            call("create_file", {"path": "src/new.py", "content": "value = 1\n"})
        )
        command = workspace.execute(
            call("run_command", {"command": "printf ok"}),
            confirm=lambda _prompt: False,
        )

        self.assertEqual(created.status, "done")
        self.assertEqual(command.status, "denied")

    def test_plan_mode_denies_edits_and_commands(self):
        workspace = CodingWorkspace(self.root, permission_mode="plan")

        edit = workspace.execute(
            call("create_file", {"path": "nope.py", "content": ""}),
            confirm=lambda _prompt: True,
        )
        command = workspace.execute(
            call("run_command", {"command": "touch nope.py"}),
            confirm=lambda _prompt: True,
        )

        self.assertEqual(edit.status, "denied")
        self.assertEqual(command.status, "denied")
        self.assertFalse((self.root / "nope.py").exists())

    def test_bypass_runs_commands_and_delete_removes_files(self):
        workspace = CodingWorkspace(self.root, permission_mode="bypass")

        command = workspace.execute(call("run_command", {"command": "printf ready"}))
        deleted = workspace.execute(call("delete_file", {"path": "src/app.py"}))

        self.assertEqual(command.status, "done")
        self.assertIn("exit_code=0\nready", command.content)
        self.assertEqual(deleted.status, "done")
        self.assertFalse((self.root / "src" / "app.py").exists())

    def test_paths_cannot_escape_workspace_or_modify_git_metadata(self):
        workspace = CodingWorkspace(self.root, permission_mode="bypass")
        (self.root / ".git").mkdir()

        escaped = workspace.execute(call("read_file", {"path": "../outside.txt"}))
        metadata = workspace.execute(
            call("create_file", {"path": ".git/config", "content": "bad"})
        )

        self.assertEqual(escaped.status, "error")
        self.assertIn("outside", escaped.content)
        self.assertEqual(metadata.status, "error")
        self.assertIn(".git", metadata.content)

    def test_exact_replacement_rejects_ambiguous_matches(self):
        path = self.root / "src" / "app.py"
        path.write_text("same\nsame\n", encoding="utf-8")
        workspace = CodingWorkspace(self.root, permission_mode="bypass")

        result = workspace.execute(
            call(
                "replace_in_file",
                {"path": "src/app.py", "old_text": "same", "new_text": "new"},
            )
        )

        self.assertEqual(result.status, "error")
        self.assertIn("found 2 matches", result.content)

    def test_invalid_tool_arguments_return_an_error_result(self):
        workspace = CodingWorkspace(self.root)
        result = workspace.execute(call("read_file", "{"))

        self.assertEqual(result.status, "error")
        self.assertIn("Invalid arguments", result.content)

    def test_system_prompt_names_real_workspace_and_permission_mode(self):
        prompt = coding_system_prompt(self.root, "plan")

        self.assertIn(str(self.root), prompt)
        self.assertIn("'plan' permissions", prompt)
        self.assertIn("Never claim a file changed", prompt)


if __name__ == "__main__":
    unittest.main()
