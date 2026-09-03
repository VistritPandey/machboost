from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Optional, Sequence


SKIP_DIRECTORIES = {
    ".git",
    ".machboost",
    ".next",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}
MAX_TOOL_OUTPUT = 40_000
MAX_FILE_BYTES = 1_000_000
PERMISSION_MODES = ("manual", "accept-edits", "plan", "bypass")


def coding_tools() -> list[dict[str, Any]]:
    return [
        _tool(
            "list_files",
            "List files and directories inside the workspace.",
            {
                "path": {"type": "string", "description": "Workspace-relative directory."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 8},
            },
        ),
        _tool(
            "read_file",
            "Read a UTF-8 workspace file with line numbers.",
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            required=("path",),
        ),
        _tool(
            "search_code",
            "Search workspace file contents using a regular expression.",
            {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string", "description": "Optional file glob such as *.py."},
            },
            required=("query",),
        ),
        _tool(
            "replace_in_file",
            "Replace one exact text block in an existing workspace file.",
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            required=("path", "old_text", "new_text"),
        ),
        _tool(
            "create_file",
            "Create a new UTF-8 file. It refuses to overwrite an existing file.",
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            required=("path", "content"),
        ),
        _tool(
            "delete_file",
            "Delete one workspace file.",
            {"path": {"type": "string"}},
            required=("path",),
        ),
        _tool(
            "run_command",
            "Run a shell command in the workspace and return stdout, stderr, and its exit code.",
            {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            required=("command",),
        ),
        _tool(
            "git_diff",
            "Show workspace Git status and the current unstaged diff.",
            {},
        ),
    ]


def coding_system_prompt(root: Path, permission_mode: str) -> str:
    return (
        "You are MachBoost Code, an interactive coding agent. "
        f"The workspace root is {root}. "
        "Use the provided tools to inspect real files instead of guessing. Read relevant code "
        "before editing, make focused changes, run appropriate checks, and continue until the "
        "request is complete. After a tool result, either call the next needed tool or give a "
        "concise final answer; never end with an empty response. Never claim a file changed "
        "unless a tool result confirms it. "
        "Use relative paths and never access files outside the workspace. "
        f"The runtime enforces {permission_mode!r} permissions. If an action is denied, explain "
        "what remains instead of repeatedly requesting it."
    )


@dataclass(frozen=True)
class ToolExecution:
    call_id: str
    name: str
    content: str
    status: str
    changed_path: Optional[str] = None

    def message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": self.content,
        }


class CodingWorkspace:
    def __init__(self, root: str | os.PathLike[str], *, permission_mode: str = "manual") -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"workspace is not a directory: {resolved}")
        if permission_mode not in PERMISSION_MODES:
            raise ValueError(f"unsupported permission mode: {permission_mode}")
        self.root = resolved
        self.permission_mode = permission_mode

    def execute(
        self,
        call: dict[str, Any],
        *,
        confirm: Optional[Callable[[str], bool]] = None,
    ) -> ToolExecution:
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "").strip()
        call_id = str(call.get("id") or f"call_{name or 'tool'}")
        try:
            arguments = _arguments(function.get("arguments"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return ToolExecution(call_id, name or "tool", f"Invalid arguments: {exc}", "error")
        canonical = {
            "bash": "run_command",
            "shell": "run_command",
            "read": "read_file",
            "grep": "search_code",
            "glob": "list_files",
            "edit": "replace_in_file",
            "write": "create_file",
        }.get(name, name)
        if canonical not in {item["function"]["name"] for item in coding_tools()}:
            return ToolExecution(call_id, name or "tool", f"Unknown tool: {name}", "error")

        approval = self._approval(canonical)
        description = self.describe(canonical, arguments)
        if approval == "deny":
            return ToolExecution(call_id, canonical, f"Denied by {self.permission_mode} mode: {description}", "denied")
        if approval == "ask" and (confirm is None or not confirm(description)):
            return ToolExecution(call_id, canonical, f"User declined: {description}", "denied")

        try:
            content, changed_path = self._dispatch(canonical, arguments)
        except Exception as exc:
            return ToolExecution(call_id, canonical, f"{type(exc).__name__}: {exc}", "error")
        return ToolExecution(call_id, canonical, content, "done", changed_path)

    def describe(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "run_command":
            return f"Run {str(arguments.get('command') or '')[:160]}"
        path = str(arguments.get("path") or ".")
        labels = {
            "list_files": "List",
            "read_file": "Read",
            "search_code": "Search",
            "replace_in_file": "Edit",
            "create_file": "Create",
            "delete_file": "Delete",
            "git_diff": "Inspect Git changes in",
        }
        return f"{labels.get(name, name)} {path}"

    def _approval(self, name: str) -> str:
        mutating = name in {"replace_in_file", "create_file", "delete_file"}
        command = name == "run_command"
        if self.permission_mode == "bypass":
            return "allow"
        if self.permission_mode == "plan":
            return "deny" if mutating or command else "allow"
        if self.permission_mode == "accept-edits":
            return "ask" if command else "allow"
        return "ask" if mutating or command else "allow"

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, Optional[str]]:
        if name == "list_files":
            return self._list_files(arguments), None
        if name == "read_file":
            return self._read_file(arguments), None
        if name == "search_code":
            return self._search_code(arguments), None
        if name == "replace_in_file":
            return self._replace_in_file(arguments)
        if name == "create_file":
            return self._create_file(arguments)
        if name == "delete_file":
            return self._delete_file(arguments)
        if name == "run_command":
            return self._run_command(arguments), None
        if name == "git_diff":
            return self.git_diff(), None
        raise ValueError(f"unsupported tool: {name}")

    def _list_files(self, arguments: dict[str, Any]) -> str:
        base = self._path(arguments.get("path") or ".", must_exist=True)
        if not base.is_dir():
            raise ValueError(f"not a directory: {self._relative(base)}")
        depth = max(1, min(8, int(arguments.get("depth") or 3)))
        rows: list[str] = []
        base_parts = len(base.parts)
        for current, directories, files in os.walk(base):
            current_path = Path(current)
            level = len(current_path.parts) - base_parts
            directories[:] = sorted(
                name for name in directories if name not in SKIP_DIRECTORIES
            )
            if level >= depth:
                directories[:] = []
            for name in directories:
                rows.append(self._relative(current_path / name) + "/")
            for name in sorted(files):
                rows.append(self._relative(current_path / name))
            if len(rows) >= 500:
                rows = rows[:500]
                rows.append("... output limited to 500 entries")
                break
        return "\n".join(rows) if rows else "Directory is empty."

    def _read_file(self, arguments: dict[str, Any]) -> str:
        path = self._path(_required(arguments, "path"), must_exist=True)
        if not path.is_file():
            raise ValueError(f"not a file: {self._relative(path)}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes")
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(1, int(arguments.get("start_line") or 1))
        end = min(len(lines), int(arguments.get("end_line") or start + 399))
        if end < start:
            raise ValueError("end_line must be greater than or equal to start_line")
        return "\n".join(
            f"{index:>6}  {lines[index - 1]}" for index in range(start, end + 1)
        ) or "File is empty."

    def _search_code(self, arguments: dict[str, Any]) -> str:
        query = _required(arguments, "query")
        base = self._path(arguments.get("path") or ".", must_exist=True)
        executable = shutil.which("rg")
        if executable is None:
            return self._search_code_python(query, base)
        command = [executable, "--line-number", "--color", "never", "--hidden"]
        for directory in sorted(SKIP_DIRECTORIES):
            command.extend(("--glob", f"!{directory}/**"))
        if arguments.get("glob"):
            command.extend(("--glob", str(arguments["glob"])))
        command.extend((query, str(base)))
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if result.returncode not in {0, 1}:
            raise RuntimeError((result.stderr or "search failed").strip())
        output = result.stdout.replace(str(self.root) + os.sep, "")
        return _limited(output.strip() or "No matches.")

    def _search_code_python(self, query: str, base: Path) -> str:
        import re

        pattern = re.compile(query)
        rows: list[str] = []
        files = [base] if base.is_file() else base.rglob("*")
        for path in files:
            if not path.is_file() or any(part in SKIP_DIRECTORIES for part in path.parts):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern.search(line):
                        rows.append(f"{self._relative(path)}:{index}:{line}")
            except (OSError, UnicodeDecodeError):
                continue
            if len(rows) >= 500:
                break
        return _limited("\n".join(rows) or "No matches.")

    def _replace_in_file(self, arguments: dict[str, Any]) -> tuple[str, str]:
        path = self._path(_required(arguments, "path"), must_exist=True, writable=True)
        old = _required(arguments, "old_text")
        new = str(arguments.get("new_text") or "")
        text = path.read_text(encoding="utf-8")
        matches = text.count(old)
        if matches != 1:
            raise ValueError(f"old_text must match exactly once; found {matches} matches")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        relative = self._relative(path)
        return f"Updated {relative}.", relative

    def _create_file(self, arguments: dict[str, Any]) -> tuple[str, str]:
        path = self._path(_required(arguments, "path"), writable=True)
        if path.exists():
            raise FileExistsError(f"file already exists: {self._relative(path)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments.get("content") or ""), encoding="utf-8")
        relative = self._relative(path)
        return f"Created {relative}.", relative

    def _delete_file(self, arguments: dict[str, Any]) -> tuple[str, str]:
        path = self._path(_required(arguments, "path"), must_exist=True, writable=True)
        if not path.is_file():
            raise ValueError("delete_file only removes files")
        relative = self._relative(path)
        path.unlink()
        return f"Deleted {relative}.", relative

    def _run_command(self, arguments: dict[str, Any]) -> str:
        command = _required(arguments, "command")
        timeout = max(1, min(300, int(arguments.get("timeout_seconds") or 120)))
        try:
            result = subprocess.run(
                ["/bin/zsh", "-lc", command],
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return _limited(f"Timed out after {timeout}s.\n{output}".strip())
        output = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        return _limited(f"exit_code={result.returncode}\n{output}".strip())

    def git_diff(self) -> str:
        if not (self.root / ".git").exists():
            return "The workspace is not a Git repository."
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        diff = subprocess.run(
            ["git", "diff", "--"],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        output = f"status:\n{status.stdout.strip() or '(clean)'}\n\ndiff:\n{diff.stdout.strip() or '(empty)'}"
        return _limited(output)

    def _path(self, value: Any, *, must_exist: bool = False, writable: bool = False) -> Path:
        raw = Path(str(value or ".")).expanduser()
        candidate = (self.root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("path is outside the selected workspace") from exc
        relative = candidate.relative_to(self.root)
        if writable and ".git" in relative.parts:
            raise PermissionError("writing inside .git is not allowed")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(self._relative(candidate))
        return candidate

    def _relative(self, path: Path) -> str:
        relative = path.relative_to(self.root)
        return "." if not relative.parts else str(relative)


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = list(required)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed
    return {}


def _required(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"missing required argument: {key}")
    return str(value)


def _limited(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + "\n... tool output truncated"
