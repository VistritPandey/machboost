from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(value: Any, *, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return {
        str(key): str(item)
        for key, item in value.items()
        if str(key).strip() and item is not None
    }


def _json_array(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class MCPServerConfig:
    id: str
    name: str
    transport: str
    url: Optional[str]
    command: Optional[str]
    args: tuple[str, ...]
    env: dict[str, str]
    headers: dict[str, str]
    enabled: bool
    tool_count: int
    last_status: Optional[str]
    last_error: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self, *, include_secrets: bool = False) -> dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "transport": self.transport,
            "url": self.url,
            "command": self.command,
            "args": list(self.args),
            "enabled": self.enabled,
            "tool_count": self.tool_count,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "env_keys": sorted(self.env),
            "header_names": sorted(self.headers),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_secrets:
            result["env"] = dict(self.env)
            result["headers"] = dict(self.headers)
        return result


@dataclass(frozen=True)
class SkillConfig:
    id: str
    name: str
    instructions: str
    enabled: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "instructions": self.instructions,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ExtensionStore:
    """Local MCP connector and reusable instruction persistence."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._migrate()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _migrate(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    transport TEXT NOT NULL,
                    url TEXT,
                    command TEXT,
                    args_json TEXT NOT NULL,
                    env_json TEXT NOT NULL,
                    headers_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    tool_count INTEGER NOT NULL DEFAULT 0,
                    last_status TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    instructions TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def list_servers(self, *, enabled_only: bool = False) -> list[MCPServerConfig]:
        query = "SELECT * FROM mcp_servers"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY name COLLATE NOCASE"
        with self._lock:
            rows = self._connection.execute(query).fetchall()
        return [self._server(row) for row in rows]

    def server(self, server_id: str) -> Optional[MCPServerConfig]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM mcp_servers WHERE id = ?", (server_id,)
            ).fetchone()
        return None if row is None else self._server(row)

    def configure_server(
        self,
        *,
        server_id: Optional[str],
        name: str,
        transport: str,
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: Any = None,
        env: Any = None,
        headers: Any = None,
        enabled: bool = True,
    ) -> MCPServerConfig:
        name = str(name or "").strip()
        if not name:
            raise ValueError("connector name is required")
        transport = str(transport or "").strip().lower()
        if transport not in {"http", "stdio"}:
            raise ValueError("connector transport must be http or stdio")
        normalized_url = str(url or "").strip() or None
        normalized_command = str(command or "").strip() or None
        if transport == "http":
            if normalized_url is None or not normalized_url.startswith(("http://", "https://")):
                raise ValueError("HTTP connectors require an http:// or https:// URL")
            normalized_command = None
        elif normalized_command is None:
            raise ValueError("stdio connectors require a command")
        else:
            normalized_url = None
        arguments = _json_array(args, field="args")
        identifier = str(server_id or "").strip() or f"mcp_{uuid.uuid4().hex}"
        now = _utc_now()
        existing = self.server(identifier)
        environment = (
            dict(existing.env)
            if env is None and existing is not None
            else _json_object(env, field="env")
        )
        request_headers = (
            dict(existing.headers)
            if headers is None and existing is not None
            else _json_object(headers, field="headers")
        )
        created_at = existing.created_at if existing is not None else now
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO mcp_servers (
                    id, name, transport, url, command, args_json, env_json,
                    headers_json, enabled, tool_count, last_status, last_error,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    transport = excluded.transport,
                    url = excluded.url,
                    command = excluded.command,
                    args_json = excluded.args_json,
                    env_json = excluded.env_json,
                    headers_json = excluded.headers_json,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    identifier,
                    name,
                    transport,
                    normalized_url,
                    normalized_command,
                    json.dumps(arguments, separators=(",", ":")),
                    json.dumps(environment, separators=(",", ":")),
                    json.dumps(request_headers, separators=(",", ":")),
                    int(bool(enabled)),
                    existing.tool_count if existing is not None else 0,
                    existing.last_status if existing is not None else None,
                    existing.last_error if existing is not None else None,
                    created_at,
                    now,
                ),
            )
        configured = self.server(identifier)
        if configured is None:
            raise RuntimeError("connector could not be saved")
        return configured

    def record_server_status(
        self,
        server_id: str,
        *,
        status: str,
        tool_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE mcp_servers
                SET last_status = ?, last_error = ?, tool_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error, max(0, int(tool_count)), _utc_now(), server_id),
            )

    def delete_server(self, server_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM mcp_servers WHERE id = ?", (server_id,)
            )
        return cursor.rowcount > 0

    def list_skills(self, *, enabled_only: bool = False) -> list[SkillConfig]:
        query = "SELECT * FROM skills"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY name COLLATE NOCASE"
        with self._lock:
            rows = self._connection.execute(query).fetchall()
        return [self._skill(row) for row in rows]

    def configure_skill(
        self,
        *,
        skill_id: Optional[str],
        name: str,
        instructions: str,
        enabled: bool = True,
    ) -> SkillConfig:
        name = str(name or "").strip()
        instructions = str(instructions or "").strip()
        if not name:
            raise ValueError("skill name is required")
        if not instructions:
            raise ValueError("skill instructions are required")
        if len(instructions) > 50_000:
            raise ValueError("skill instructions cannot exceed 50000 characters")
        identifier = str(skill_id or "").strip() or f"skill_{uuid.uuid4().hex}"
        now = _utc_now()
        with self._lock:
            existing = self._connection.execute(
                "SELECT created_at FROM skills WHERE id = ?", (identifier,)
            ).fetchone()
        created_at = str(existing["created_at"]) if existing is not None else now
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO skills (id, name, instructions, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    instructions = excluded.instructions,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (identifier, name, instructions, int(bool(enabled)), created_at, now),
            )
        skill = next(
            (item for item in self.list_skills() if item.id == identifier), None
        )
        if skill is None:
            raise RuntimeError("skill could not be saved")
        return skill

    def delete_skill(self, skill_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM skills WHERE id = ?", (skill_id,)
            )
        return cursor.rowcount > 0

    def skill_prompt(self, skill_ids: Optional[Iterable[str]] = None) -> Optional[str]:
        requested = None if skill_ids is None else {str(value) for value in skill_ids}
        skills = [
            skill
            for skill in self.list_skills(enabled_only=requested is None)
            if requested is None or skill.id in requested
        ]
        if not skills:
            return None
        sections = [f"## {skill.name}\n{skill.instructions}" for skill in skills]
        return "Reusable instructions enabled in MachBoost:\n\n" + "\n\n".join(sections)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _server(row: sqlite3.Row) -> MCPServerConfig:
        return MCPServerConfig(
            id=str(row["id"]),
            name=str(row["name"]),
            transport=str(row["transport"]),
            url=str(row["url"]) if row["url"] else None,
            command=str(row["command"]) if row["command"] else None,
            args=tuple(json.loads(str(row["args_json"]))),
            env=dict(json.loads(str(row["env_json"]))),
            headers=dict(json.loads(str(row["headers_json"]))),
            enabled=bool(row["enabled"]),
            tool_count=int(row["tool_count"]),
            last_status=str(row["last_status"]) if row["last_status"] else None,
            last_error=str(row["last_error"]) if row["last_error"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _skill(row: sqlite3.Row) -> SkillConfig:
        return SkillConfig(
            id=str(row["id"]),
            name=str(row["name"]),
            instructions=str(row["instructions"]),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


class MCPConnectorManager:
    def __init__(self, store: ExtensionStore) -> None:
        self.store = store

    def list_tools(self, server_id: str) -> list[dict[str, Any]]:
        server = self._required_server(server_id)
        try:
            tools = asyncio.run(self._list_tools(server))
        except Exception as exc:
            self.store.record_server_status(
                server.id, status="error", error=str(exc), tool_count=0
            )
            raise RuntimeError(f"{server.name}: {exc}") from exc
        self.store.record_server_status(
            server.id, status="ready", tool_count=len(tools), error=None
        )
        return tools

    def search_tools(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        words = tuple(
            value.lower()
            for value in str(query or "").split()
            if value.strip()
        )
        rows: list[tuple[int, dict[str, Any]]] = []
        for server in self.store.list_servers(enabled_only=True):
            try:
                tools = self.list_tools(server.id)
            except RuntimeError:
                continue
            for tool in tools:
                haystack = " ".join(
                    (
                        server.name,
                        str(tool.get("name") or ""),
                        str(tool.get("description") or ""),
                    )
                ).lower()
                score = sum(3 if word in str(tool.get("name") or "").lower() else 1 for word in words if word in haystack)
                if words and score == 0:
                    continue
                rows.append((score, tool))
        rows.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("server_name") or "").lower(),
                str(item[1].get("name") or "").lower(),
            )
        )
        return [tool for _, tool in rows[: max(1, min(int(limit), 25))]]

    def call_tool(
        self,
        server_id: str,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        server = self._required_server(server_id)
        if not server.enabled:
            raise ValueError(f"connector is disabled: {server.name}")
        try:
            return asyncio.run(
                self._call_tool(server, str(name or "").strip(), arguments or {})
            )
        except Exception as exc:
            self.store.record_server_status(
                server.id,
                status="error",
                error=str(exc),
                tool_count=server.tool_count,
            )
            raise RuntimeError(f"{server.name}: {exc}") from exc

    async def _list_tools(self, server: MCPServerConfig) -> list[dict[str, Any]]:
        async with self._client(server) as client:
            result = await client.list_tools()
        tools = getattr(result, "tools", ())
        return [
            {
                "server_id": server.id,
                "server_name": server.name,
                "name": str(tool.name),
                "title": getattr(tool, "title", None),
                "description": getattr(tool, "description", None),
                "input_schema": dict(getattr(tool, "input_schema", {}) or {}),
                "annotations": _model_dump(getattr(tool, "annotations", None)),
            }
            for tool in tools
        ]

    async def _call_tool(
        self,
        server: MCPServerConfig,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not name:
            raise ValueError("MCP tool name is required")
        async with self._client(server) as client:
            result = await client.call_tool(name, arguments)
        content = [_model_dump(item) for item in getattr(result, "content", ())]
        text = _mcp_result_text(content, getattr(result, "structured_content", None))
        self.store.record_server_status(
            server.id,
            status="ready",
            tool_count=server.tool_count,
            error=None,
        )
        return {
            "server_id": server.id,
            "server_name": server.name,
            "tool": name,
            "is_error": bool(getattr(result, "is_error", False)),
            "content": content,
            "structured_content": getattr(result, "structured_content", None),
            "text": text,
        }

    @asynccontextmanager
    async def _client(self, server: MCPServerConfig) -> AsyncIterator[Any]:
        try:
            from mcp import Client, StdioServerParameters
        except ImportError as exc:
            raise RuntimeError(
                "MCP support is missing. Reinstall MachBoost to restore its connector runtime."
            ) from exc
        if server.transport == "stdio":
            parameters = StdioServerParameters(
                command=str(server.command),
                args=list(server.args),
                env=dict(server.env) or None,
            )
            async with Client(parameters) as client:
                yield client
            return
        if not server.headers:
            async with Client(str(server.url)) as client:
                yield client
            return
        try:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise RuntimeError("authenticated MCP HTTP support is unavailable") from exc
        async with httpx2.AsyncClient(headers=server.headers) as http_client:
            transport = streamable_http_client(str(server.url), http_client=http_client)
            async with Client(transport) as client:
                yield client

    def _required_server(self, server_id: str) -> MCPServerConfig:
        server = self.store.server(str(server_id or ""))
        if server is None:
            raise ValueError("MCP connector was not found")
        return server


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(by_alias=True, exclude_none=True)
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def _mcp_result_text(content: Sequence[Any], structured: Any) -> str:
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
        elif item.get("type") in {"image", "audio"}:
            parts.append(
                f"[{item.get('type')} result: {item.get('mimeType') or 'binary'}]"
            )
        elif item.get("type") == "resource" and item.get("resource"):
            resource = item["resource"]
            if isinstance(resource, dict) and resource.get("text"):
                parts.append(str(resource["text"]))
    if structured is not None:
        scalar_result = (
            structured.get("result")
            if isinstance(structured, dict) and set(structured) == {"result"}
            else None
        )
        if not isinstance(scalar_result, str) or scalar_result not in parts:
            parts.append(json.dumps(structured, ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts).strip()


MCP_GATEWAY_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "search_mcp_tools",
            "description": "Find tools exposed by the user's enabled MCP connectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 25},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": "Call one MCP tool returned by search_mcp_tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["server_id", "name", "arguments"],
            },
        },
    },
)
