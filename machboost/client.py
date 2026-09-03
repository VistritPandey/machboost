from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .connections import active_connection, active_connection_token
from .server import DEFAULT_HOST, DEFAULT_PORT


class MachBoostAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def default_endpoint() -> str:
    value = os.environ.get("MACHBOOST_HOST", "").strip()
    if not value:
        profile = active_connection()
        return profile.endpoint if profile else f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


def _community_app_api_token() -> str:
    credentials = (
        Path.home()
        / "Library"
        / "Application Support"
        / "MachBoost"
        / "credentials.community.json"
    )
    try:
        metadata = credentials.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            return ""
        payload = json.loads(credentials.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    token = payload.get("lan-api-token") if isinstance(payload, dict) else None
    return str(token).strip() if token else ""


def _machboost_app_uses_community_credentials() -> Optional[bool]:
    app_locations = (
        Path("/Applications/MachBoost.app"),
        Path.home() / "Applications" / "MachBoost.app",
    )
    for app in app_locations:
        if not app.exists():
            continue
        result = subprocess.run(
            ["/usr/bin/codesign", "-dv", "--verbose=4", str(app)],
            check=False,
            capture_output=True,
            text=True,
        )
        details = f"{result.stdout}\n{result.stderr}"
        if "TeamIdentifier=not set" in details or "Signature=adhoc" in details:
            return True
        if "TeamIdentifier=" in details:
            return False
    return None


def machboost_app_api_token() -> str:
    if platform.system() != "Darwin":
        return ""
    community_credentials = _machboost_app_uses_community_credentials()
    if community_credentials is not False:
        token = _community_app_api_token()
        if token:
            return token
    if not Path("/usr/bin/security").exists():
        return ""
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-w",
            "-s",
            "io.machboost.MachBoost",
            "-a",
            "lan-api-token",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


class MachBoostClient:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        *,
        timeout: float = 300.0,
        api_token: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> None:
        self.endpoint = (endpoint or default_endpoint()).rstrip("/")
        self.timeout = float(timeout)
        self.api_token = api_token if api_token is not None else os.environ.get("MACHBOOST_API_TOKEN")
        if self.api_token is None and endpoint is None:
            self.api_token = active_connection_token()
        self.device_id = str(device_id or "").strip() or None

    def health(self) -> dict[str, Any]:
        return self.get("/healthz")

    def is_healthy(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except MachBoostAPIError:
            return False

    def ps(self) -> list[dict[str, Any]]:
        return list(self.get("/api/ps").get("models") or ())

    def tags(self) -> list[dict[str, Any]]:
        return list(self.get("/api/tags").get("models") or ())

    def catalog(self) -> list[dict[str, Any]]:
        return list(self.get("/api/catalog").get("models") or ())

    def metrics(self) -> dict[str, Any]:
        return self.get("/api/metrics")

    def extensions(self) -> dict[str, Any]:
        return self.get("/api/extensions")

    def configure_mcp_server(
        self,
        name: str,
        *,
        server_id: Optional[str] = None,
        transport: str = "http",
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: tuple[str, ...] = (),
        env: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "transport": transport,
            "args": list(args),
            "env": dict(env or {}),
            "headers": dict(headers or {}),
            "enabled": bool(enabled),
        }
        if server_id:
            payload["id"] = server_id
        if url:
            payload["url"] = url
        if command:
            payload["command"] = command
        return dict(self.post("/api/mcp/servers", payload).get("server") or {})

    def delete_mcp_server(self, server_id: str) -> bool:
        return bool(
            self.post("/api/mcp/servers/delete", {"server_id": server_id}).get(
                "removed"
            )
        )

    def test_mcp_server(self, server_id: str) -> list[dict[str, Any]]:
        return list(
            self.post("/api/mcp/servers/test", {"server_id": server_id}).get(
                "tools"
            )
            or ()
        )

    def search_mcp_tools(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return list(
            self.post("/api/mcp/search", {"query": query, "limit": int(limit)}).get(
                "tools"
            )
            or ()
        )

    def call_mcp_tool(
        self,
        server_id: str,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return dict(
            self.post(
                "/api/mcp/call",
                {
                    "server_id": server_id,
                    "name": name,
                    "arguments": dict(arguments or {}),
                },
            ).get("result")
            or {}
        )

    def configure_skill(
        self,
        name: str,
        instructions: str,
        *,
        skill_id: Optional[str] = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "enabled": bool(enabled),
        }
        if skill_id:
            payload["id"] = skill_id
        return dict(self.post("/api/skills", payload).get("skill") or {})

    def delete_skill(self, skill_id: str) -> bool:
        return bool(
            self.post("/api/skills/delete", {"skill_id": skill_id}).get("removed")
        )

    def team_status(self) -> dict[str, Any]:
        return self.get("/api/team/status")

    def team_keys(self) -> list[dict[str, Any]]:
        return list(self.get("/api/team/keys").get("keys") or ())

    def team_connect(self) -> dict[str, Any]:
        return self.get("/api/team/connect")

    def team_clients(self, *, active_within_seconds: float = 120.0) -> list[dict[str, Any]]:
        query = urlencode({"active_within_seconds": float(active_within_seconds)})
        return list(self.get(f"/api/team/clients?{query}").get("clients") or ())

    def report_team_presence(
        self,
        device_name: str,
        app_version: str,
        *,
        device_id: Optional[str] = None,
        workspace_name: Optional[str] = None,
        workspace_fingerprint: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        resolved_device_id = str(device_id or self.device_id or "").strip()
        if not resolved_device_id:
            raise ValueError("device_id is required")
        payload: dict[str, Any] = {
            "device_id": resolved_device_id,
            "device_name": device_name,
            "app_version": app_version,
            "mode": "connect",
        }
        payload.update(
            {
                key: value
                for key, value in {
                    "workspace_name": workspace_name,
                    "workspace_fingerprint": workspace_fingerprint,
                    "model": model,
                }.items()
                if value is not None
            }
        )
        return dict(self.post("/api/team/presence", payload).get("client") or {})

    def team_model_requests(
        self, *, status: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        query = urlencode(
            {
                key: value
                for key, value in {"status": status, "limit": int(limit)}.items()
                if value is not None
            }
        )
        return list(
            self.get(f"/api/team/model-requests?{query}").get("requests") or ()
        )

    def request_team_model(
        self,
        model: str,
        *,
        device_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model}
        resolved_device_id = str(device_id or self.device_id or "").strip()
        if resolved_device_id:
            payload["device_id"] = resolved_device_id
        if note is not None:
            payload["note"] = note
        return dict(
            self.post("/api/team/model-requests", payload).get("request") or {}
        )

    def resolve_team_model_request(
        self,
        request_id: str,
        *,
        status: str,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"request_id": request_id, "status": status}
        if note is not None:
            payload["note"] = note
        return dict(
            self.post("/api/team/model-requests/resolve", payload).get("request")
            or {}
        )

    def create_team_key(
        self,
        name: str,
        *,
        scopes: tuple[str, ...] = (
            "inference",
            "models:read",
            "workspaces:read",
        ),
        allowed_models: tuple[str, ...] = (),
        max_concurrent: int = 2,
        requests_per_minute: int = 60,
    ) -> dict[str, Any]:
        return self.post(
            "/api/team/keys",
            {
                "name": name,
                "scopes": list(scopes),
                "allowed_models": list(allowed_models),
                "max_concurrent": int(max_concurrent),
                "requests_per_minute": int(requests_per_minute),
            },
        )

    def revoke_team_key(self, key_id: str) -> bool:
        try:
            return bool(
                self.post("/api/team/keys/revoke", {"key_id": key_id}).get(
                    "revoked"
                )
            )
        except MachBoostAPIError as exc:
            if exc.status == 404:
                return False
            raise

    def update_team_settings(
        self,
        *,
        trace_mode: str,
        retention_days: Optional[int],
        max_storage_bytes: int,
    ) -> dict[str, Any]:
        return dict(
            self.post(
                "/api/team/settings",
                {
                    "trace_mode": trace_mode,
                    "retention_days": retention_days,
                    "max_storage_bytes": int(max_storage_bytes),
                },
            ).get("settings")
            or {}
        )

    def traces(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(self.get(f"/api/traces?limit={int(limit)}").get("traces") or ())

    def evaluations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return list(
            self.get(f"/api/evaluations?limit={int(limit)}").get("evaluations")
            or ()
        )

    def evaluate_traces(
        self,
        trace_ids: list[str],
        *,
        name: str = "Trace evaluation",
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"trace_ids": trace_ids, "name": name}
        if model:
            payload["model"] = model
        return dict(
            self.post("/api/evaluations", payload).get("evaluation") or {}
        )

    def memories(
        self,
        *,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = urlencode(
            {
                key: value
                for key, value in {
                    "workspace_id": workspace_id,
                    "limit": int(limit),
                }.items()
                if value is not None
            }
        )
        return list(self.get(f"/api/memory?{query}").get("memories") or ())

    def create_memory(
        self,
        workspace_id: str,
        title: str,
        content: str,
        *,
        scope: str = "private",
        kind: str = "fact",
        query_text: str = "",
        revision: Optional[str] = None,
        dependencies: Optional[dict[str, str]] = None,
        evidence: tuple[str, ...] = (),
        confidence: float = 0.5,
        validated_by: tuple[str, ...] = (),
        pinned: bool = False,
        ttl_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "scope": scope,
            "kind": kind,
            "title": title,
            "content": content,
            "query_text": query_text,
            "dependencies": dict(dependencies or {}),
            "evidence": list(evidence),
            "confidence": float(confidence),
            "validated_by": list(validated_by),
            "pinned": bool(pinned),
        }
        if revision is not None:
            payload["revision"] = revision
        if ttl_seconds is not None:
            payload["ttl_seconds"] = float(ttl_seconds)
        return dict(self.post("/api/memory", payload).get("memory") or {})

    def delete_memories(self, memory_ids: list[str]) -> int:
        return int(
            self.post("/api/memory/delete", {"memory_ids": memory_ids}).get(
                "removed"
            )
            or 0
        )

    def cache_metrics(self) -> dict[str, Any]:
        return self.get("/api/cache/metrics")

    def providers(self) -> list[dict[str, Any]]:
        return list(self.get("/api/providers").get("providers") or ())

    def configure_provider(
        self,
        name: str,
        base_url: str,
        models: tuple[str, ...],
        *,
        provider_id: Optional[str] = None,
        enabled: bool = True,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        monthly_budget_usd: Optional[float] = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "base_url": base_url,
            "models": list(models),
            "enabled": bool(enabled),
            "input_cost_per_million": float(input_cost_per_million),
            "output_cost_per_million": float(output_cost_per_million),
            "timeout_seconds": float(timeout_seconds),
        }
        optional = {
            "id": provider_id,
            "api_key": api_key,
            "api_key_env": api_key_env,
            "monthly_budget_usd": monthly_budget_usd,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return dict(self.post("/api/providers", payload).get("provider") or {})

    def set_provider_secret(self, provider_id: str, api_key: str) -> bool:
        return bool(
            self.post(
                "/api/providers/secret",
                {"provider_id": provider_id, "api_key": api_key},
            ).get("has_secret")
        )

    def provider_usage(self, provider_id: Optional[str] = None) -> dict[str, Any]:
        suffix = "?" + urlencode({"provider_id": provider_id}) if provider_id else ""
        return self.get("/api/providers/usage" + suffix)

    def delete_provider(self, provider_id: str) -> bool:
        try:
            return bool(
                self.post(
                    "/api/providers/delete", {"provider_id": provider_id}
                ).get("removed")
            )
        except MachBoostAPIError as exc:
            if exc.status == 404:
                return False
            raise

    def workspaces(self) -> list[dict[str, Any]]:
        return list(self.get("/api/workspaces").get("workspaces") or ())

    def register_workspace(
        self,
        path: str | os.PathLike[str],
        *,
        name: Optional[str] = None,
        index: bool = True,
        max_file_bytes: int = 1_000_000,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": os.fspath(path),
            "index": bool(index),
            "max_file_bytes": int(max_file_bytes),
        }
        if name:
            payload["name"] = name
        response = self.post("/api/workspaces", payload)
        return dict(response.get("workspace") or {})

    def reindex_workspace(
        self,
        workspace_id: str,
        *,
        max_file_bytes: int = 1_000_000,
    ) -> dict[str, Any]:
        response = self.post(
            "/api/workspaces/index",
            {
                "workspace_id": workspace_id,
                "max_file_bytes": int(max_file_bytes),
            },
        )
        return dict(response.get("workspace") or {})

    def query_workspace(
        self,
        workspace_id: str,
        query: str,
        *,
        top_k: int = 12,
        max_chars: int = 48_000,
    ) -> dict[str, Any]:
        return self.post(
            "/api/workspaces/query",
            {
                "workspace_id": workspace_id,
                "query": query,
                "top_k": int(top_k),
                "max_chars": int(max_chars),
            },
        )

    def remove_workspace(self, workspace_id: str) -> bool:
        try:
            return bool(
                self.post(
                    "/api/workspaces/delete",
                    {"workspace_id": workspace_id},
                ).get("removed")
            )
        except MachBoostAPIError as exc:
            if exc.status == 404:
                return False
            raise

    def pull(
        self,
        model: str,
        *,
        revision: Optional[str] = None,
        stream: bool = False,
        request_id: Optional[str] = None,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"model": model, "stream": bool(stream)}
        if revision:
            payload["revision"] = revision
        if request_id:
            payload["request_id"] = request_id
        if stream:
            return self.stream("/api/pull", payload)
        return self.post("/api/pull", payload)

    def create_model(
        self,
        name: str,
        source: str,
        *,
        system: str = "",
        template: str = "",
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.post(
            "/api/create",
            {
                "model": name,
                "from": source,
                "system": system,
                "template": template,
                "parameters": dict(options or {}),
            },
        )

    def copy_model(self, source: str, destination: str) -> dict[str, Any]:
        return self.post(
            "/api/copy", {"source": source, "destination": destination}
        )

    def delete_model(self, model: str, *, purge: bool = False) -> bool:
        payload: dict[str, Any] = {"model": model}
        if purge:
            payload["purge"] = True
        try:
            return bool(self.post("/api/delete", payload).get("removed"))
        except MachBoostAPIError as exc:
            if exc.status == 404:
                return False
            raise

    def embed(
        self,
        model: str,
        inputs: str | list[str],
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = None,
    ) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": model,
            "input": inputs,
            "options": dict(options or {}),
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return list(self.post("/api/embed", payload).get("embeddings") or ())

    def load(
        self,
        model: str,
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = "5m",
        warmup: bool | str = False,
    ) -> dict[str, Any]:
        return self.post(
            "/api/load",
            {
                "model": model,
                "options": dict(options or {}),
                "keep_alive": keep_alive,
                "warmup": warmup,
            },
        )

    def show(
        self,
        model: str,
        *,
        preflight: bool = False,
        allow_network: bool = False,
        backend: str = "auto",
    ) -> dict[str, Any]:
        return self.post(
            "/api/show",
            {
                "model": model,
                "preflight": bool(preflight),
                "allow_network": bool(allow_network),
                "backend": backend,
            },
        )

    def cancel(self, request_id: str) -> bool:
        try:
            return bool(
                self.post("/api/cancel", {"request_id": request_id}).get(
                    "cancelled"
                )
            )
        except MachBoostAPIError as exc:
            if exc.status == 404:
                return False
            raise

    def stop(self, model: Optional[str] = None) -> dict[str, Any]:
        return self.post("/api/stop", {"model": model} if model else {})

    def shutdown(self) -> dict[str, Any]:
        return self.post("/api/shutdown", {})

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        options: Optional[dict[str, Any]] = None,
        context: Optional[list[str] | str] = None,
        images: Optional[list[str] | str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Any = None,
        format: Any = None,
        think: Optional[bool | str] = None,
        keep_alive: Any = None,
        stream: bool = True,
        affinity_key: Optional[str] = None,
        queue_timeout: Optional[float] = None,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        workspace_query: Optional[str] = None,
        workspace_top_k: Optional[int] = None,
        workspace_max_chars: Optional[int] = None,
        machboost: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]] | dict[str, Any]:
        chat_messages = [dict(message) for message in messages]
        if images is not None:
            if not chat_messages:
                raise ValueError("images require at least one chat message")
            target = next(
                (message for message in reversed(chat_messages) if message.get("role") == "user"),
                chat_messages[-1],
            )
            target["images"] = images
        request_options = dict(options or {})
        if affinity_key is not None:
            request_options["affinity_key"] = affinity_key
        if queue_timeout is not None:
            request_options["queue_timeout"] = float(queue_timeout)
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "options": request_options,
            "stream": bool(stream),
        }
        if context is not None:
            payload["context"] = context
        if tools is not None:
            payload["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if request_id is not None:
            payload["request_id"] = request_id
        if workspace_id is not None:
            payload["workspace_id"] = workspace_id
        if workspace_query is not None:
            payload["workspace_query"] = workspace_query
        if workspace_top_k is not None:
            payload["workspace_top_k"] = int(workspace_top_k)
        if workspace_max_chars is not None:
            payload["workspace_max_chars"] = int(workspace_max_chars)
        if machboost is not None:
            payload["machboost"] = dict(machboost)
        if stream:
            return self.stream("/api/chat", payload)
        return self.post("/api/chat", payload)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: Optional[dict[str, Any]] = None,
        context: Optional[list[str] | str] = None,
        images: Optional[list[str] | str] = None,
        format: Any = None,
        think: Optional[bool | str] = None,
        keep_alive: Any = None,
        stream: bool = True,
        affinity_key: Optional[str] = None,
        queue_timeout: Optional[float] = None,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        workspace_query: Optional[str] = None,
        workspace_top_k: Optional[int] = None,
        workspace_max_chars: Optional[int] = None,
        machboost: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]] | dict[str, Any]:
        request_options = dict(options or {})
        if affinity_key is not None:
            request_options["affinity_key"] = affinity_key
        if queue_timeout is not None:
            request_options["queue_timeout"] = float(queue_timeout)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "options": request_options,
            "stream": bool(stream),
        }
        if context is not None:
            payload["context"] = context
        if images is not None:
            payload["images"] = images
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if request_id is not None:
            payload["request_id"] = request_id
        if workspace_id is not None:
            payload["workspace_id"] = workspace_id
        if workspace_query is not None:
            payload["workspace_query"] = workspace_query
        if workspace_top_k is not None:
            payload["workspace_top_k"] = int(workspace_top_k)
        if workspace_max_chars is not None:
            payload["workspace_max_chars"] = int(workspace_max_chars)
        if machboost is not None:
            payload["machboost"] = dict(machboost)
        if stream:
            return self.stream("/api/generate", payload)
        return self.post("/api/generate", payload)

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        response = None
        for attempt in range(2):
            request = Request(
                self.endpoint + path,
                method="POST",
                data=data,
                headers=self._headers(
                    content_type=True,
                    accept="application/x-ndjson",
                ),
            )
            try:
                response = urlopen(request, timeout=self.timeout)
                break
            except HTTPError as exc:
                if attempt == 0 and self._authorize_local_retry(exc.code):
                    continue
                raise api_error(exc) from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise api_error(exc) from exc
        if response is None:
            raise MachBoostAPIError("server did not return a streaming response")
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MachBoostAPIError(f"invalid streaming response: {line[:200]}") from exc
                if "error" in row:
                    raise MachBoostAPIError(str(row["error"]))
                yield row

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        raw = ""
        for attempt in range(2):
            request = Request(
                self.endpoint + path,
                method=method,
                data=data,
                headers=self._headers(content_type=data is not None),
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                if attempt == 0 and self._authorize_local_retry(exc.code):
                    continue
                raise api_error(exc) from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise api_error(exc) from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MachBoostAPIError(f"invalid server response: {raw[:200]}") from exc
        if not isinstance(value, dict):
            raise MachBoostAPIError("server response was not a JSON object")
        if "error" in value:
            raise MachBoostAPIError(str(value["error"]))
        return value

    def _authorize_local_retry(self, status: int) -> bool:
        if status != 401 or self.api_token:
            return False
        host = (urlparse(self.endpoint).hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return False
        self.api_token = machboost_app_api_token() or None
        return self.api_token is not None

    def _headers(
        self,
        *,
        content_type: bool = False,
        accept: Optional[str] = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = "application/json"
        if accept:
            headers["Accept"] = accept
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if self.device_id:
            headers["X-MachBoost-Device-ID"] = self.device_id
        return headers


def api_error(exc: Exception) -> MachBoostAPIError:
    if isinstance(exc, HTTPError):
        code = None
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            message = payload.get("error") or str(exc)
            code = payload.get("code")
        except Exception:
            message = str(exc)
        return MachBoostAPIError(
            str(message),
            status=exc.code,
            code=str(code) if code is not None else None,
        )
    if isinstance(exc, URLError):
        return MachBoostAPIError(f"cannot reach MachBoost server: {exc.reason}")
    return MachBoostAPIError(str(exc))


def ensure_server(
    endpoint: Optional[str] = None,
    *,
    timeout: float = 30.0,
    log_path: Optional[Path] = None,
) -> tuple[MachBoostClient, bool]:
    from . import __version__

    client = MachBoostClient(endpoint, timeout=max(1.0, timeout))
    parsed = urlparse(client.endpoint)
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or DEFAULT_PORT
    is_local = host in {"127.0.0.1", "localhost", "::1"}

    try:
        health = client.health()
    except MachBoostAPIError:
        health = {}
    if health.get("status") == "ok":
        running_version = str(health.get("version") or "unknown")
        if running_version == __version__:
            return client, False
        if not is_local:
            raise MachBoostAPIError(
                f"MachBoost server version {running_version} does not match client {__version__}"
            )
        client.shutdown()
        shutdown_deadline = time.monotonic() + min(5.0, timeout)
        while time.monotonic() < shutdown_deadline:
            if not client.is_healthy():
                break
            time.sleep(0.05)
        else:
            raise MachBoostAPIError(
                f"stale MachBoost server {running_version} did not shut down"
            )

    if not is_local:
        raise MachBoostAPIError(f"refusing to auto-start a server for non-local endpoint {client.endpoint!r}")

    cache_dir = Path.home() / ".cache" / "machboost"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or cache_dir / "server.log"
    pid_path = cache_dir / "server.pid"
    command = [
        sys.executable,
        "-m",
        "machboost",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="ascii")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            started_health = client.health()
        except MachBoostAPIError:
            started_health = {}
        if (
            started_health.get("status") == "ok"
            and str(started_health.get("version") or "unknown") == __version__
        ):
            return client, True
        return_code = process.poll()
        if return_code is not None:
            raise MachBoostAPIError(
                f"MachBoost server exited with code {return_code}; inspect {log_path}"
            )
        time.sleep(0.1)
    process.terminate()
    raise MachBoostAPIError(f"MachBoost server did not become ready within {timeout:.1f}s; inspect {log_path}")
