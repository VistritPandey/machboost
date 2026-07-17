from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .server import DEFAULT_HOST, DEFAULT_PORT


class MachBoostAPIError(RuntimeError):
    pass


def default_endpoint() -> str:
    value = os.environ.get("MACHBOOST_HOST", "").strip()
    if not value:
        return f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


class MachBoostClient:
    def __init__(self, endpoint: Optional[str] = None, *, timeout: float = 300.0) -> None:
        self.endpoint = (endpoint or default_endpoint()).rstrip("/")
        self.timeout = float(timeout)

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

    def pull(self, model: str, *, revision: Optional[str] = None) -> dict[str, Any]:
        payload = {"model": model}
        if revision:
            payload["revision"] = revision
        return self.post("/api/pull", payload)

    def load(
        self,
        model: str,
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = "5m",
        warmup: bool = False,
    ) -> dict[str, Any]:
        return self.post(
            "/api/load",
            {
                "model": model,
                "options": dict(options or {}),
                "keep_alive": keep_alive,
                "warmup": bool(warmup),
            },
        )

    def show(self, model: str) -> dict[str, Any]:
        return self.post("/api/show", {"model": model})

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
        keep_alive: Any = None,
        stream: bool = True,
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "options": dict(options or {}),
            "stream": bool(stream),
        }
        if context is not None:
            payload["context"] = context
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
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
        keep_alive: Any = None,
        stream: bool = True,
    ) -> Iterator[dict[str, Any]] | dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "options": dict(options or {}),
            "stream": bool(stream),
        }
        if context is not None:
            payload["context"] = context
        if images is not None:
            payload["images"] = images
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if stream:
            return self.stream("/api/generate", payload)
        return self.post("/api/generate", payload)

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint + path,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        )
        try:
            response = urlopen(request, timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise api_error(exc) from exc
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
        request = Request(
            self.endpoint + path,
            method=method,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
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


def api_error(exc: Exception) -> MachBoostAPIError:
    if isinstance(exc, HTTPError):
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            message = payload.get("error") or str(exc)
        except Exception:
            message = str(exc)
        return MachBoostAPIError(str(message))
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
