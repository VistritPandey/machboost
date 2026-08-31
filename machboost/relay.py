from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


RELAY_SCHEMA = "machboost.claude-loopback-relay.v1"
DEFAULT_RELAY_PORT = 11436
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def relay_home() -> Path:
    return Path(os.environ.get("MACHBOOST_HOME", "~/.machboost")).expanduser()


def relay_state_path() -> Path:
    return relay_home() / "claude-loopback-relay.json"


def relay_status(path: Optional[Path] = None) -> dict[str, Any]:
    state = _read_json(path or relay_state_path())
    if state.get("schema") != RELAY_SCHEMA:
        return {}
    state["running"] = _relay_process_matches(int(state.get("pid") or 0))
    return state


def start_claude_gateway_relay(
    upstream: str,
    upstream_token: str,
    *,
    preferred_port: int = DEFAULT_RELAY_PORT,
    state_path: Optional[Path] = None,
    timeout: float = 10.0,
) -> tuple[str, str]:
    upstream = _normalize_upstream(upstream)
    upstream_token = str(upstream_token).strip()
    if not upstream_token:
        raise ValueError("the shared MachBoost host API key is missing")
    state_path = state_path or relay_state_path()
    stop_claude_gateway_relay(state_path=state_path)
    port = _available_loopback_port(preferred_port)
    local_token = "mbr_" + secrets.token_urlsafe(32)
    config = json.dumps(
        {
            "upstream": upstream,
            "upstream_token": upstream_token,
            "local_token": local_token,
        }
    ).encode("utf-8")
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, config)
    finally:
        os.close(write_fd)

    log_dir = relay_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "claude-relay.log"
    command = [
        sys.executable,
        "-m",
        "machboost.relay",
        "serve",
        "--port",
        str(port),
        "--config-fd",
        str(read_fd),
    ]
    try:
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                close_fds=True,
                pass_fds=(read_fd,),
                start_new_session=True,
            )
    finally:
        os.close(read_fd)

    endpoint = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + max(1.0, timeout)
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            with urlopen(endpoint + "/health", timeout=0.5) as response:
                if response.status == 200:
                    _write_json(
                        state_path,
                        {
                            "schema": RELAY_SCHEMA,
                            "pid": process.pid,
                            "endpoint": endpoint,
                            "upstream": upstream,
                            "started_at": time.time(),
                        },
                    )
                    return endpoint, local_token
        except (OSError, URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    if process.poll() is None:
        process.terminate()
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"the Claude loopback relay did not start{detail}")


def stop_claude_gateway_relay(*, state_path: Optional[Path] = None) -> None:
    state_path = state_path or relay_state_path()
    state = _read_json(state_path)
    pid = int(state.get("pid") or 0)
    if _relay_process_matches(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and _process_exists(pid):
            time.sleep(0.05)
        if _relay_process_matches(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass


class LoopbackRelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        upstream: str,
        upstream_token: str,
        local_token: str,
    ) -> None:
        super().__init__(address, LoopbackRelayHandler)
        self.upstream = urlparse(upstream)
        self.upstream_token = upstream_token
        self.local_token = local_token


class LoopbackRelayHandler(BaseHTTPRequestHandler):
    server: LoopbackRelayServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()

    def _proxy(self) -> None:
        if not self._authorized():
            self._send_json(401, {"error": "authentication required"})
            return
        if not (self.path.startswith("/v1/") or self.path.startswith("/api/")):
            self._send_json(404, {"error": "unsupported relay endpoint"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        upstream = self.server.upstream
        connection_class = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        port = upstream.port or (443 if upstream.scheme == "https" else 80)
        connection = connection_class(upstream.hostname, port, timeout=600)
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
            and name.lower() not in {"host", "authorization", "content-length"}
        }
        headers["Authorization"] = f"Bearer {self.server.upstream_token}"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        base_path = upstream.path.rstrip("/")
        response_started = False
        try:
            connection.request(self.command, base_path + self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except (OSError, http.client.HTTPException) as exc:
            if not response_started and not self.wfile.closed:
                self._send_json(502, {"error": f"shared host unavailable: {exc}"})
        finally:
            connection.close()

    def _authorized(self) -> bool:
        value = self.headers.get("Authorization", "")
        return secrets.compare_digest(value, f"Bearer {self.server.local_token}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_relay(port: int, config_fd: int) -> None:
    with os.fdopen(config_fd, "rb", closefd=True) as stream:
        config = json.loads(stream.read().decode("utf-8"))
    server = LoopbackRelayServer(
        ("127.0.0.1", int(port)),
        upstream=_normalize_upstream(str(config["upstream"])),
        upstream_token=str(config["upstream_token"]),
        local_token=str(config["local_token"]),
    )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server.serve_forever(poll_interval=0.2)


def _normalize_upstream(value: str) -> str:
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("relay upstream must be an HTTP(S) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("relay upstream URL cannot contain credentials, query, or fragment")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{host}{port}{path}"


def _available_loopback_port(preferred: int) -> int:
    for port in range(max(1024, int(preferred)), max(1024, int(preferred)) + 32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no loopback port is available for the Claude gateway relay")


def _relay_process_matches(pid: int) -> bool:
    if not _process_exists(pid):
        return False
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "machboost.relay" in result.stdout


def _process_exists(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config-fd", type=int, required=True)
    args = parser.parse_args(argv)
    serve_relay(args.port, args.config_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
