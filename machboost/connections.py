from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


KEYCHAIN_SERVICE = "com.machboost.cli.connection"
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class ConnectionProfile:
    id: str
    name: str
    endpoint: str


class ConnectionStore:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        home = Path(os.environ.get("MACHBOOST_HOME", "~/.machboost")).expanduser()
        self.path = Path(path).expanduser() if path else home / "connections.json"

    def list(self) -> list[ConnectionProfile]:
        return sorted(self._load()["profiles"], key=lambda item: item.name.lower())

    def active(self) -> Optional[ConnectionProfile]:
        data = self._load()
        active_id = data.get("active")
        return next((item for item in data["profiles"] if item.id == active_id), None)

    def get(self, name: str) -> ConnectionProfile:
        key = str(name).strip().lower()
        profile = next(
            (item for item in self._load()["profiles"] if item.name.lower() == key or item.id == name),
            None,
        )
        if profile is None:
            raise KeyError(name)
        return profile

    def save(self, name: str, endpoint: str, *, api_token: Optional[str] = None) -> ConnectionProfile:
        name = str(name).strip()
        if not PROFILE_NAME.fullmatch(name):
            raise ValueError("connection name must use letters, numbers, dots, dashes, or underscores")
        endpoint = normalize_endpoint(endpoint)
        if is_loopback_endpoint(endpoint):
            raise ValueError("the local server is already available as `local`; connect to another Mac instead")
        data = self._load()
        existing = next((item for item in data["profiles"] if item.name.lower() == name.lower()), None)
        profile = ConnectionProfile(
            id=existing.id if existing else "connection_" + uuid.uuid4().hex,
            name=name,
            endpoint=endpoint,
        )
        data["profiles"] = [item for item in data["profiles"] if item.id != profile.id] + [profile]
        data["active"] = profile.id
        self._write(data)
        if api_token:
            set_connection_secret(profile.id, api_token)
        return profile

    def select(self, name: str) -> Optional[ConnectionProfile]:
        if str(name).strip().lower() == "local":
            data = self._load()
            data["active"] = None
            self._write(data)
            return None
        profile = self.get(name)
        data = self._load()
        data["active"] = profile.id
        self._write(data)
        return profile

    def remove(self, name: str) -> ConnectionProfile:
        profile = self.get(name)
        data = self._load()
        data["profiles"] = [item for item in data["profiles"] if item.id != profile.id]
        if data.get("active") == profile.id:
            data["active"] = None
        self._write(data)
        delete_connection_secret(profile.id)
        return profile

    def token(self, profile: ConnectionProfile) -> Optional[str]:
        return get_connection_secret(profile.id)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema": "machboost.connections.v1", "active": None, "profiles": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            profiles = [ConnectionProfile(**item) for item in raw.get("profiles", [])]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid MachBoost connections file: {exc}") from exc
        return {
            "schema": "machboost.connections.v1",
            "active": raw.get("active"),
            "profiles": profiles,
        }

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "machboost.connections.v1",
            "active": data.get("active"),
            "profiles": [asdict(item) for item in data.get("profiles", [])],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.path)


def normalize_endpoint(value: str) -> str:
    endpoint = str(value).strip()
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP(S) URL or host:port")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint cannot contain credentials, a query, or a fragment")
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    if path:
        raise ValueError("endpoint must point to the MachBoost server root, not an API path")
    port = f":{parsed.port}" if parsed.port else ""
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}{port}"


def is_loopback_endpoint(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def active_connection() -> Optional[ConnectionProfile]:
    try:
        return ConnectionStore().active()
    except ValueError:
        return None


def active_connection_token() -> Optional[str]:
    profile = active_connection()
    return get_connection_secret(profile.id) if profile else None


def set_connection_secret(profile_id: str, token: str) -> None:
    token = str(token).strip()
    if not token:
        return
    _run_security(
        ["add-generic-password", "-U", "-a", profile_id, "-s", KEYCHAIN_SERVICE, "-w", token]
    )


def get_connection_secret(profile_id: str) -> Optional[str]:
    try:
        result = _run_security(
            ["find-generic-password", "-w", "-a", profile_id, "-s", KEYCHAIN_SERVICE]
        )
    except RuntimeError:
        return None
    return result.stdout.strip() or None


def delete_connection_secret(profile_id: str) -> None:
    try:
        _run_security(["delete-generic-password", "-a", profile_id, "-s", KEYCHAIN_SERVICE])
    except RuntimeError:
        pass


def _run_security(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    if platform.system() != "Darwin" or not Path("/usr/bin/security").exists():
        raise RuntimeError("saved connection tokens require macOS Keychain")
    result = subprocess.run(
        ["/usr/bin/security", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "macOS Keychain operation failed")
    return result
