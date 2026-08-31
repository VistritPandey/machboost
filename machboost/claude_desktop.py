from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from .relay import relay_state_path, relay_status, stop_claude_gateway_relay


PROFILE_ID = "00000000-0000-4000-8000-000000000135"
PROFILE_NAME = "MachBoost"
MODEL_CONFIG_SCHEMA = "machboost.claude-desktop-models.v1"
PROFILE_STATE_SCHEMA = "machboost.claude-desktop-profile-state.v1"


@dataclass(frozen=True)
class ClaudeDesktopRoute:
    id: str
    family: str
    created_at: str
    family_default: bool


CLAUDE_DESKTOP_ROUTES = (
    ClaudeDesktopRoute("claude-fable-5", "fable", "2026-06-09T00:00:00Z", True),
    ClaudeDesktopRoute("claude-opus-5", "opus", "2026-07-24T00:00:00Z", True),
    ClaudeDesktopRoute("claude-sonnet-5", "sonnet", "2026-06-30T00:00:00Z", True),
    ClaudeDesktopRoute(
        "claude-haiku-4-5-20251001",
        "haiku",
        "2025-10-01T00:00:00Z",
        True,
    ),
    ClaudeDesktopRoute(
        "claude-sonnet-4-6",
        "sonnet",
        "2025-11-18T00:00:00Z",
        False,
    ),
)
CLAUDE_DESKTOP_ROUTE_IDS = frozenset(route.id for route in CLAUDE_DESKTOP_ROUTES)


def machboost_home() -> Path:
    return Path(os.environ.get("MACHBOOST_HOME", "~/.machboost")).expanduser()


def model_config_path() -> Path:
    return machboost_home() / "claude-desktop-models.json"


def load_model_mappings(path: Optional[Path] = None) -> dict[str, str]:
    path = path or model_config_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    if payload.get("schema") != MODEL_CONFIG_SCHEMA:
        return {}
    mappings = payload.get("mappings")
    if not isinstance(mappings, dict):
        return {}
    return {
        str(route): str(model).strip()
        for route, model in mappings.items()
        if route in CLAUDE_DESKTOP_ROUTE_IDS and str(model).strip()
    }


def save_model_mappings(models: Iterable[str], path: Optional[Path] = None) -> dict[str, str]:
    selected = _unique_models(models)
    if not selected:
        raise ValueError("select at least one model for Claude Desktop")
    mappings = {
        route.id: model
        for route, model in zip(CLAUDE_DESKTOP_ROUTES, selected)
    }
    path = path or model_config_path()
    _write_json(path, {"schema": MODEL_CONFIG_SCHEMA, "mappings": mappings})
    return mappings


def claude_desktop_models(
    available_models: Iterable[str],
    *,
    configured: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    available = _unique_models(available_models)
    available_set = set(available)
    configured = load_model_mappings() if configured is None else configured
    assigned: dict[str, str] = {}
    used: set[str] = set()

    for route in CLAUDE_DESKTOP_ROUTES:
        model = str(configured.get(route.id) or "").strip()
        if model and model in available_set:
            assigned[route.id] = model
            used.add(model)

    remaining = (model for model in available if model not in used)
    for route in CLAUDE_DESKTOP_ROUTES:
        if route.id in assigned:
            continue
        model = next(remaining, None)
        if model is None:
            break
        assigned[route.id] = model

    return [
        {
            "id": route.id,
            "type": "model",
            "display_name": assigned[route.id],
            "created_at": route.created_at,
            "max_tokens": 64_000,
            "anthropic_family_tier": route.family,
            "is_family_default": route.family_default,
        }
        for route in CLAUDE_DESKTOP_ROUTES
        if route.id in assigned
    ]


def resolve_claude_desktop_model(
    requested_model: str,
    available_models: Iterable[str],
    *,
    configured: Optional[dict[str, str]] = None,
) -> str:
    requested_model = str(requested_model).strip()
    if requested_model not in CLAUDE_DESKTOP_ROUTE_IDS:
        return requested_model
    for model in claude_desktop_models(available_models, configured=configured):
        if model["id"] == requested_model:
            return str(model["display_name"])
    raise ValueError(f"Claude Desktop route {requested_model!r} has no available MachBoost model")


def estimate_anthropic_input_tokens(payload: dict[str, Any]) -> int:
    text: list[str] = []
    _collect_text(payload.get("system"), text)
    _collect_text(payload.get("messages"), text)
    _collect_text(payload.get("tools"), text)
    characters = sum(len(value) for value in text)
    images = _count_content_type(payload.get("messages"), "image")
    # Claude Desktop uses this for context budgeting. A slight overestimate is
    # safer than allowing a request to cross the selected model's context limit.
    return max(1, (characters + 2) // 3 + images * 1_600)


def normalize_gateway_endpoint(value: str) -> str:
    endpoint = str(value).strip()
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Claude Desktop gateway must be an HTTP(S) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Claude Desktop gateway URL cannot contain credentials, query, or fragment")
    if parsed.path.rstrip("/"):
        raise ValueError("Claude Desktop gateway URL must point to the server root")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


class ClaudeDesktopProfileManager:
    def __init__(
        self,
        *,
        home: Optional[Path] = None,
        application_support: Optional[Path] = None,
        state_path: Optional[Path] = None,
        gateway_relay_state_path: Optional[Path] = None,
    ) -> None:
        self.home = Path(home or Path.home())
        self.application_support = Path(
            application_support or self.home / "Library" / "Application Support"
        )
        self.state_path = Path(
            state_path or machboost_home() / "claude-desktop-profile-state.json"
        )
        self.gateway_relay_state_path = Path(
            gateway_relay_state_path or relay_state_path()
        )

    def status(self) -> dict[str, Any]:
        target = self._third_party_targets()[0]
        meta = _read_json(target[1])
        profile = _read_json(target[2])
        connected = meta.get("appliedId") == PROFILE_ID and (
            profile.get("inferenceProvider") == "gateway"
        )
        endpoint = profile.get("inferenceGatewayBaseUrl") if connected else None
        relay = relay_status(self.gateway_relay_state_path)
        relayed = bool(
            connected
            and relay.get("running")
            and relay.get("endpoint") == endpoint
        )
        return {
            "schema": "machboost.claude-desktop-status.v1",
            "installed": self.installed_application() is not None,
            "connected": connected,
            "endpoint": endpoint,
            "upstream": relay.get("upstream") if relayed else endpoint,
            "relayed": relayed,
            "profile": PROFILE_NAME if connected else None,
        }

    def configure(self, endpoint: str, api_key: str, *, auto_mode: bool = True) -> dict[str, Any]:
        if platform.system() != "Darwin":
            raise RuntimeError("Claude Desktop integration is currently supported on macOS")
        endpoint = normalize_gateway_endpoint(endpoint)
        api_key = str(api_key).strip()
        if not api_key:
            raise ValueError("Claude Desktop gateway API key is required")

        self._capture_state_once()
        for path in self._normal_configs():
            config = _read_json(path)
            config["deploymentMode"] = "3p"
            _write_json(path, config)
        for desktop_config, meta_path, profile_path in self._third_party_targets():
            desktop = _read_json(desktop_config)
            desktop["deploymentMode"] = "3p"
            _write_json(desktop_config, desktop)

            meta = _read_json(meta_path)
            entries = [
                entry
                for entry in meta.get("entries", [])
                if not isinstance(entry, dict) or entry.get("id") != PROFILE_ID
            ]
            entries.append({"id": PROFILE_ID, "name": PROFILE_NAME})
            meta.update({"appliedId": PROFILE_ID, "entries": entries})
            _write_json(meta_path, meta)

            profile = _read_json(profile_path)
            profile.update(
                {
                    "inferenceProvider": "gateway",
                    "inferenceGatewayBaseUrl": endpoint,
                    "inferenceGatewayApiKey": api_key,
                    "inferenceGatewayAuthScheme": "bearer",
                    "deploymentDisplayName": PROFILE_NAME,
                    "chatTabEnabled": True,
                    "disableDeploymentModeChooser": True,
                    "coworkEgressAllowedHosts": ["*"],
                    "disableEssentialTelemetry": True,
                    "disableNonessentialTelemetry": True,
                    "autoModeEnabled": bool(auto_mode),
                }
            )
            profile.pop("inferenceModels", None)
            _write_json(profile_path, profile)
        return self.status()

    def restore(self) -> dict[str, Any]:
        state = _read_json(self.state_path)
        snapshots = state.get("snapshots", {}) if state.get("schema") == PROFILE_STATE_SCHEMA else {}
        for path in self._normal_configs():
            self._restore_deployment_mode(path, snapshots.get(str(path)))
        for desktop_config, meta_path, profile_path in self._third_party_targets():
            self._restore_deployment_mode(desktop_config, snapshots.get(str(desktop_config)))

            meta = _read_json(meta_path)
            meta["entries"] = [
                entry
                for entry in meta.get("entries", [])
                if not isinstance(entry, dict) or entry.get("id") != PROFILE_ID
            ]
            previous_meta = snapshots.get(str(meta_path)) or {}
            if previous_meta.get("applied_id_present"):
                meta["appliedId"] = previous_meta.get("applied_id")
            elif meta.get("appliedId") == PROFILE_ID:
                meta.pop("appliedId", None)
            _write_json(meta_path, meta)

            previous_profile = snapshots.get(str(profile_path)) or {}
            if previous_profile.get("exists"):
                _write_json(profile_path, previous_profile.get("value") or {})
            else:
                try:
                    profile_path.unlink()
                except FileNotFoundError:
                    pass
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass
        stop_claude_gateway_relay(state_path=self.gateway_relay_state_path)
        return self.status()

    def restart_application(self) -> None:
        app = self.installed_application()
        if app is None:
            raise RuntimeError("Claude Desktop is not installed")
        subprocess.run(
            ["/usr/bin/osascript", "-e", 'tell application "Claude" to quit'],
            check=False,
            capture_output=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            running = subprocess.run(
                ["/usr/bin/pgrep", "-x", "Claude"],
                check=False,
                capture_output=True,
            ).returncode == 0
            if not running:
                break
            time.sleep(0.2)
        time.sleep(0.5)
        launched = subprocess.run(
            ["/usr/bin/open", str(app)],
            check=False,
            capture_output=True,
            text=True,
        )
        if launched.returncode != 0:
            launched = subprocess.run(
                ["/usr/bin/open", "-b", "com.anthropic.claudefordesktop"],
                check=False,
                capture_output=True,
                text=True,
            )
        if launched.returncode != 0:
            detail = launched.stderr.strip() or "LaunchServices rejected the request"
            raise RuntimeError(f"Claude Desktop was configured but could not reopen: {detail}")

    def installed_application(self) -> Optional[Path]:
        for app in (Path("/Applications/Claude.app"), self.home / "Applications/Claude.app"):
            if app.exists():
                return app
        return None

    def _normal_configs(self) -> list[Path]:
        roots = [self.application_support / "Claude"]
        nest = self.application_support / "Claude Nest"
        if nest.exists():
            roots.append(nest)
        return [root / "claude_desktop_config.json" for root in roots]

    def _third_party_targets(self) -> list[tuple[Path, Path, Path]]:
        roots = [self.application_support / "Claude-3p"]
        nest = self.application_support / "Claude Nest-3p"
        if nest.exists():
            roots.append(nest)
        return [
            (
                root / "claude_desktop_config.json",
                root / "configLibrary" / "_meta.json",
                root / "configLibrary" / f"{PROFILE_ID}.json",
            )
            for root in roots
        ]

    def _capture_state_once(self) -> None:
        if self.state_path.exists():
            return
        snapshots: dict[str, Any] = {}
        for path in self._normal_configs():
            value = _read_json(path)
            snapshots[str(path)] = {
                "exists": path.exists(),
                "deployment_mode_present": "deploymentMode" in value,
                "deployment_mode": value.get("deploymentMode"),
            }
        for desktop, meta, profile in self._third_party_targets():
            value = _read_json(desktop)
            snapshots[str(desktop)] = {
                "exists": desktop.exists(),
                "deployment_mode_present": "deploymentMode" in value,
                "deployment_mode": value.get("deploymentMode"),
            }
            value = _read_json(meta)
            snapshots[str(meta)] = {
                "exists": meta.exists(),
                "applied_id_present": "appliedId" in value,
                "applied_id": value.get("appliedId"),
            }
            snapshots[str(profile)] = {
                "exists": profile.exists(),
                "value": _read_json(profile) if profile.exists() else None,
            }
        _write_json(
            self.state_path,
            {"schema": PROFILE_STATE_SCHEMA, "snapshots": snapshots},
        )

    @staticmethod
    def _restore_deployment_mode(path: Path, snapshot: Any) -> None:
        value = _read_json(path)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        if snapshot.get("deployment_mode_present"):
            value["deploymentMode"] = snapshot.get("deployment_mode")
        else:
            value.pop("deploymentMode", None)
        _write_json(path, value)


def _unique_models(models: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in models:
        model = str(value).strip()
        if not model or model in seen:
            continue
        seen.add(model)
        result.append(model)
    return result


def _collect_text(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        output.append(value)
    elif isinstance(value, list):
        for item in value:
            _collect_text(item, output)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in {"data", "image", "image_url"}:
                _collect_text(item, output)


def _count_content_type(value: Any, content_type: str) -> int:
    if isinstance(value, list):
        return sum(_count_content_type(item, content_type) for item in value)
    if isinstance(value, dict):
        return int(value.get("type") == content_type) + sum(
            _count_content_type(item, content_type) for item in value.values()
        )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)
