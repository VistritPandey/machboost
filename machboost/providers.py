from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlparse


ROUTE_MODES = {"local_only", "local_first", "external_first", "external_only"}
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int = 502, transient: bool = False):
        super().__init__(message)
        self.status = int(status)
        self.transient = bool(transient)


class ProviderBudgetError(ProviderError):
    def __init__(self, message: str):
        super().__init__(message, status=429, transient=False)


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    name: str
    base_url: str
    models: tuple[str, ...]
    enabled: bool = True
    api_key_env: Optional[str] = None
    monthly_budget_usd: Optional[float] = None
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    timeout_seconds: float = 120.0
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self, *, has_secret: bool = False, spent_usd: float = 0.0) -> dict[str, Any]:
        result = asdict(self)
        result["models"] = list(self.models)
        result["has_secret"] = has_secret
        result["spent_this_month_usd"] = round(float(spent_usd), 8)
        result["remaining_budget_usd"] = (
            None
            if self.monthly_budget_usd is None
            else round(max(0.0, self.monthly_budget_usd - spent_usd), 8)
        )
        return result


@dataclass(frozen=True)
class ProviderResult:
    provider_id: str
    model: str
    response: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float


Transport = Callable[
    [ProviderConfig, str, dict[str, Any], dict[str, str]],
    dict[str, Any],
]


class ProviderStore:
    """Persistent provider metadata with process-only API secrets and usage budgets."""

    def __init__(
        self,
        path: str | Path,
        *,
        transport: Optional[Transport] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.transport = transport or _default_transport
        self._secrets: dict[str, str] = {}
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30.0
        )
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._secrets.clear()
            self._connection.close()

    def configure(
        self,
        *,
        name: str,
        base_url: str,
        models: Sequence[str],
        provider_id: Optional[str] = None,
        enabled: bool = True,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        monthly_budget_usd: Optional[float] = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        timeout_seconds: float = 120.0,
    ) -> ProviderConfig:
        name = str(name).strip()
        base_url = str(base_url).strip().rstrip("/")
        normalized_models = tuple(
            dict.fromkeys(str(model).strip() for model in models if str(model).strip())
        )
        if not name or not base_url or not normalized_models:
            raise ValueError("provider name, base_url, and at least one model are required")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider base_url must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("remote provider URLs must use HTTPS")
        budget = None if monthly_budget_usd is None else float(monthly_budget_usd)
        if budget is not None and budget < 0:
            raise ValueError("monthly_budget_usd cannot be negative")
        input_cost = float(input_cost_per_million)
        output_cost = float(output_cost_per_million)
        timeout = float(timeout_seconds)
        if input_cost < 0 or output_cost < 0 or timeout <= 0:
            raise ValueError("provider costs cannot be negative and timeout must be positive")
        provider_id = str(provider_id or "provider_" + uuid.uuid4().hex).strip()
        now = self.clock()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at FROM provider_configs WHERE id = ?", (provider_id,)
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            self._connection.execute(
                """
                INSERT INTO provider_configs (
                    id, name, base_url, models_json, enabled, api_key_env,
                    monthly_budget_usd, input_cost_per_million,
                    output_cost_per_million, timeout_seconds, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name, base_url = excluded.base_url,
                    models_json = excluded.models_json, enabled = excluded.enabled,
                    api_key_env = excluded.api_key_env,
                    monthly_budget_usd = excluded.monthly_budget_usd,
                    input_cost_per_million = excluded.input_cost_per_million,
                    output_cost_per_million = excluded.output_cost_per_million,
                    timeout_seconds = excluded.timeout_seconds,
                    updated_at = excluded.updated_at
                """,
                (
                    provider_id,
                    name,
                    base_url,
                    json.dumps(normalized_models),
                    1 if enabled else 0,
                    str(api_key_env).strip() if api_key_env else None,
                    budget,
                    input_cost,
                    output_cost,
                    timeout,
                    created_at,
                    now,
                ),
            )
            if api_key is not None:
                secret = str(api_key).strip()
                if secret:
                    self._secrets[provider_id] = secret
                else:
                    self._secrets.pop(provider_id, None)
        return self.get(provider_id)

    def get(self, provider_id: str) -> ProviderConfig:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM provider_configs WHERE id = ?", (provider_id,)
            ).fetchone()
        if row is None:
            raise KeyError(provider_id)
        return _config_from_row(row)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM provider_configs ORDER BY name, id"
            ).fetchall()
        return [
            config.to_dict(
                has_secret=bool(self._secret_for(config)),
                spent_usd=self.monthly_spend(config.id),
            )
            for config in (_config_from_row(row) for row in rows)
        ]

    def delete(self, provider_id: str) -> bool:
        with self._lock, self._connection:
            removed = self._connection.execute(
                "DELETE FROM provider_configs WHERE id = ?", (provider_id,)
            ).rowcount
            self._secrets.pop(provider_id, None)
        return bool(removed)

    def set_secret(self, provider_id: str, api_key: Optional[str]) -> None:
        self.get(provider_id)
        with self._lock:
            secret = str(api_key or "").strip()
            if secret:
                self._secrets[provider_id] = secret
            else:
                self._secrets.pop(provider_id, None)

    def candidates(self, model: str, provider_id: Optional[str] = None) -> list[ProviderConfig]:
        if provider_id:
            configs = [self.get(provider_id)]
        else:
            with self._lock:
                rows = self._connection.execute(
                    "SELECT * FROM provider_configs WHERE enabled = 1 ORDER BY name, id"
                ).fetchall()
            configs = [_config_from_row(row) for row in rows]
        return [
            config
            for config in configs
            if config.enabled and (model in config.models or "*" in config.models)
        ]

    def chat(
        self,
        payload: dict[str, Any],
        *,
        provider_id: Optional[str] = None,
    ) -> ProviderResult:
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("model is required")
        candidates = self.candidates(model, provider_id)
        if not candidates:
            raise ProviderError(f"no enabled external provider supports model {model}", status=404)
        errors: list[str] = []
        for config in candidates:
            try:
                self._check_budget(config)
                secret = self._secret_for(config)
                if not secret:
                    raise ProviderError(
                        f"provider {config.name} has no API key configured",
                        status=401,
                    )
                started = time.monotonic()
                response = self.transport(
                    config,
                    "/chat/completions",
                    payload,
                    {"Authorization": f"Bearer {secret}"},
                )
                latency = max(0.0, time.monotonic() - started)
                usage = response.get("usage") if isinstance(response, dict) else None
                usage = usage if isinstance(usage, dict) else {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                cost = (
                    prompt_tokens * config.input_cost_per_million
                    + completion_tokens * config.output_cost_per_million
                ) / 1_000_000.0
                self._record_usage(config.id, model, prompt_tokens, completion_tokens, cost)
                return ProviderResult(
                    provider_id=config.id,
                    model=model,
                    response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    latency_seconds=latency,
                )
            except ProviderError as exc:
                errors.append(f"{config.name}: {exc}")
                if not exc.transient:
                    raise
        raise ProviderError("all external providers failed: " + "; ".join(errors), transient=True)

    def monthly_spend(self, provider_id: str, *, month: Optional[str] = None) -> float:
        month = month or _month_key(self.clock())
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM provider_usage WHERE provider_id = ? AND month = ?",
                (provider_id, month),
            ).fetchone()
        return float(row[0])

    def usage(self, provider_id: Optional[str] = None) -> dict[str, Any]:
        clauses = " WHERE provider_id = ?" if provider_id else ""
        values = (provider_id,) if provider_id else ()
        with self._lock:
            rows = self._connection.execute(
                "SELECT provider_id, month, SUM(requests), SUM(prompt_tokens), "
                "SUM(completion_tokens), SUM(cost_usd) FROM provider_usage"
                + clauses
                + " GROUP BY provider_id, month ORDER BY month DESC, provider_id",
                values,
            ).fetchall()
        return {
            "schema": "machboost.provider-usage.v1",
            "usage": [
                {
                    "provider_id": str(row[0]),
                    "month": str(row[1]),
                    "requests": int(row[2]),
                    "prompt_tokens": int(row[3]),
                    "completion_tokens": int(row[4]),
                    "cost_usd": round(float(row[5]), 8),
                }
                for row in rows
            ],
        }

    def _secret_for(self, config: ProviderConfig) -> Optional[str]:
        with self._lock:
            direct = self._secrets.get(config.id)
        if direct:
            return direct
        if config.api_key_env:
            return os.environ.get(config.api_key_env) or None
        return None

    def _check_budget(self, config: ProviderConfig) -> None:
        if config.monthly_budget_usd is None:
            return
        if self.monthly_spend(config.id) >= config.monthly_budget_usd:
            raise ProviderBudgetError(f"provider {config.name} reached its monthly budget")

    def _record_usage(
        self,
        provider_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO provider_usage (
                    provider_id, month, model, requests, prompt_tokens,
                    completion_tokens, cost_usd, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(provider_id, month, model) DO UPDATE SET
                    requests = requests + 1,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    cost_usd = cost_usd + excluded.cost_usd,
                    updated_at = excluded.updated_at
                """,
                (
                    provider_id,
                    _month_key(self.clock()),
                    model,
                    max(0, int(prompt_tokens)),
                    max(0, int(completion_tokens)),
                    max(0.0, float(cost_usd)),
                    self.clock(),
                ),
            )

    def _migrate(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS provider_configs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    models_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    api_key_env TEXT,
                    monthly_budget_usd REAL,
                    input_cost_per_million REAL NOT NULL,
                    output_cost_per_million REAL NOT NULL,
                    timeout_seconds REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_usage (
                    provider_id TEXT NOT NULL,
                    month TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requests INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(provider_id, month, model),
                    FOREIGN KEY(provider_id) REFERENCES provider_configs(id) ON DELETE CASCADE
                );
                """
            )


def route_with_fallback(
    mode: str,
    *,
    local: Callable[[], Any],
    external: Callable[[], Any],
) -> tuple[str, Any]:
    mode = str(mode).strip().lower()
    if mode not in ROUTE_MODES:
        raise ValueError(f"route mode must be one of: {', '.join(sorted(ROUTE_MODES))}")
    order = {
        "local_only": (("local", local),),
        "external_only": (("external", external),),
        "local_first": (("local", local), ("external", external)),
        "external_first": (("external", external), ("local", local)),
    }[mode]
    first_error: Optional[Exception] = None
    for index, (name, callback) in enumerate(order):
        try:
            return name, callback()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if index == len(order) - 1 or not _can_fallback(exc):
                raise
    raise first_error or RuntimeError("route failed")


def _can_fallback(exc: Exception) -> bool:
    if isinstance(exc, ProviderError):
        return exc.transient
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if str(getattr(exc, "reason", "")) in {
        "queue_full",
        "queue_timeout",
        "request_timeout",
        "server_unavailable",
    }:
        return True
    return bool(getattr(exc, "transient", False))


def _default_transport(
    config: ProviderConfig,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    request = urllib.request.Request(
        config.base_url + "/v1" + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(
            f"upstream HTTP {exc.code}: {detail}",
            status=exc.code,
            transient=exc.code in TRANSIENT_STATUS_CODES,
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"upstream connection failed: {exc}", transient=True) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderError("upstream returned invalid JSON", transient=True) from exc


def _config_from_row(row: sqlite3.Row) -> ProviderConfig:
    return ProviderConfig(
        id=str(row["id"]),
        name=str(row["name"]),
        base_url=str(row["base_url"]),
        models=tuple(json.loads(row["models_json"] or "[]")),
        enabled=bool(row["enabled"]),
        api_key_env=row["api_key_env"],
        monthly_budget_usd=(
            None if row["monthly_budget_usd"] is None else float(row["monthly_budget_usd"])
        ),
        input_cost_per_million=float(row["input_cost_per_million"]),
        output_cost_per_million=float(row["output_cost_per_million"]),
        timeout_seconds=float(row["timeout_seconds"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _month_key(timestamp: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(timestamp))
