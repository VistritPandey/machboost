from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence


DEFAULT_SCOPES = ("inference", "models:read", "workspaces:read")
TRACE_MODES = {"off", "metadata", "redacted", "full"}
_SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[-_ ]?key|token|password|secret)(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")


class TeamAccessError(RuntimeError):
    def __init__(self, message: str, *, reason: str, status: int = 403) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class TeamPrincipal:
    id: str
    name: str
    scopes: tuple[str, ...]
    allowed_models: tuple[str, ...]
    max_concurrent: int
    requests_per_minute: int
    kind: str = "key"

    def permits(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes

    def permits_model(self, model: str) -> bool:
        return not self.allowed_models or model in self.allowed_models

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "scopes": list(self.scopes),
            "allowed_models": list(self.allowed_models),
            "max_concurrent": self.max_concurrent,
            "requests_per_minute": self.requests_per_minute,
        }


@dataclass(frozen=True)
class TeamSettings:
    trace_mode: str = "metadata"
    retention_days: Optional[int] = 7
    max_storage_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.trace_mode not in TRACE_MODES:
            raise ValueError(f"trace_mode must be one of: {', '.join(sorted(TRACE_MODES))}")
        if self.retention_days is not None and self.retention_days < 1:
            raise ValueError("retention_days must be positive or null")
        if self.max_storage_bytes < 1:
            raise ValueError("max_storage_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CreatedTeamKey:
    token: str
    principal: TeamPrincipal
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "key": {**self.principal.to_dict(), "created_at": self.created_at},
        }


class TeamStore:
    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
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
            self._connection.close()

    def create_key(
        self,
        name: str,
        *,
        scopes: Sequence[str] = DEFAULT_SCOPES,
        allowed_models: Sequence[str] = (),
        max_concurrent: int = 2,
        requests_per_minute: int = 60,
    ) -> CreatedTeamKey:
        name = str(name).strip()
        if not name:
            raise ValueError("key name is required")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        normalized_scopes = _normalized_values(scopes)
        if not normalized_scopes:
            raise ValueError("at least one scope is required")
        token = "mbk_" + secrets.token_urlsafe(32)
        key_id = "key_" + uuid.uuid4().hex
        created_at = _timestamp(self.clock())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO api_keys (
                    id, name, token_hash, scopes, allowed_models,
                    max_concurrent, requests_per_minute, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    key_id,
                    name,
                    _token_hash(token),
                    json.dumps(normalized_scopes),
                    json.dumps(_normalized_values(allowed_models)),
                    int(max_concurrent),
                    int(requests_per_minute),
                    created_at,
                ),
            )
        return CreatedTeamKey(
            token=token,
            principal=TeamPrincipal(
                id=key_id,
                name=name,
                scopes=normalized_scopes,
                allowed_models=_normalized_values(allowed_models),
                max_concurrent=int(max_concurrent),
                requests_per_minute=int(requests_per_minute),
            ),
            created_at=created_at,
        )

    def authenticate(self, token: str) -> Optional[TeamPrincipal]:
        token = str(token).strip()
        if not token:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM api_keys WHERE token_hash = ? AND enabled = 1",
                (_token_hash(token),),
            ).fetchone()
        return _principal_from_row(row) if row is not None else None

    def list_keys(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                **_principal_from_row(row).to_dict(),
                "enabled": bool(row["enabled"]),
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
            }
            for row in rows
        ]

    def touch_key(self, key_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (_timestamp(self.clock()), key_id),
            )

    def revoke_key(self, key_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE api_keys SET enabled = 0 WHERE id = ? AND enabled = 1",
                (key_id,),
            )
        return cursor.rowcount > 0

    def settings(self) -> TeamSettings:
        with self._lock:
            rows = self._connection.execute(
                "SELECT name, value FROM settings"
            ).fetchall()
        values = {str(row["name"]): json.loads(row["value"]) for row in rows}
        return TeamSettings(
            trace_mode=str(values.get("trace_mode", "metadata")),
            retention_days=values.get("retention_days", 7),
            max_storage_bytes=int(values.get("max_storage_bytes", 256 * 1024 * 1024)),
        )

    def update_settings(
        self,
        *,
        trace_mode: Optional[str] = None,
        retention_days: Any = ...,
        max_storage_bytes: Optional[int] = None,
    ) -> TeamSettings:
        current = self.settings()
        updated = TeamSettings(
            trace_mode=current.trace_mode if trace_mode is None else trace_mode,
            retention_days=(
                current.retention_days if retention_days is ... else retention_days
            ),
            max_storage_bytes=(
                current.max_storage_bytes
                if max_storage_bytes is None
                else int(max_storage_bytes)
            ),
        )
        with self._lock, self._connection:
            for name, value in updated.to_dict().items():
                self._connection.execute(
                    """
                    INSERT INTO settings (name, value) VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET value = excluded.value
                    """,
                    (name, json.dumps(value)),
                )
        self.prune_traces()
        return updated

    def record_trace(
        self,
        *,
        request_id: str,
        principal: TeamPrincipal,
        endpoint: str,
        model: str,
        status: str,
        started_at: float,
        finished_at: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        time_to_first_token_s: Optional[float] = None,
        input_data: Any = None,
        output_text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        settings = self.settings()
        if settings.trace_mode == "off":
            return None
        trace_id = "trace_" + uuid.uuid4().hex
        stored_input: Any = None
        stored_output: Optional[str] = None
        if settings.trace_mode in {"redacted", "full"}:
            stored_input = input_data
            stored_output = output_text
        if settings.trace_mode == "redacted":
            stored_input = _redact(stored_input)
            stored_output = _redact(stored_output)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO traces (
                    id, request_id, principal_id, principal_name, endpoint, model,
                    status, started_at, finished_at, duration_seconds,
                    prompt_tokens, completion_tokens, time_to_first_token_seconds,
                    input_json, output_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    request_id,
                    principal.id,
                    principal.name,
                    endpoint,
                    model,
                    status,
                    _timestamp(started_at),
                    _timestamp(finished_at),
                    max(0.0, finished_at - started_at),
                    max(0, int(prompt_tokens)),
                    max(0, int(completion_tokens)),
                    time_to_first_token_s,
                    json.dumps(stored_input) if stored_input is not None else None,
                    stored_output,
                    json.dumps(metadata or {}),
                ),
            )
        self.prune_traces()
        return trace_id

    def list_traces(
        self,
        *,
        limit: int = 100,
        principal_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if principal_id:
            clauses.append("principal_id = ?")
            values.append(principal_id)
        if model:
            clauses.append("model = ?")
            values.append(model)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM traces{where} ORDER BY started_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [_trace_from_row(row, include_content=False) for row in rows]

    def trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
        return _trace_from_row(row, include_content=True) if row is not None else None

    def delete_traces(self, trace_ids: Optional[Sequence[str]] = None) -> int:
        with self._lock, self._connection:
            if trace_ids:
                normalized = _normalized_values(trace_ids)
                placeholders = ",".join("?" for _ in normalized)
                cursor = self._connection.execute(
                    f"DELETE FROM traces WHERE id IN ({placeholders})", normalized
                )
            else:
                cursor = self._connection.execute("DELETE FROM traces")
        return cursor.rowcount

    def create_evaluation(
        self,
        *,
        name: str,
        trace_ids: Sequence[str],
        evaluator: str,
        summary: dict[str, Any],
        scores: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        evaluation_id = "eval_" + uuid.uuid4().hex
        created_at = _timestamp(self.clock())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO evaluations (
                    id, name, evaluator, trace_ids_json, summary_json,
                    scores_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    name.strip() or "Evaluation",
                    evaluator,
                    json.dumps(_normalized_values(trace_ids)),
                    json.dumps(summary),
                    json.dumps(list(scores)),
                    created_at,
                ),
            )
        return {
            "id": evaluation_id,
            "name": name.strip() or "Evaluation",
            "evaluator": evaluator,
            "trace_ids": list(_normalized_values(trace_ids)),
            "summary": summary,
            "scores": list(scores),
            "created_at": created_at,
        }

    def list_evaluations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [_evaluation_from_row(row) for row in rows]

    def prune_traces(self) -> int:
        settings = self.settings()
        removed = 0
        with self._lock, self._connection:
            if settings.retention_days is not None:
                cutoff = self.clock() - (settings.retention_days * 86_400)
                cursor = self._connection.execute(
                    "DELETE FROM traces WHERE started_at < ?", (_timestamp(cutoff),)
                )
                removed += cursor.rowcount
            rows = self._connection.execute(
                """
                SELECT id,
                    length(coalesce(input_json, '')) +
                    length(coalesce(output_text, '')) +
                    length(coalesce(metadata_json, '')) AS bytes
                FROM traces ORDER BY started_at DESC
                """
            ).fetchall()
            total = 0
            overflow: list[str] = []
            for row in rows:
                total += int(row["bytes"] or 0)
                if total > settings.max_storage_bytes:
                    overflow.append(str(row["id"]))
            if overflow:
                placeholders = ",".join("?" for _ in overflow)
                cursor = self._connection.execute(
                    f"DELETE FROM traces WHERE id IN ({placeholders})", overflow
                )
                removed += cursor.rowcount
        return removed

    def status(self) -> dict[str, Any]:
        with self._lock:
            key_count = int(
                self._connection.execute(
                    "SELECT count(*) FROM api_keys WHERE enabled = 1"
                ).fetchone()[0]
            )
            trace_count = int(
                self._connection.execute("SELECT count(*) FROM traces").fetchone()[0]
            )
            evaluation_count = int(
                self._connection.execute(
                    "SELECT count(*) FROM evaluations"
                ).fetchone()[0]
            )
        return {
            "schema": "machboost.team-status.v1",
            "keys": key_count,
            "traces": trace_count,
            "evaluations": evaluation_count,
            "settings": self.settings().to_dict(),
        }

    def _migrate(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    scopes TEXT NOT NULL,
                    allowed_models TEXT NOT NULL,
                    max_concurrent INTEGER NOT NULL,
                    requests_per_minute INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    principal_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    time_to_first_token_seconds REAL,
                    input_json TEXT,
                    output_text TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS traces_started_at ON traces(started_at);
                CREATE INDEX IF NOT EXISTS traces_principal_id ON traces(principal_id);
                CREATE INDEX IF NOT EXISTS traces_model ON traces(model);
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    evaluator TEXT NOT NULL,
                    trace_ids_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )


class TeamAdmissionController:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.clock = clock
        self._lock = threading.RLock()
        self._active: dict[str, int] = defaultdict(int)
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    @contextmanager
    def slot(self, principal: TeamPrincipal, model: str) -> Iterator[None]:
        if not principal.permits("inference"):
            raise TeamAccessError("key lacks inference scope", reason="scope_denied")
        if not principal.permits_model(model):
            raise TeamAccessError("model is not allowed for this key", reason="model_denied")
        now = self.clock()
        with self._lock:
            history = self._requests[principal.id]
            while history and history[0] <= now - 60.0:
                history.popleft()
            if len(history) >= principal.requests_per_minute:
                raise TeamAccessError(
                    "request rate limit exceeded",
                    reason="rate_limited",
                    status=429,
                )
            if self._active[principal.id] >= principal.max_concurrent:
                raise TeamAccessError(
                    "concurrent request limit exceeded",
                    reason="concurrency_limited",
                    status=429,
                )
            history.append(now)
            self._active[principal.id] += 1
        try:
            yield
        finally:
            with self._lock:
                self._active[principal.id] = max(0, self._active[principal.id] - 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_by_principal": {
                    key: count for key, count in self._active.items() if count > 0
                }
            }


def performance_evaluation(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        raise ValueError("at least one trace is required")
    durations = sorted(float(trace.get("duration_seconds") or 0.0) for trace in traces)
    ttfts = sorted(
        float(trace["time_to_first_token_seconds"])
        for trace in traces
        if trace.get("time_to_first_token_seconds") is not None
    )
    completed = sum(trace.get("status") == "completed" for trace in traces)
    completion_tokens = sum(int(trace.get("completion_tokens") or 0) for trace in traces)
    total_duration = sum(durations)
    return {
        "trace_count": len(traces),
        "completion_rate": completed / len(traces),
        "latency_seconds": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
        },
        "time_to_first_token_seconds": {
            "p50": _percentile(ttfts, 0.50),
            "p95": _percentile(ttfts, 0.95),
        },
        "generation_tokens_per_second": (
            completion_tokens / total_duration if total_duration > 0 else 0.0
        ),
    }


def _principal_from_row(row: sqlite3.Row) -> TeamPrincipal:
    return TeamPrincipal(
        id=str(row["id"]),
        name=str(row["name"]),
        scopes=tuple(json.loads(row["scopes"])),
        allowed_models=tuple(json.loads(row["allowed_models"])),
        max_concurrent=int(row["max_concurrent"]),
        requests_per_minute=int(row["requests_per_minute"]),
    )


def _trace_from_row(row: sqlite3.Row, *, include_content: bool) -> dict[str, Any]:
    result = {
        "id": row["id"],
        "request_id": row["request_id"],
        "principal": {"id": row["principal_id"], "name": row["principal_name"]},
        "endpoint": row["endpoint"],
        "model": row["model"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_seconds": row["duration_seconds"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "time_to_first_token_seconds": row["time_to_first_token_seconds"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }
    if include_content:
        result["input"] = (
            json.loads(row["input_json"]) if row["input_json"] is not None else None
        )
        result["output"] = row["output_text"]
    return result


def _evaluation_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "evaluator": row["evaluator"],
        "trace_ids": json.loads(row["trace_ids_json"]),
        "summary": json.loads(row["summary_json"]),
        "scores": json.loads(row["scores_json"]),
        "created_at": row["created_at"],
    }


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
        return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", redacted)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if str(key).lower() in {"authorization", "api_key", "token", "password", "secret"}
            else _redact(item)
            for key, item in value.items()
        }
    return value


def _normalized_values(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _timestamp(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(len(values) * ratio + 0.999999) - 1))
    return float(values[index])
