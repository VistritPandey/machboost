from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


MEMORY_KINDS = {
    "decision",
    "experience",
    "fact",
    "failure",
    "fix",
    "procedure",
    "tool_result",
}
MEMORY_SCOPES = {"private", "team"}
_WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{1,80}")
_SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[-_ ]?key|token|password|secret)(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")


@dataclass(frozen=True)
class CacheNamespace:
    organization: str
    workspace_id: Optional[str]
    revision: Optional[str]
    scope: str
    principal_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.scope not in MEMORY_SCOPES:
            raise ValueError("cache scope must be private or team")
        if self.scope == "private" and not self.principal_id:
            raise ValueError("private cache namespaces require a principal_id")

    @property
    def key(self) -> str:
        payload = {
            "organization": self.organization,
            "workspace_id": self.workspace_id,
            "revision": self.revision,
            "scope": self.scope,
            "principal_id": self.principal_id if self.scope == "private" else None,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key}


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    namespace: str
    workspace_id: str
    scope: str
    principal_id: Optional[str]
    kind: str
    title: str
    content: str
    query_text: str
    revision: Optional[str]
    dependencies: dict[str, str]
    evidence: tuple[str, ...]
    confidence: float
    validated_by: tuple[str, ...]
    pinned: bool
    created_at: float
    updated_at: float
    expires_at: Optional[float]
    hit_count: int = 0
    last_used_at: Optional[float] = None
    stale: bool = False
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = list(self.evidence)
        result["validated_by"] = list(self.validated_by)
        return result


@dataclass(frozen=True)
class MemorySearch:
    query: str
    records: tuple[MemoryRecord, ...]
    context: str
    truncated: bool
    stale_rejected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "records": [record.to_dict() for record in self.records],
            "context": self.context,
            "truncated": self.truncated,
            "stale_rejected": self.stale_rejected,
        }


class TeamMemoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        max_storage_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.max_storage_bytes = max(1, int(max_storage_bytes))
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

    def put(
        self,
        *,
        namespace: str,
        workspace_id: str,
        scope: str,
        principal_id: Optional[str],
        kind: str,
        title: str,
        content: str,
        query_text: str = "",
        revision: Optional[str] = None,
        dependencies: Optional[Mapping[str, str]] = None,
        evidence: Sequence[str] = (),
        confidence: float = 0.5,
        validated_by: Sequence[str] = (),
        pinned: bool = False,
        ttl_seconds: Optional[float] = None,
    ) -> MemoryRecord:
        kind = str(kind).strip().lower()
        scope = str(scope).strip().lower()
        if kind not in MEMORY_KINDS:
            raise ValueError(f"unsupported memory kind: {kind}")
        if scope not in MEMORY_SCOPES:
            raise ValueError("memory scope must be private or team")
        if scope == "private" and not principal_id:
            raise ValueError("private memories require a principal_id")
        workspace_id = str(workspace_id).strip()
        title = _redact(str(title).strip())[:240]
        content = _redact(str(content).strip())
        query_text = _redact(str(query_text).strip())
        if not workspace_id or not title or not content:
            raise ValueError("workspace_id, title, and content are required")
        confidence = float(confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if ttl_seconds is not None and float(ttl_seconds) <= 0:
            raise ValueError("ttl_seconds must be positive or null")
        normalized_dependencies = {
            str(path): str(digest)
            for path, digest in (dependencies or {}).items()
            if str(path) and str(digest)
        }
        normalized_evidence = _unique_values(evidence)
        normalized_validation = _unique_values(validated_by)
        now = self.clock()
        expires_at = None if ttl_seconds is None else now + float(ttl_seconds)
        content_hash = hashlib.sha256(
            _canonical_json(
                {
                    "namespace": namespace,
                    "workspace_id": workspace_id,
                    "scope": scope,
                    "principal_id": principal_id if scope == "private" else None,
                    "kind": kind,
                    "title": title,
                    "content": content,
                    "revision": revision,
                    "dependencies": normalized_dependencies,
                }
            ).encode("utf-8")
        ).hexdigest()
        memory_id = "mem_" + uuid.uuid4().hex
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT id, created_at, hit_count, last_used_at FROM memories WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if previous is not None:
                memory_id = str(previous["id"])
                created_at = float(previous["created_at"])
                hit_count = int(previous["hit_count"])
                last_used_at = previous["last_used_at"]
                self._connection.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
                self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                created_at = now
                hit_count = 0
                last_used_at = None
            self._connection.execute(
                """
                INSERT INTO memories (
                    id, content_hash, namespace, workspace_id, scope, principal_id,
                    kind, title, content, query_text, revision, dependencies_json,
                    evidence_json, confidence, validated_by_json, pinned,
                    created_at, updated_at, expires_at, hit_count, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    content_hash,
                    namespace,
                    workspace_id,
                    scope,
                    principal_id if scope == "private" else None,
                    kind,
                    title,
                    content,
                    query_text,
                    revision,
                    json.dumps(normalized_dependencies, sort_keys=True),
                    json.dumps(normalized_evidence),
                    confidence,
                    json.dumps(normalized_validation),
                    1 if pinned else 0,
                    created_at,
                    now,
                    expires_at,
                    hit_count,
                    last_used_at,
                ),
            )
            self._connection.execute(
                "INSERT INTO memory_fts(id, title, content, query_text, evidence) VALUES (?, ?, ?, ?, ?)",
                (
                    memory_id,
                    title,
                    content,
                    query_text,
                    " ".join(normalized_evidence),
                ),
            )
        self.increment_metric(namespace, "memory_writes")
        self.prune()
        return self.get(memory_id, principal_id=principal_id, admin=True)

    def get(
        self,
        memory_id: str,
        *,
        principal_id: Optional[str] = None,
        admin: bool = False,
    ) -> MemoryRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None or not _record_visible(row, principal_id=principal_id, admin=admin):
            raise KeyError(memory_id)
        return _record_from_row(row)

    def search(
        self,
        *,
        namespace: str,
        workspace_id: str,
        query: str,
        revision: Optional[str],
        dependency_digests: Optional[Mapping[str, str]] = None,
        principal_id: Optional[str] = None,
        include_stale: bool = False,
        limit: int = 8,
        max_chars: int = 12_000,
    ) -> MemorySearch:
        self.prune_expired()
        terms = _query_terms(query)
        parameters: list[Any] = [namespace, workspace_id]
        visibility = "(scope = 'team' OR (scope = 'private' AND principal_id = ?))"
        parameters.append(principal_id or "")
        rows: list[sqlite3.Row]
        with self._lock:
            if terms:
                expression = " OR ".join(f'"{term}"*' for term in terms[:24])
                rows = self._connection.execute(
                    f"""
                    SELECT memories.*, bm25(memory_fts) AS lexical_score
                    FROM memory_fts JOIN memories ON memories.id = memory_fts.id
                    WHERE memory_fts MATCH ? AND memories.namespace = ?
                        AND memories.workspace_id = ? AND {visibility}
                    ORDER BY lexical_score
                    LIMIT ?
                    """,
                    [expression, *parameters, max(50, int(limit) * 8)],
                ).fetchall()
            else:
                rows = self._connection.execute(
                    f"""
                    SELECT memories.*, 0.0 AS lexical_score FROM memories
                    WHERE namespace = ? AND workspace_id = ? AND {visibility}
                    ORDER BY pinned DESC, updated_at DESC LIMIT ?
                    """,
                    [*parameters, max(50, int(limit) * 8)],
                ).fetchall()
        current_digests = dict(dependency_digests or {})
        candidates: list[MemoryRecord] = []
        stale_rejected = 0
        now = self.clock()
        for row in rows:
            record = _record_from_row(row)
            stale = _is_stale(record, revision, current_digests)
            if stale and not include_stale:
                stale_rejected += 1
                continue
            lexical = max(0.0, -float(row["lexical_score"] or 0.0))
            age_days = max(0.0, (now - record.updated_at) / 86_400.0)
            freshness = 1.0 / (1.0 + age_days / 30.0)
            validation = min(0.35, len(record.validated_by) * 0.12)
            revision_bonus = 0.25 if record.revision and record.revision == revision else 0.0
            score = (
                lexical
                + record.confidence
                + validation
                + revision_bonus
                + (0.4 if record.pinned else 0.0)
                + min(0.25, math.log1p(record.hit_count) / 10.0)
                + freshness * 0.3
                - (1.0 if stale else 0.0)
            )
            candidates.append(
                MemoryRecord(**{**asdict(record), "stale": stale, "score": round(score, 6)})
            )
        selected = tuple(sorted(candidates, key=lambda item: item.score, reverse=True)[: max(1, int(limit))])
        context, truncated, used_ids = _render_context(selected, max_chars=max_chars)
        if used_ids:
            with self._lock, self._connection:
                placeholders = ",".join("?" for _ in used_ids)
                self._connection.execute(
                    f"UPDATE memories SET hit_count = hit_count + 1, last_used_at = ? WHERE id IN ({placeholders})",
                    [now, *used_ids],
                )
            self.increment_metric(namespace, "memory_hits", len(used_ids))
        else:
            self.increment_metric(namespace, "memory_misses")
        return MemorySearch(
            query=query,
            records=tuple(record for record in selected if record.id in used_ids),
            context=context,
            truncated=truncated,
            stale_rejected=stale_rejected,
        )

    def list(
        self,
        *,
        namespace: Optional[str] = None,
        workspace_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        admin: bool = False,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        clauses: list[str] = []
        values: list[Any] = []
        if namespace:
            clauses.append("namespace = ?")
            values.append(namespace)
        if workspace_id:
            clauses.append("workspace_id = ?")
            values.append(workspace_id)
        if not admin:
            clauses.append("(scope = 'team' OR (scope = 'private' AND principal_id = ?))")
            values.append(principal_id or "")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memories{where} ORDER BY pinned DESC, updated_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def delete(
        self,
        memory_ids: Sequence[str],
        *,
        principal_id: Optional[str] = None,
        admin: bool = False,
    ) -> int:
        normalized = _unique_values(memory_ids)
        removed = 0
        with self._lock, self._connection:
            for memory_id in normalized:
                row = self._connection.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                if row is None or not _record_visible(
                    row, principal_id=principal_id, admin=admin
                ):
                    continue
                self._connection.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
                self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                removed += 1
        return removed

    def put_exact(
        self,
        *,
        namespace: str,
        workspace_id: Optional[str],
        revision: Optional[str],
        model: str,
        request: Any,
        response: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        ttl_seconds: float = 3600.0,
    ) -> str:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        key = request_fingerprint(
            namespace=namespace,
            workspace_id=workspace_id,
            revision=revision,
            model=model,
            request=request,
        )
        now = self.clock()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO exact_cache (
                    cache_key, namespace, workspace_id, revision, model,
                    response_json, prompt_tokens, completion_tokens, cost_usd,
                    created_at, expires_at, hit_count, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    cost_usd = excluded.cost_usd,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    namespace,
                    workspace_id,
                    revision,
                    model,
                    _canonical_json(response),
                    max(0, int(prompt_tokens)),
                    max(0, int(completion_tokens)),
                    max(0.0, float(cost_usd)),
                    now,
                    now + float(ttl_seconds),
                ),
            )
        self.increment_metric(namespace, "exact_cache_writes")
        return key

    def get_exact(
        self,
        *,
        namespace: str,
        workspace_id: Optional[str],
        revision: Optional[str],
        model: str,
        request: Any,
    ) -> Optional[dict[str, Any]]:
        key = request_fingerprint(
            namespace=namespace,
            workspace_id=workspace_id,
            revision=revision,
            model=model,
            request=request,
        )
        now = self.clock()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM exact_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None or float(row["expires_at"]) <= now:
                if row is not None:
                    self._connection.execute(
                        "DELETE FROM exact_cache WHERE cache_key = ?", (key,)
                    )
                self.increment_metric(namespace, "exact_cache_misses")
                return None
            self._connection.execute(
                "UPDATE exact_cache SET hit_count = hit_count + 1, last_used_at = ? WHERE cache_key = ?",
                (now, key),
            )
        self.increment_metric(namespace, "exact_cache_hits")
        self.increment_metric(namespace, "avoided_prompt_tokens", int(row["prompt_tokens"]))
        self.increment_metric(
            namespace, "avoided_completion_tokens", int(row["completion_tokens"])
        )
        self.increment_metric(namespace, "avoided_cost_microusd", int(float(row["cost_usd"]) * 1_000_000))
        return {
            "cache_key": key,
            "response": json.loads(row["response_json"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "cost_usd": float(row["cost_usd"]),
            "created_at": float(row["created_at"]),
            "hit_count": int(row["hit_count"]) + 1,
        }

    def increment_metric(self, namespace: str, name: str, amount: int = 1) -> None:
        if not name or amount == 0:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO cache_metrics(namespace, name, value) VALUES (?, ?, ?)
                ON CONFLICT(namespace, name) DO UPDATE SET value = value + excluded.value
                """,
                (namespace, name, int(amount)),
            )

    def metrics(self, namespace: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            if namespace:
                rows = self._connection.execute(
                    "SELECT namespace, name, value FROM cache_metrics WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT namespace, name, value FROM cache_metrics"
                ).fetchall()
        by_namespace: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}
        for row in rows:
            group = by_namespace.setdefault(str(row["namespace"]), {})
            group[str(row["name"])] = int(row["value"])
            totals[str(row["name"])] = totals.get(str(row["name"]), 0) + int(row["value"])
        return {
            "schema": "machboost.cache-metrics.v1",
            "totals": totals,
            "namespaces": by_namespace,
        }

    def status(self) -> dict[str, Any]:
        self.prune_expired()
        with self._lock:
            memory_count, memory_bytes = self._connection.execute(
                """
                SELECT count(*), coalesce(sum(length(title) + length(content) +
                    length(query_text) + length(dependencies_json) + length(evidence_json)), 0)
                FROM memories
                """
            ).fetchone()
            exact_count = int(
                self._connection.execute("SELECT count(*) FROM exact_cache").fetchone()[0]
            )
        return {
            "schema": "machboost.memory-status.v1",
            "memories": int(memory_count),
            "memory_bytes": int(memory_bytes),
            "exact_cache_entries": exact_count,
            "max_storage_bytes": self.max_storage_bytes,
            "metrics": self.metrics(),
        }

    def prune_expired(self) -> int:
        now = self.clock()
        with self._lock, self._connection:
            expired = [
                str(row[0])
                for row in self._connection.execute(
                    "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
            ]
            for memory_id in expired:
                self._connection.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
            self._connection.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            exact = self._connection.execute(
                "DELETE FROM exact_cache WHERE expires_at <= ?", (now,)
            ).rowcount
        return len(expired) + exact

    def prune(self) -> int:
        removed = self.prune_expired()
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT id, namespace, pinned, confidence, hit_count, updated_at,
                    length(title) + length(content) + length(query_text) +
                    length(dependencies_json) + length(evidence_json) AS bytes
                FROM memories
                """
            ).fetchall()
            total = sum(int(row["bytes"] or 0) for row in rows)
            if total <= self.max_storage_bytes:
                return removed
            namespace_bytes: dict[str, int] = {}
            for row in rows:
                key = str(row["namespace"])
                namespace_bytes[key] = namespace_bytes.get(key, 0) + int(row["bytes"] or 0)
            now = self.clock()
            candidates = []
            for row in rows:
                if bool(row["pinned"]):
                    continue
                size = max(1, int(row["bytes"] or 0))
                age_days = max(0.0, (now - float(row["updated_at"])) / 86_400.0)
                utility = (
                    float(row["confidence"])
                    + math.log1p(int(row["hit_count"]))
                    + 1.0 / (1.0 + age_days)
                ) / size
                over_share = namespace_bytes[str(row["namespace"])] > self.max_storage_bytes * 0.7
                candidates.append((not over_share, utility, str(row["id"]), str(row["namespace"]), size))
            for _, _, memory_id, namespace, size in sorted(candidates):
                if total <= self.max_storage_bytes:
                    break
                self._connection.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
                self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                namespace_bytes[namespace] = max(0, namespace_bytes[namespace] - size)
                total -= size
                removed += 1
        return removed

    def _migrate(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    namespace TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    principal_id TEXT,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    revision TEXT,
                    dependencies_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    validated_by_json TEXT NOT NULL,
                    pinned INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at REAL
                );
                CREATE INDEX IF NOT EXISTS memories_namespace_workspace
                    ON memories(namespace, workspace_id);
                CREATE INDEX IF NOT EXISTS memories_expiry ON memories(expires_at);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    id UNINDEXED, title, content, query_text, evidence,
                    tokenize = 'unicode61 tokenchars ''_./:-'''
                );
                CREATE TABLE IF NOT EXISTS exact_cache (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    workspace_id TEXT,
                    revision TEXT,
                    model TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at REAL
                );
                CREATE INDEX IF NOT EXISTS exact_cache_expiry ON exact_cache(expires_at);
                CREATE TABLE IF NOT EXISTS cache_metrics (
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    PRIMARY KEY(namespace, name)
                );
                """
            )


def request_fingerprint(
    *,
    namespace: str,
    workspace_id: Optional[str],
    revision: Optional[str],
    model: str,
    request: Any,
) -> str:
    payload = {
        "schema": "machboost.exact-cache.v1",
        "namespace": namespace,
        "workspace_id": workspace_id,
        "revision": revision,
        "model": model,
        "request": request,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def exchange_memory(
    *,
    user_text: str,
    assistant_text: str,
    evidence: Sequence[str] = (),
    validated_by: Sequence[str] = (),
    max_chars: int = 6_000,
) -> dict[str, Any]:
    user_text = _redact(str(user_text).strip())
    assistant_text = _redact(str(assistant_text).strip())
    title_line = next((line.strip() for line in user_text.splitlines() if line.strip()), "Completed task")
    title = title_line[:160]
    content = f"Request: {user_text}\n\nOutcome: {assistant_text}"
    if len(content) > max_chars:
        user_budget = min(len(user_text), max_chars // 3)
        output_budget = max(0, max_chars - user_budget - 22)
        content = f"Request: {user_text[:user_budget]}\n\nOutcome: {assistant_text[:output_budget]}"
    kind = "fix" if re.search(r"(?i)\b(fix|bug|error|fail|regression)\b", user_text) else "experience"
    confidence = 0.8 if validated_by else 0.55
    return {
        "kind": kind,
        "title": title,
        "content": content,
        "query_text": user_text,
        "evidence": list(_unique_values(evidence)),
        "validated_by": list(_unique_values(validated_by)),
        "confidence": confidence,
    }


def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=str(row["id"]),
        namespace=str(row["namespace"]),
        workspace_id=str(row["workspace_id"]),
        scope=str(row["scope"]),
        principal_id=row["principal_id"],
        kind=str(row["kind"]),
        title=str(row["title"]),
        content=str(row["content"]),
        query_text=str(row["query_text"]),
        revision=row["revision"],
        dependencies=dict(json.loads(row["dependencies_json"] or "{}")),
        evidence=tuple(json.loads(row["evidence_json"] or "[]")),
        confidence=float(row["confidence"]),
        validated_by=tuple(json.loads(row["validated_by_json"] or "[]")),
        pinned=bool(row["pinned"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        expires_at=None if row["expires_at"] is None else float(row["expires_at"]),
        hit_count=int(row["hit_count"]),
        last_used_at=None if row["last_used_at"] is None else float(row["last_used_at"]),
    )


def _record_visible(
    row: sqlite3.Row,
    *,
    principal_id: Optional[str],
    admin: bool,
) -> bool:
    return admin or str(row["scope"]) == "team" or (
        str(row["scope"]) == "private" and row["principal_id"] == principal_id
    )


def _is_stale(
    record: MemoryRecord,
    revision: Optional[str],
    dependency_digests: Mapping[str, str],
) -> bool:
    if record.dependencies:
        return any(
            dependency_digests.get(path) != digest
            for path, digest in record.dependencies.items()
        )
    return bool(record.revision and revision and record.revision != revision)


def _render_context(
    records: Sequence[MemoryRecord], *, max_chars: int
) -> tuple[str, bool, tuple[str, ...]]:
    max_chars = max(0, int(max_chars))
    if max_chars <= 0 or not records:
        return "", bool(records), ()
    header = (
        "# Relevant team memory\n"
        "Treat these as leads backed by their evidence, not as current source code. "
        "Prefer current repository evidence when they conflict.\n"
    )
    if len(header) > max_chars:
        return header[:max_chars], True, ()
    sections = [header]
    used = len(header)
    used_ids: list[str] = []
    truncated = False
    for record in records:
        validation = ", ".join(record.validated_by) or "unverified"
        evidence = ", ".join(record.evidence) or "none"
        section = (
            f"\n## {record.title}\n"
            f"Type: {record.kind}; confidence: {record.confidence:.2f}; "
            f"validation: {validation}; evidence: {evidence}\n"
            f"{record.content}\n"
        )
        remaining = max_chars - used
        if remaining <= 0:
            truncated = True
            break
        if len(section) > remaining:
            if remaining >= 160:
                sections.append(section[:remaining])
                used_ids.append(record.id)
            truncated = True
            break
        sections.append(section)
        used_ids.append(record.id)
        used += len(section)
    return "".join(sections).rstrip(), truncated, tuple(used_ids)


def _query_terms(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token.lower()
            for token in _WORD_PATTERN.findall(str(value))
            if len(token) >= 2
        )
    )


def _unique_values(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _redact(value: str) -> str:
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", redacted)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
