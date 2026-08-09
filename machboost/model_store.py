from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class StoredModel:
    name: str
    source: str
    system: str
    template: str
    options: dict[str, Any]
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelStore:
    """Lightweight Ollama-style aliases over existing native model repositories."""

    def __init__(self, path: str | Path, *, clock=time.time) -> None:
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
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_aliases (
                    name TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    system TEXT NOT NULL,
                    template TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create(
        self,
        name: str,
        source: str,
        *,
        system: str = "",
        template: str = "",
        options: Optional[dict[str, Any]] = None,
    ) -> StoredModel:
        name = _model_name(name)
        source = _model_name(source)
        normalized_options = dict(options or {})
        json.dumps(normalized_options)
        now = float(self.clock())
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at FROM model_aliases WHERE name = ?", (name,)
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            self._connection.execute(
                """
                INSERT INTO model_aliases (
                    name, source, system, template, options_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    source = excluded.source,
                    system = excluded.system,
                    template = excluded.template,
                    options_json = excluded.options_json,
                    updated_at = excluded.updated_at
                """,
                (
                    name,
                    source,
                    str(system),
                    str(template),
                    json.dumps(normalized_options, sort_keys=True),
                    created_at,
                    now,
                ),
            )
        return self.get(name)

    def get(self, name: str) -> StoredModel:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM model_aliases WHERE name = ?", (_model_name(name),)
            ).fetchone()
        if row is None:
            raise KeyError(name)
        return _from_row(row)

    def find(self, name: str) -> Optional[StoredModel]:
        try:
            return self.get(name)
        except KeyError:
            return None

    def list(self) -> list[StoredModel]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM model_aliases ORDER BY name"
            ).fetchall()
        return [_from_row(row) for row in rows]

    def copy(self, source: str, destination: str) -> StoredModel:
        original = self.get(source)
        return self.create(
            destination,
            original.source,
            system=original.system,
            template=original.template,
            options=original.options,
        )

    def delete(self, name: str) -> bool:
        with self._lock, self._connection:
            removed = self._connection.execute(
                "DELETE FROM model_aliases WHERE name = ?", (_model_name(name),)
            ).rowcount
        return bool(removed)

    def resolve(self, name: str) -> tuple[str, Optional[StoredModel]]:
        stored = self.find(name)
        return (stored.source, stored) if stored is not None else (name, None)


def apply_stored_model(
    model: StoredModel,
    options: dict[str, Any],
) -> dict[str, Any]:
    merged = {**model.options, **options}
    if model.system and "_system" not in merged:
        merged["_system"] = model.system
    if model.template and "_template" not in merged:
        merged["_template"] = model.template
    return merged


def _model_name(value: str) -> str:
    normalized = str(value).strip()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError("model name must be non-empty and contain no whitespace")
    return normalized


def _from_row(row: sqlite3.Row) -> StoredModel:
    return StoredModel(
        name=str(row["name"]),
        source=str(row["source"]),
        system=str(row["system"]),
        template=str(row["template"]),
        options=dict(json.loads(row["options_json"] or "{}")),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )
