from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import threading
from typing import Any, Iterable, Optional, Sequence


DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_CHUNK_CHARS = 6_000
DEFAULT_CHUNK_LINES = 120
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_QUERY_CHARS = 48_000
DEFAULT_TOP_K = 12
DEFAULT_CAPSULE_CHARS = 36_000
DEFAULT_CAPSULE_RATIO = 0.75
DEFAULT_EVIDENCE_CHARS = 8_000
DEFAULT_HIT_CHARS = 1_600
DEFAULT_HIT_CONTEXT_LINES = 12

SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}
SKIP_SUFFIXES = {
    ".7z",
    ".a",
    ".avi",
    ".bin",
    ".class",
    ".dmg",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".key",
    ".lockb",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".otf",
    ".p12",
    ".pdf",
    ".pem",
    ".pfx",
    ".png",
    ".pyc",
    ".so",
    ".tar",
    ".tgz",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
}
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
LANGUAGE_BY_SUFFIX = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".css": "CSS",
    ".dart": "Dart",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".go": "Go",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".lua": "Lua",
    ".md": "Markdown",
    ".mjs": "JavaScript",
    ".php": "PHP",
    ".proto": "Protocol Buffers",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
    ".swift": "Swift",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".vue": "Vue",
    ".xml": "XML",
    ".yaml": "YAML",
    ".yml": "YAML",
}
QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "with",
}
SYMBOL_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+"
        r"([A-Za-z_$][A-Za-z0-9_$]*)"
    ),
    re.compile(
        r"^\s*(?:public|private|protected|internal|static|final|open|abstract|\s)*"
        r"(?:class|struct|enum|interface|protocol|actor)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
    ),
    re.compile(r"^\s*(?:type|interface)\s+([A-Za-z_$][A-Za-z0-9_$]*)"),
)


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    path: str
    created_at: str
    updated_at: str
    indexed_at: Optional[str] = None
    revision: Optional[str] = None
    file_count: int = 0
    chunk_count: int = 0
    total_bytes: int = 0
    languages: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["languages"] = [
            {"name": name, "files": count} for name, count in self.languages
        ]
        return result


@dataclass(frozen=True)
class IndexReport:
    workspace: Workspace
    scanned_files: int
    indexed_files: int
    unchanged_files: int
    removed_files: int
    skipped_files: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace.to_dict(),
            "scanned_files": self.scanned_files,
            "indexed_files": self.indexed_files,
            "unchanged_files": self.unchanged_files,
            "removed_files": self.removed_files,
            "skipped_files": self.skipped_files,
        }


@dataclass(frozen=True)
class SearchHit:
    path: str
    start_line: int
    end_line: int
    symbols: tuple[str, ...]
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbols": list(self.symbols),
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True)
class WorkspaceQuery:
    workspace: Workspace
    query: str
    context: str
    hits: tuple[SearchHit, ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace.to_dict(),
            "query": self.query,
            "context": self.context,
            "hits": [hit.to_dict() for hit in self.hits],
            "truncated": self.truncated,
        }


class WorkspaceStore:
    def __init__(self, home: Optional[os.PathLike[str] | str] = None) -> None:
        self.home = Path(home).expanduser() if home else default_workspace_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def register(self, path: os.PathLike[str] | str, *, name: Optional[str] = None) -> Workspace:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise WorkspaceError(f"Workspace path is not a directory: {root}")
        workspace_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        now = _timestamp()
        with self._lock:
            previous = self._read_metadata(workspace_id)
            workspace = Workspace(
                id=workspace_id,
                name=(name or (previous.name if previous else root.name)).strip() or root.name,
                path=str(root),
                created_at=previous.created_at if previous else now,
                updated_at=now,
                indexed_at=previous.indexed_at if previous else None,
                revision=previous.revision if previous else None,
                file_count=previous.file_count if previous else 0,
                chunk_count=previous.chunk_count if previous else 0,
                total_bytes=previous.total_bytes if previous else 0,
                languages=previous.languages if previous else (),
            )
            self._workspace_dir(workspace_id).mkdir(parents=True, exist_ok=True)
            self._write_metadata(workspace)
            self._initialize_database(workspace_id)
            return workspace

    def list(self) -> list[Workspace]:
        workspaces = []
        for metadata_path in sorted(self.home.glob("*/workspace.json")):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                workspaces.append(_workspace_from_dict(payload))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(workspaces, key=lambda item: item.updated_at, reverse=True)

    def get(self, workspace_id: str) -> Workspace:
        workspace = self._read_metadata(workspace_id)
        if workspace is None:
            raise WorkspaceError(f"Unknown workspace: {workspace_id}")
        return workspace

    def file_digests(
        self,
        workspace_id: str,
        paths: Optional[Sequence[str]] = None,
    ) -> dict[str, str]:
        """Return content digests from the latest completed workspace index."""
        self.get(workspace_id)
        database_path = self._database_path(workspace_id)
        if not database_path.exists():
            return {}
        normalized = tuple(
            dict.fromkeys(str(path).strip() for path in (paths or ()) if str(path).strip())
        )
        try:
            with closing(sqlite3.connect(database_path)) as connection:
                if not normalized:
                    rows = connection.execute(
                        "SELECT path, digest FROM files ORDER BY path"
                    ).fetchall()
                else:
                    placeholders = ",".join("?" for _ in normalized)
                    rows = connection.execute(
                        f"SELECT path, digest FROM files WHERE path IN ({placeholders})",
                        normalized,
                    ).fetchall()
        except sqlite3.OperationalError as exc:
            raise WorkspaceError(f"Workspace digest lookup failed: {exc}") from exc
        return {str(path): str(digest) for path, digest in rows}

    def remove(self, workspace_id: str) -> bool:
        with self._lock:
            target = self._workspace_dir(workspace_id)
            if not target.exists():
                return False
            shutil.rmtree(target)
            return True

    def index(
        self,
        workspace_id: str,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        cancel_event: Optional[threading.Event] = None,
    ) -> IndexReport:
        with self._lock:
            workspace = self.get(workspace_id)
            root = Path(workspace.path)
            if not root.is_dir():
                raise WorkspaceError(f"Workspace is no longer available: {root}")
            self._initialize_database(workspace_id)
            discovered = tuple(_discover_files(root))
            database_path = self._database_path(workspace_id)
            indexed_files = 0
            unchanged_files = 0
            skipped_files = 0

            with closing(sqlite3.connect(database_path)) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                existing = {
                    row[0]: (int(row[1]), int(row[2]), str(row[3]))
                    for row in connection.execute(
                        "SELECT path, size, mtime_ns, digest FROM files"
                    )
                }
                current_paths: set[str] = set()
                for absolute_path, relative_path in discovered:
                    _check_cancelled(cancel_event)
                    current_paths.add(relative_path)
                    try:
                        stat = absolute_path.stat()
                    except OSError:
                        skipped_files += 1
                        continue
                    known = existing.get(relative_path)
                    if known and known[:2] == (stat.st_size, stat.st_mtime_ns):
                        unchanged_files += 1
                        continue
                    text = _read_text_file(absolute_path, max_file_bytes=max_file_bytes)
                    if text is None:
                        skipped_files += 1
                        self._delete_file(connection, relative_path)
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if known and known[2] == digest:
                        connection.execute(
                            "UPDATE files SET size = ?, mtime_ns = ? WHERE path = ?",
                            (stat.st_size, stat.st_mtime_ns, relative_path),
                        )
                        unchanged_files += 1
                        continue
                    self._replace_file(
                        connection,
                        relative_path,
                        text,
                        size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        digest=digest,
                    )
                    indexed_files += 1

                removed = sorted(set(existing) - current_paths)
                for relative_path in removed:
                    self._delete_file(connection, relative_path)
                connection.commit()

                file_count, total_bytes = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM files"
                ).fetchone()
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
                language_counts = Counter(
                    language_for_path(row[0])
                    for row in connection.execute("SELECT path FROM files")
                )
                revision = _workspace_revision(root, connection)

            updated = Workspace(
                id=workspace.id,
                name=workspace.name,
                path=workspace.path,
                created_at=workspace.created_at,
                updated_at=_timestamp(),
                indexed_at=_timestamp(),
                revision=revision,
                file_count=int(file_count),
                chunk_count=int(chunk_count),
                total_bytes=int(total_bytes),
                languages=tuple(
                    sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))
                ),
            )
            self._write_metadata(updated)
            return IndexReport(
                workspace=updated,
                scanned_files=len(discovered),
                indexed_files=indexed_files,
                unchanged_files=unchanged_files,
                removed_files=len(removed),
                skipped_files=skipped_files,
            )

    def query(
        self,
        workspace_id: str,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_chars: int = DEFAULT_QUERY_CHARS,
    ) -> WorkspaceQuery:
        workspace = self.get(workspace_id)
        terms = _query_terms(query)
        if not terms:
            return WorkspaceQuery(
                workspace=workspace,
                query=query,
                context=self.capsule(
                    workspace_id,
                    max_chars=min(32_000, max(1_000, max_chars)),
                ),
                hits=(),
                truncated=False,
            )
        expression = " OR ".join(f'"{term}"*' for term in terms)
        rows: list[Sequence[Any]]
        try:
            with closing(sqlite3.connect(self._database_path(workspace_id))) as connection:
                rows = connection.execute(
                    """
                    SELECT path, start_line, end_line, symbols, content, bm25(chunks)
                    FROM chunks
                    WHERE chunks MATCH ?
                    ORDER BY bm25(chunks)
                    LIMIT ?
                    """,
                    (expression, max(top_k * 4, top_k)),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            raise WorkspaceError(f"Workspace query failed: {exc}") from exc

        selected: list[SearchHit] = []
        per_file: Counter[str] = Counter()
        for row in rows:
            path = str(row[0])
            if per_file[path] >= 3:
                continue
            selected.append(
                SearchHit(
                    path=path,
                    start_line=int(row[1]),
                    end_line=int(row[2]),
                    symbols=_parse_symbols(str(row[3])),
                    text=str(row[4]),
                    score=round(-float(row[5]), 6),
                )
            )
            per_file[path] += 1
            if len(selected) >= max(1, top_k):
                break

        capsule = self.capsule(
            workspace_id,
            max_chars=min(
                DEFAULT_CAPSULE_CHARS,
                max(1_000, int(max_chars * DEFAULT_CAPSULE_RATIO)),
            ),
        )
        sections = [capsule]
        used = len(capsule)
        evidence_limit = min(DEFAULT_EVIDENCE_CHARS, max(0, max_chars - used))
        evidence_used = 0
        included: list[SearchHit] = []
        truncated = False
        for hit in selected:
            available = evidence_limit - evidence_used
            if available <= 0:
                truncated = True
                break
            focused = _focus_hit(
                hit,
                terms,
                max_chars=min(DEFAULT_HIT_CHARS, available),
            )
            header = (
                f"\n\n## {focused.path}:"
                f"{focused.start_line}-{focused.end_line}\n"
            )
            available -= len(header)
            if available <= 0:
                truncated = True
                break
            body = focused.text[:available]
            if len(body) < len(hit.text):
                truncated = True
            sections.append(header + body)
            used += len(header) + len(body)
            evidence_used += len(header) + len(body)
            included.append(
                SearchHit(
                    path=focused.path,
                    start_line=focused.start_line,
                    end_line=focused.end_line,
                    symbols=focused.symbols,
                    text=body,
                    score=focused.score,
                )
            )

        return WorkspaceQuery(
            workspace=workspace,
            query=query,
            context="".join(sections),
            hits=tuple(included),
            truncated=truncated,
        )

    def capsule(self, workspace_id: str, *, max_chars: int = 32_000) -> str:
        workspace = self.get(workspace_id)
        language_summary = ", ".join(
            f"{name} ({count})" for name, count in workspace.languages[:8]
        )
        revision = workspace.revision or "unavailable"
        header = (
            f"# Workspace: {workspace.name}\n"
            f"Revision: {revision}\n"
            f"Indexed files: {workspace.file_count}\n"
            f"Indexed chunks: {workspace.chunk_count}\n"
            f"Languages: {language_summary or 'unknown'}\n"
            "Use only the repository evidence below for repository-specific claims. "
            "Cite evidence as path:start-end. Say when the evidence is insufficient."
        )
        remaining = max(0, max_chars - len(header) - len("\n\n# Repository map\n"))
        if remaining <= 0:
            return header[:max_chars]

        symbols_by_path: dict[str, list[str]] = {}
        with closing(sqlite3.connect(self._database_path(workspace_id))) as connection:
            file_paths = [
                str(row[0])
                for row in connection.execute("SELECT path FROM files ORDER BY path")
            ]
            for path, payload in connection.execute(
                "SELECT path, symbols FROM chunks ORDER BY path, start_line"
            ):
                names = symbols_by_path.setdefault(str(path), [])
                for symbol in _parse_symbols(str(payload)):
                    if symbol not in names and len(names) < 24:
                        names.append(symbol)

        lines: list[str] = []
        used = 0
        for path in file_paths:
            names = symbols_by_path.get(path, ())
            line = path
            if names:
                line += ": " + ", ".join(names)
            line += "\n"
            if used + len(line) > remaining:
                omitted = len(file_paths) - len(lines)
                suffix = f"... {omitted} additional files omitted\n"
                if used + len(suffix) <= remaining:
                    lines.append(suffix)
                break
            lines.append(line)
            used += len(line)
        return header + "\n\n# Repository map\n" + "".join(lines).rstrip()

    def _initialize_database(self, workspace_id: str) -> None:
        self._workspace_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        try:
            with closing(
                sqlite3.connect(self._database_path(workspace_id))
            ) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS files (
                        path TEXT PRIMARY KEY,
                        size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        digest TEXT NOT NULL,
                        indexed_at TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                        path UNINDEXED,
                        start_line UNINDEXED,
                        end_line UNINDEXED,
                        symbols,
                        content,
                        tokenize = 'unicode61 tokenchars ''_'''
                    );
                    """
                )
        except sqlite3.OperationalError as exc:
            raise WorkspaceError(
                "This Python SQLite build does not include FTS5, which MachBoost "
                "needs for repository search."
            ) from exc

    def _replace_file(
        self,
        connection: sqlite3.Connection,
        relative_path: str,
        text: str,
        *,
        size: int,
        mtime_ns: int,
        digest: str,
    ) -> None:
        self._delete_file(connection, relative_path)
        connection.execute(
            "INSERT INTO files(path, size, mtime_ns, digest, indexed_at) VALUES (?, ?, ?, ?, ?)",
            (relative_path, size, mtime_ns, digest, _timestamp()),
        )
        for start_line, end_line, chunk_text, symbols in _chunk_text(text):
            connection.execute(
                """
                INSERT INTO chunks(path, start_line, end_line, symbols, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relative_path,
                    start_line,
                    end_line,
                    _symbol_index_payload(symbols),
                    chunk_text,
                ),
            )

    @staticmethod
    def _delete_file(connection: sqlite3.Connection, relative_path: str) -> None:
        connection.execute("DELETE FROM chunks WHERE path = ?", (relative_path,))
        connection.execute("DELETE FROM files WHERE path = ?", (relative_path,))

    def _workspace_dir(self, workspace_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{16}", workspace_id):
            raise WorkspaceError("Invalid workspace id")
        return self.home / workspace_id

    def _database_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "index.sqlite3"

    def _metadata_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "workspace.json"

    def _read_metadata(self, workspace_id: str) -> Optional[Workspace]:
        path = self._metadata_path(workspace_id)
        if not path.exists():
            return None
        try:
            return _workspace_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            raise WorkspaceError(f"Invalid workspace metadata: {path}") from exc

    def _write_metadata(self, workspace: Workspace) -> None:
        path = self._metadata_path(workspace.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(workspace.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def default_workspace_home() -> Path:
    configured = os.environ.get("MACHBOOST_WORKSPACE_HOME")
    if configured:
        return Path(configured).expanduser()
    if os.uname().sysname == "Darwin":
        return Path.home() / "Library" / "Application Support" / "MachBoost" / "workspaces"
    return Path.home() / ".cache" / "machboost" / "workspaces"


def language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in LANGUAGE_BY_SUFFIX:
        return LANGUAGE_BY_SUFFIX[suffix]
    return "Other"


def _discover_files(root: Path) -> Iterable[tuple[Path, str]]:
    git_paths = _git_files(root)
    if git_paths is not None:
        for relative_path in git_paths:
            absolute_path = root / relative_path
            if _eligible_path(absolute_path, relative_path) and absolute_path.is_file():
                yield absolute_path, relative_path
        return

    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in SKIP_DIRECTORIES and not (Path(directory) / name).is_symlink()
        )
        base = Path(directory)
        for file_name in sorted(file_names):
            absolute_path = base / file_name
            relative_path = absolute_path.relative_to(root).as_posix()
            if _eligible_path(absolute_path, relative_path):
                yield absolute_path, relative_path


def _git_files(root: Path) -> Optional[list[str]]:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    )


def _git_revision(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _workspace_revision(root: Path, connection: sqlite3.Connection) -> str:
    manifest = hashlib.sha256()
    for path, digest in connection.execute(
        "SELECT path, digest FROM files ORDER BY path"
    ):
        manifest.update(str(path).encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(str(digest).encode("ascii"))
        manifest.update(b"\n")
    content_revision = manifest.hexdigest()[:12]
    git_revision = _git_revision(root)
    return (
        f"{git_revision}-{content_revision}"
        if git_revision
        else content_revision
    )


def _eligible_path(absolute_path: Path, relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if absolute_path.is_symlink() or any(part in SKIP_DIRECTORIES for part in parts):
        return False
    name = absolute_path.name.lower()
    if name in SENSITIVE_NAMES or name.startswith(".env."):
        return False
    if absolute_path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return True


def _read_text_file(path: Path, *, max_file_bytes: int) -> Optional[str]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > max_file_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        decoded = data.decode("utf-8", errors="replace")
        if decoded.count("\ufffd") > max(3, len(decoded) // 100):
            return None
        return decoded


def _chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    max_lines: int = DEFAULT_CHUNK_LINES,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterable[tuple[int, int, str, tuple[str, ...]]]:
    lines = text.splitlines()
    if not lines:
        return
    start = 0
    while start < len(lines):
        end = start
        used = 0
        while end < len(lines) and end - start < max_lines:
            line_size = len(lines[end]) + 1
            if end > start and used + line_size > max_chars:
                break
            used += line_size
            end += 1
        if end == start:
            end += 1
        chunk_lines = lines[start:end]
        yield (
            start + 1,
            end,
            "\n".join(chunk_lines),
            _extract_symbols(chunk_lines),
        )
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)


def _focus_hit(
    hit: SearchHit,
    terms: Sequence[str],
    *,
    max_chars: int,
) -> SearchHit:
    if len(hit.text) <= max_chars:
        return hit
    lines = hit.text.splitlines()
    matching_lines = [
        index
        for index, line in enumerate(lines)
        if any(term in line.lower() for term in terms)
    ]
    center = matching_lines[0] if matching_lines else 0
    start = max(0, center - DEFAULT_HIT_CONTEXT_LINES)
    end = start
    used = 0
    while end < len(lines):
        line_size = len(lines[end]) + (1 if end > start else 0)
        if end > start and used + line_size > max_chars:
            break
        used += line_size
        end += 1
    excerpt_lines = lines[start:end]
    return SearchHit(
        path=hit.path,
        start_line=hit.start_line + start,
        end_line=hit.start_line + max(start, end - 1),
        symbols=_extract_symbols(excerpt_lines),
        text="\n".join(excerpt_lines),
        score=hit.score,
    )


def _extract_symbols(lines: Sequence[str]) -> tuple[str, ...]:
    symbols: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for pattern in SYMBOL_PATTERNS:
            match = pattern.match(line)
            if match and match.group(1) not in seen:
                seen.add(match.group(1))
                symbols.append(match.group(1))
                break
    return tuple(symbols[:32])


def _symbol_index_payload(symbols: Sequence[str]) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        for term in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+",
            symbol.replace("_", " "),
        ):
            normalized = term.lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
    return json.dumps(
        {"names": list(symbols), "terms": terms},
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_symbols(payload: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return tuple(filter(None, payload.split(" ")))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("names"), list):
        return tuple(filter(None, payload.split(" ")))
    return tuple(str(name) for name in parsed["names"] if name)


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,63}", query):
        normalized = term.lower()
        if normalized in QUERY_STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms[:24]


def _workspace_from_dict(payload: dict[str, Any]) -> Workspace:
    languages = payload.get("languages") or ()
    parsed_languages = tuple(
        (str(item["name"]), int(item["files"]))
        if isinstance(item, dict)
        else (str(item[0]), int(item[1]))
        for item in languages
    )
    return Workspace(
        id=str(payload["id"]),
        name=str(payload["name"]),
        path=str(payload["path"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        indexed_at=payload.get("indexed_at"),
        revision=payload.get("revision"),
        file_count=int(payload.get("file_count", 0)),
        chunk_count=int(payload.get("chunk_count", 0)),
        total_bytes=int(payload.get("total_bytes", 0)),
        languages=parsed_languages,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise WorkspaceError("Workspace indexing cancelled")
