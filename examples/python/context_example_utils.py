from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {".git", ".venv", "build", "dist", "node_modules", "target", "vendor"}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class Passage:
    source: str
    text: str
    score: int = 0


def read_text_paths(
    paths: Iterable[str],
    *,
    max_chars: int = 200_000,
    exclude: Iterable[str | Path] = (),
) -> list[Passage]:
    excluded = {Path(path).expanduser().resolve() for path in exclude}
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file()
                and candidate.suffix.lower() in TEXT_SUFFIXES
                and not any(part in SKIP_DIRS for part in candidate.parts)
            )
        else:
            raise FileNotFoundError(path)

    passages: list[Passage] = []
    remaining = max(0, int(max_chars))
    for path in dict.fromkeys(files):
        if remaining <= 0:
            break
        if path in excluded or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text = text[:remaining]
        if text:
            passages.append(Passage(source=str(path), text=text))
            remaining -= len(text)
    return passages


def split_passages(
    documents: Sequence[Passage],
    *,
    max_chars: int = 1_600,
) -> list[Passage]:
    chunks: list[Passage] = []
    for document in documents:
        paragraphs = re.split(r"\n\s*\n", document.text)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            while paragraph:
                boundary = min(len(paragraph), max_chars)
                if boundary < len(paragraph):
                    split_at = paragraph.rfind(" ", 0, boundary)
                    boundary = split_at if split_at > max_chars // 2 else boundary
                chunk = paragraph[:boundary].strip()
                if chunk:
                    chunks.append(Passage(source=document.source, text=chunk))
                paragraph = paragraph[boundary:].strip()
    return chunks


def retrieve_passages(
    question: str,
    passages: Sequence[Passage],
    *,
    limit: int = 4,
) -> list[Passage]:
    query_terms = _terms(question)
    ranked = []
    for index, passage in enumerate(passages):
        passage_terms = _terms(passage.text)
        score = sum(1 for term in query_terms if term in passage_terms)
        ranked.append((score, -index, Passage(passage.source, passage.text, score)))
    ranked.sort(reverse=True)
    return [entry[2] for entry in ranked[: max(1, int(limit))]]


def load_accelerator(
    *,
    model: str,
    backend: str,
    context_texts: Sequence[str],
    ngram: int,
    max_draft_tokens: int,
    reentry_probe_tokens: int,
    device: str | None = None,
    local_files_only: bool = False,
):
    from machboost import Accelerator

    common = {
        "context": context_texts,
        "ngram": ngram,
        "max_draft_tokens": max_draft_tokens,
        "reentry_probe_tokens": reentry_probe_tokens,
    }
    if backend == "mlx":
        return Accelerator.from_mlx(model, **common)
    if backend == "hf":
        return Accelerator.from_huggingface(
            model,
            device=device,
            local_files_only=local_files_only,
            **common,
        )
    raise ValueError(f"unsupported backend: {backend}")


def stats_line(stats) -> str:
    generated = int(stats.generated_tokens)
    accepted = int(stats.accepted_draft_tokens)
    acceptance = accepted / generated if generated else 0.0
    return (
        f"generated={generated} accepted_draft_tokens={accepted} "
        f"acceptance={acceptance:.1%} target_calls={int(stats.target_calls)} "
        f"estimated_call_speedup={float(stats.estimated_speedup):.3f}x"
    )


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9_./-]+", text.lower())
        if len(term) > 1 and term not in STOP_WORDS
    }
