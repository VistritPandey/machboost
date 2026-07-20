from __future__ import annotations

import argparse
from pathlib import Path
import sys

from context_example_utils import (
    DEFAULT_MODEL,
    load_accelerator,
    read_text_paths,
    stats_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue a source file using patterns retrieved from the rest of a repository."
    )
    parser.add_argument("--repo", default=".", help="Repository directory used as draft context.")
    parser.add_argument("--file", required=True, help="Source file whose current end is the cursor.")
    parser.add_argument("--context", action="append", default=[], help="Additional context path; repeatable.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("mlx", "hf"), default="mlx")
    parser.add_argument("--prefix-chars", type=int, default=8_000)
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=16)
    parser.add_argument("--reentry-probe-tokens", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.file).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    prefix = target.read_text(encoding="utf-8")[-max(1, args.prefix_chars) :]
    documents = read_text_paths(
        [args.repo, *args.context],
        max_chars=args.max_context_chars,
        exclude=[target],
    )
    if not documents:
        raise SystemExit("no repository context found after excluding the target file")

    accelerator = load_accelerator(
        model=args.model,
        backend=args.backend,
        context_texts=[document.text for document in documents],
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
        reentry_probe_tokens=args.reentry_probe_tokens,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    prompt = (
        "Continue the source file at the cursor. Return only code that belongs after the "
        f"current final character. File: {target.name}\n\n{prefix}"
    )
    result = accelerator.generate_result(prompt, max_tokens=args.max_tokens)
    print(result.text)
    print(stats_line(result.stats), file=sys.stderr)
    if not result.stats.accepted_draft_tokens:
        print(
            "No repository continuation was accepted, so MachBoost used native generation for this completion.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
