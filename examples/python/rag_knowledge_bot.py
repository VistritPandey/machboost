from __future__ import annotations

import argparse
import sys

from context_example_utils import (
    DEFAULT_MODEL,
    load_accelerator,
    read_text_paths,
    retrieve_passages,
    split_passages,
    stats_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask one grounded question over local documents with visible MachBoost stats."
    )
    parser.add_argument("question")
    parser.add_argument("--docs", action="append", required=True, help="Text file or directory; repeatable.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("mlx", "hf"), default="mlx")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--max-draft-tokens", type=int, default=16)
    parser.add_argument("--reentry-probe-tokens", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_text_paths(args.docs, max_chars=args.max_context_chars)
    if not documents:
        raise SystemExit("no readable text documents found")
    retrieved = retrieve_passages(
        args.question,
        split_passages(documents),
        limit=args.top_k,
    )
    sources = "\n\n".join(
        f"[Source: {passage.source}]\n{passage.text}" for passage in retrieved
    )
    if args.show_context:
        print("Retrieved context:\n" + sources + "\n", file=sys.stderr)

    accelerator = load_accelerator(
        model=args.model,
        backend=args.backend,
        context_texts=[passage.text for passage in retrieved],
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
        reentry_probe_tokens=args.reentry_probe_tokens,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    answer, stats = accelerator.generate_chat(
        [
            {
                "role": "user",
                "content": (
                    "Return only the smallest complete verbatim source line that answers the question. "
                    "Do not add an introduction, explanation, quotation marks, or paraphrase. "
                    "A number by itself is not a complete source line. "
                    "Return NOT FOUND when the sources do not contain an answer.\n\n"
                    f"{sources}\n\nQuestion: {args.question}"
                ),
            }
        ],
        max_tokens=args.max_tokens,
    )
    print(answer)
    print(stats_line(stats), file=sys.stderr)
    if not stats.accepted_draft_tokens:
        print(
            "MachBoost did not find an accepted context continuation; this request used the native path.",
            file=sys.stderr,
        )
    print(
        "One answer is not a benchmark. Validate representative questions with benchmark_context_workload.py.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
