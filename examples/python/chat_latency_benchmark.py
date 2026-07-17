from __future__ import annotations

import argparse
import json

from machboost import benchmark_chat_latency, ensure_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare client-observed MachBoost and Ollama chat latency."
    )
    parser.add_argument("model", nargs="?", default="llama3.2:3b")
    parser.add_argument("--ollama-model")
    parser.add_argument("--engine", choices=("machboost", "ollama", "both"), default="both")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    client = None
    if args.engine in {"machboost", "both"}:
        client, _ = ensure_server()
    artifact = benchmark_chat_latency(
        args.model,
        ollama_model=args.ollama_model,
        prompt="Reply to this greeting naturally in one short sentence: hey",
        system="Answer directly and concisely.",
        runs=args.runs,
        warmups=args.warmups,
        max_tokens=args.max_tokens,
        engine=args.engine,
        machboost_client=client,
    )
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
