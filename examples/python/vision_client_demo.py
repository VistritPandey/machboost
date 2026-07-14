from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from machboost import MachBoostClient, ensure_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask repeated questions over one image through the resident MLX-VLM backend."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5-vl:3b")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--endpoint")
    parser.add_argument(
        "prompt",
        nargs="*",
        default=[
            "Describe the image in one short sentence.",
            "List the most important visible text.",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    client, _ = ensure_server(args.endpoint)
    client.load(args.model, options={"vision_cache_size": 20}, keep_alive="forever")

    for prompt in args.prompt:
        response = ask(client, args.model, image, prompt, max_tokens=args.max_tokens)
        stats = dict((response.get("machboost") or {}).get("stats") or {})
        print(f"\nQuestion: {prompt}")
        print(f"Answer: {(response.get('message') or {}).get('content', '').strip()}")
        print(
            "Stats: "
            f"{float(stats.get('total_duration_seconds') or 0.0):.3f}s, "
            f"vision_hit={bool(stats.get('visual_cache_hit'))}, "
            f"prefix_tokens={int(stats.get('prompt_cache_prefix_tokens') or 0)}"
        )


def ask(
    client: MachBoostClient,
    model: str,
    image: Path,
    prompt: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    response = client.chat(
        model,
        [{"role": "user", "content": prompt}],
        images=[str(image)],
        options={"temperature": 0.0, "num_predict": max_tokens},
        keep_alive="forever",
        stream=False,
    )
    if not isinstance(response, dict):
        raise TypeError("expected a non-streaming response")
    return response


if __name__ == "__main__":
    main()
