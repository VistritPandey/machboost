"""Stream one fresh prompt through MachBoost's target-verified DFlash backend."""

from __future__ import annotations

import argparse
import json

from machboost import DFlashAccelerator, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Explain why verified decoding is lossless.")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--draft-quant", default="w4:gs64")
    args = parser.parse_args()

    resolution = resolve_model(args.model, "dflash")
    accelerator = DFlashAccelerator.from_pretrained(
        resolution.model,
        draft_quant=args.draft_quant,
    )
    try:
        _, stats = accelerator.generate_chat(
            [{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            enable_thinking=False,
            on_text=lambda chunk: print(chunk, end="", flush=True),
        )
        print()
        print(json.dumps(stats.to_dict(), indent=2))
    finally:
        accelerator.close()


if __name__ == "__main__":
    main()
