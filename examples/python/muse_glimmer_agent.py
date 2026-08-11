"""Exercise Muse Glimmer reasoning, tool calling, and optional vision."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from machboost import MachBoostClient


MODEL = "muse-glimmer:30b-mlx"


def local_build_status(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "status": "passing",
        "revision": "example-7f31c2a",
    }


def print_response(label: str, response: dict[str, Any]) -> None:
    message = response.get("message") or {}
    thinking = str(message.get("thinking") or "").strip()
    content = str(message.get("content") or "").strip()
    if thinking:
        print(f"\n{label} reasoning:\n{thinking}")
    if content:
        print(f"\n{label} answer:\n{content}")
    stats = (response.get("machboost") or {}).get("stats") or {}
    print(
        f"\n{label} stats: tokens={response.get('eval_count', 0)} "
        f"decode={float(stats.get('generation_tokens_per_second') or 0.0):.2f} tok/s"
    )


def run_tool_round(client: MachBoostClient, args: argparse.Namespace) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_build_status",
                "description": "Return the current local build status for a service.",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        }
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Use get_build_status for the checkout service. "
                "Then report its status and revision."
            ),
        }
    ]
    first = client.chat(
        args.model,
        messages,
        tools=tools,
        think=args.think,
        options={"backend": "ollama-mlx", "num_predict": args.max_tokens},
        keep_alive=args.keep_alive,
        stream=False,
    )
    tool_calls = list((first.get("message") or {}).get("tool_calls") or ())
    if not tool_calls:
        raise RuntimeError("Muse Glimmer did not return the requested tool call")
    call = tool_calls[0]
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    result = local_build_status(str(arguments.get("service") or "checkout"))
    print("tool call:", json.dumps(call, indent=2, sort_keys=True))
    print("local result:", json.dumps(result, sort_keys=True))

    messages.extend(
        [
            {
                "role": "assistant",
                "content": str((first.get("message") or {}).get("content") or ""),
                "tool_calls": tool_calls,
            },
            {
                "role": "tool",
                "tool_name": str(function.get("name") or "get_build_status"),
                "content": json.dumps(result),
            },
        ]
    )
    final = client.chat(
        args.model,
        messages,
        tools=tools,
        think=args.think,
        options={"backend": "ollama-mlx", "num_predict": args.max_tokens},
        keep_alive=args.keep_alive,
        stream=False,
    )
    print_response("tool round", final)


def run_vision_round(client: MachBoostClient, args: argparse.Namespace) -> None:
    response = client.chat(
        args.model,
        [
            {
                "role": "user",
                "content": "Describe the main subject and visible action in one sentence.",
            }
        ],
        images=[args.image],
        think=args.think,
        options={"backend": "ollama-mlx", "num_predict": args.max_tokens},
        keep_alive=args.keep_alive,
        stream=False,
    )
    print_response("vision round", response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default=os.environ.get("MACHBOOST_HOST"))
    parser.add_argument("--image", help="Local path, URL, data URL, or base64 image.")
    parser.add_argument("--think", choices=("low", "medium", "high", "xhigh"), default="medium")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--keep-alive", default="30m")
    args = parser.parse_args()

    client = MachBoostClient(args.endpoint, timeout=1_200)
    preflight = client.show(args.model, preflight=True, backend="ollama-mlx")
    status = preflight.get("preflight") or preflight
    if not status.get("runtime_available"):
        raise SystemExit(status.get("reason") or "Install current Ollama first.")
    if not status.get("cached"):
        raise SystemExit(f"Run `machboost pull {args.model}` before this example.")

    client.load(
        args.model,
        options={"backend": "ollama-mlx"},
        keep_alive=args.keep_alive,
        warmup=True,
    )
    run_tool_round(client, args)
    if args.image:
        run_vision_round(client, args)


if __name__ == "__main__":
    main()
