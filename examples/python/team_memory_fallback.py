#!/usr/bin/env python3
"""Configure and exercise MachBoost team memory with optional provider fallback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from machboost import MachBoostClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--provider-base-url", default=os.environ.get("EXTERNAL_BASE_URL"))
    args = parser.parse_args()

    token = os.environ.get("MACHBOOST_API_TOKEN")
    if not token:
        parser.error("set MACHBOOST_API_TOKEN to the team administrator token")

    admin = MachBoostClient(args.endpoint, api_token=token)
    workspace = admin.register_workspace(args.repository.resolve(), name=args.repository.name)
    workspace_id = workspace["id"]

    admin.create_memory(
        workspace_id,
        "Safe checkout retry",
        "Retry gateway timeouts with the original idempotency key.",
        scope="team",
        kind="procedure",
        query_text="checkout payment timeout duplicate charge retry",
        confidence=0.95,
        validated_by=("review", "ci"),
        pinned=True,
    )

    provider_id = None
    if args.provider_base_url:
        provider = admin.configure_provider(
            "Optional external fallback",
            args.provider_base_url,
            (args.model,),
            api_key_env="EXTERNAL_API_KEY",
            monthly_budget_usd=25.0,
        )
        provider_id = provider["id"]

    route = (
        {"mode": "local_first", "provider_id": provider_id}
        if provider_id
        else {"mode": "local_only"}
    )
    response = admin.chat(
        args.model,
        [{"role": "user", "content": "How should checkout timeouts be retried?"}],
        workspace_id=workspace_id,
        machboost={
            "memory": {
                "mode": "private",
                "search": True,
                "remember": True,
                "exact_cache": True,
            },
            "route": route,
        },
        stream=False,
    )

    print(response["message"]["content"])
    print(json.dumps(admin.cache_metrics(), indent=2, sort_keys=True))
    print(
        "Exact-cache counters report avoided work for eligible repeated requests; "
        "they are not a model decode-speed measurement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
