"""Create a limited employee key, inspect traces, and run a local evaluation."""

from __future__ import annotations

import os

from machboost import MachBoostClient


def main() -> None:
    endpoint = os.environ.get("MACHBOOST_HOST", "http://127.0.0.1:11435")
    admin_token = os.environ.get("MACHBOOST_API_TOKEN")
    if not admin_token:
        raise SystemExit("Set MACHBOOST_API_TOKEN to the team node administrator token.")

    client = MachBoostClient(endpoint, api_token=admin_token)
    created = client.create_team_key(
        "Example coding agent",
        scopes=(
            "inference",
            "models:read",
            "workspaces:read",
            "traces:read",
            "evaluations:read",
            "evaluations:write",
        ),
        allowed_models=("qwen2.5:3b",),
        max_concurrent=2,
        requests_per_minute=60,
    )
    print("Employee token (shown once):", created["token"])
    print("Key policy:", created["key"])

    traces = client.traces(limit=10)
    if not traces:
        print("No traces yet. Send a request with the employee token, then rerun.")
        return

    evaluation = client.evaluate_traces(
        [trace["id"] for trace in traces],
        name="Example gateway evaluation",
    )
    print("Evaluation:", evaluation["summary"])


if __name__ == "__main__":
    main()
