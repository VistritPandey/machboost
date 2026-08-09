#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machboost.memory import CacheNamespace, TeamMemoryStore


DEVELOPERS = ("alice", "bob", "carol", "diego", "erin")


def run_benchmark(database: Path) -> dict[str, Any]:
    store = TeamMemoryStore(database)
    started = time.perf_counter()
    try:
        team_r1 = CacheNamespace("acme", "checkout", None, "team")
        alice_r1 = CacheNamespace("acme", "checkout", None, "private", "alice")
        billing_team = CacheNamespace("acme", "billing", None, "team")

        store.put(
            namespace=team_r1.key,
            workspace_id="checkout",
            scope="team",
            principal_id=None,
            kind="fix",
            title="Retry checkout idempotently",
            content="Reuse the idempotency key and retry only gateway timeouts.",
            query_text="checkout payment timeout duplicate charge retry",
            revision="checkout-r1",
            dependencies={"checkout/payment.py": "digest-r1"},
            evidence=["checkout/payment.py:80-112", "tests/test_checkout.py"],
            confidence=0.95,
            validated_by=["ci", "review"],
        )
        store.put(
            namespace=alice_r1.key,
            workspace_id="checkout",
            scope="private",
            principal_id="alice",
            kind="experience",
            title="Alice's unfinished experiment",
            content="A private branch tried a different retry interval.",
            query_text="checkout retry interval experiment",
            revision="checkout-r1",
            dependencies={},
        )
        store.put(
            namespace=billing_team.key,
            workspace_id="billing",
            scope="team",
            principal_id=None,
            kind="procedure",
            title="Reconcile invoices",
            content="Run the invoice reconciliation job after importing settlements.",
            query_text="invoice settlement reconciliation",
            revision="billing-r1",
            dependencies={"billing/reconcile.py": "billing-digest"},
            confidence=0.9,
            validated_by=["ci"],
        )

        exact_request = {
            "messages": [{"role": "user", "content": "How do checkout retries work?"}],
            "temperature": 0,
        }
        for developer in DEVELOPERS:
            namespace = CacheNamespace(
                "acme", "checkout", "checkout-r1", "private", developer
            )
            store.put_exact(
                namespace=namespace.key,
                workspace_id="checkout",
                revision="checkout-r1",
                model="company-coder",
                request=exact_request,
                response={"answer": "Retry gateway timeouts with the same idempotency key."},
                prompt_tokens=2_400,
                completion_tokens=120,
                cost_usd=0.012,
            )

        scenarios = []
        bob_shared = store.search(
            namespace=team_r1.key,
            workspace_id="checkout",
            query="duplicate charge after payment timeout",
            revision="checkout-r1",
            dependency_digests={"checkout/payment.py": "digest-r1"},
            principal_id="bob",
        )
        scenarios.append(
            _scenario(
                "adjacent_issue_shared_memory",
                passed=bool(bob_shared.records),
                details={"retrieved": len(bob_shared.records)},
            )
        )

        bob_private = store.search(
            namespace=alice_r1.key,
            workspace_id="checkout",
            query="retry interval experiment",
            revision="checkout-r1",
            principal_id="bob",
        )
        scenarios.append(
            _scenario(
                "private_memory_isolation",
                passed=not bob_private.records,
                details={"visible_private_records": len(bob_private.records)},
            )
        )

        cross_workspace = store.search(
            namespace=billing_team.key,
            workspace_id="billing",
            query="checkout duplicate charge timeout",
            revision="billing-r1",
            dependency_digests={"billing/reconcile.py": "billing-digest"},
            principal_id="carol",
        )
        scenarios.append(
            _scenario(
                "workspace_isolation",
                passed=not cross_workspace.records,
                details={"cross_workspace_records": len(cross_workspace.records)},
            )
        )

        stale = store.search(
            namespace=team_r1.key,
            workspace_id="checkout",
            query="checkout payment timeout",
            revision="checkout-r2",
            dependency_digests={"checkout/payment.py": "digest-r2"},
            principal_id="diego",
        )
        scenarios.append(
            _scenario(
                "revision_dependency_invalidation",
                passed=not stale.records and stale.stale_rejected == 1,
                details={"stale_rejected": stale.stale_rejected},
            )
        )

        exact_hits = 0
        for developer in DEVELOPERS:
            namespace = CacheNamespace(
                "acme", "checkout", "checkout-r1", "private", developer
            )
            if store.get_exact(
                namespace=namespace.key,
                workspace_id="checkout",
                revision="checkout-r1",
                model="company-coder",
                request=exact_request,
            ):
                exact_hits += 1
        scenarios.append(
            _scenario(
                "five_developer_exact_reuse",
                passed=exact_hits == len(DEVELOPERS),
                details={"developers": len(DEVELOPERS), "cache_hits": exact_hits},
            )
        )

        metrics = store.metrics()
        totals = metrics["totals"]
        elapsed = time.perf_counter() - started
        return {
            "schema": "machboost.team-memory-benchmark.v1",
            "developers": list(DEVELOPERS),
            "scenario_count": len(scenarios),
            "passed": all(item["passed"] for item in scenarios),
            "scenarios": scenarios,
            "savings": {
                "avoided_prompt_tokens": totals.get("avoided_prompt_tokens", 0),
                "avoided_completion_tokens": totals.get("avoided_completion_tokens", 0),
                "avoided_cost_usd": totals.get("avoided_cost_microusd", 0) / 1_000_000.0,
                "exact_cache_hits": totals.get("exact_cache_hits", 0),
            },
            "cache_metrics": metrics,
            "benchmark_wall_seconds": elapsed,
            "note": (
                "This deterministic systems benchmark validates reuse and isolation. "
                "It does not claim model decode speedup."
            ),
        }
    finally:
        store.close()


def _scenario(name: str, *, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark MachBoost team memory behavior.")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.database:
        result = run_benchmark(args.database)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_benchmark(Path(temporary) / "team.sqlite3")
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
