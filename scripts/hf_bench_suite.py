#!/usr/bin/env python3
"""Repeatable benchmark suite for the Hugging Face speculation spike."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "machboost.hf_bench_suite.v1"


@dataclass
class Fixture:
    name: str
    workflow: str
    description: str
    expectation: str
    prompt_path: str
    context_path: str
    source_mode: str


DEFAULT_FIXTURES = ["prompt_visible_readme", "prompt_visible_code", "controlled_context", "hidden_readme"]
USE_CASE_FIXTURES = [
    "prompt_visible_readme",
    "prompt_visible_code",
    "rag_answer",
    "log_template",
    "json_config",
    "test_boilerplate",
    "repo_chat_quote",
    "controlled_context",
]
NEGATIVE_FIXTURES = ["hidden_readme", "creative_open", "short_answer"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_file(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def excerpt_after(text: str, marker: str, chars: int = 420) -> str:
    pos = text.find(marker)
    if pos < 0:
        return text[:chars]
    return text[pos : pos + chars]


def expand_selected(raw_names: list[str], fixtures: dict[str, Fixture]) -> list[str]:
    expanded: list[str] = []
    aliases = {
        "default": DEFAULT_FIXTURES,
        "use_cases": USE_CASE_FIXTURES,
        "negative_controls": NEGATIVE_FIXTURES,
        "all": list(fixtures.keys()),
    }
    for name in raw_names:
        names = aliases.get(name, [name])
        for item in names:
            if item in fixtures and item not in expanded:
                expanded.append(item)
    return expanded


def build_fixtures(root: Path, selected: list[str]) -> list[Fixture]:
    project = repo_root()
    fixtures: dict[str, Fixture] = {}

    controlled_prompt = "Write a Python helper that formats benchmark results as a short table:"
    controlled_context = (
        controlled_prompt
        + " the helper should take a list of benchmark results and return a string with the results formatted as a table. "
        + "The helper should support columns for fixture, baseline tokens per second, boosted tokens per second, exact match, "
        + "and wall-clock speedup."
    )
    fixtures["controlled_context"] = Fixture(
        name="controlled_context",
        workflow="controlled",
        description="Best-case local context where the continuation is directly present.",
        expectation="positive",
        prompt_path=write_file(root / "controlled_context" / "prompt.txt", controlled_prompt),
        context_path=write_file(root / "controlled_context" / "context.txt", controlled_context),
        source_mode="context",
    )

    readme_path = project / "README.md"
    readme = readme_path.read_text(encoding="utf-8", errors="ignore") if readme_path.exists() else ""
    readme_excerpt = "# machboost\n\n`machboost` is a Mac-first CLI for running heavy local workloads under safe performance profiles."
    if readme:
        prompt_visible = (
            "You are editing this local README. The next excerpt is copied from the document below. "
            "Continue the excerpt exactly from the document. Do not add commentary.\n\n"
            "<document>\n"
            + readme
            + "\n</document>\n\n<excerpt>\n"
            + readme_excerpt
        )
        fixtures["prompt_visible_readme"] = Fixture(
            name="prompt_visible_readme",
            workflow="docs_continuation",
            description="README continuation where the model and drafter both see the reference text.",
            expectation="positive",
            prompt_path=write_file(root / "prompt_visible_readme" / "prompt.txt", prompt_visible),
            context_path=write_file(root / "prompt_visible_readme" / "context.md", readme),
            source_mode="prompt-context",
        )
        hidden_prompt = (
            "Continue the following project README excerpt exactly, using local context if it appears there. "
            "Do not add commentary.\n\n"
            + readme_excerpt
        )
        fixtures["hidden_readme"] = Fixture(
            name="hidden_readme",
            workflow="hidden_context",
            description="Negative control: drafter sees README context, but the model only sees the short prompt.",
            expectation="negative",
            prompt_path=write_file(root / "hidden_readme" / "prompt.txt", hidden_prompt),
            context_path=write_file(root / "hidden_readme" / "context.md", readme),
            source_mode="context",
        )

    script_path = project / "scripts" / "hf_corpus_speculate.py"
    code = script_path.read_text(encoding="utf-8", errors="ignore") if script_path.exists() else ""
    if code:
        marker = "def parse_args"
        code_slice = excerpt_after(code, marker, chars=1400)
        prefix = code_slice[:260]
        code_prompt = (
            "You are editing this Python file. Continue the excerpt exactly from the file content below. "
            "Do not explain.\n\n"
            "<file>\n"
            + code_slice
            + "\n</file>\n\n<excerpt>\n"
            + prefix
        )
        fixtures["prompt_visible_code"] = Fixture(
            name="prompt_visible_code",
            workflow="code_completion",
            description="Code continuation where the target span is visible in the prompt.",
            expectation="positive",
            prompt_path=write_file(root / "prompt_visible_code" / "prompt.txt", code_prompt),
            context_path=write_file(root / "prompt_visible_code" / "context.py", code_slice),
            source_mode="prompt-context",
        )

    rag_context = """# Retrieval chunk: onboarding policy
The local assistant must answer with the policy text exactly when asked about machine benchmark records.
Policy: Benchmark records must include model name, fixture name, repeat count, exact-match rate, total tokens per second, decode tokens per second, selected draft length, and forward reduction.

# Retrieval chunk: output rule
When the user asks what a benchmark record contains, answer by copying the Policy sentence.
"""
    rag_prompt = (
        "Use the retrieved chunks below to answer the question. Copy the answer sentence exactly.\n\n"
        "<retrieved>\n"
        + rag_context
        + "</retrieved>\n\nQuestion: What must benchmark records include?\nAnswer:"
    )
    fixtures["rag_answer"] = Fixture(
        name="rag_answer",
        workflow="rag_answer",
        description="RAG-style answer where the model should copy a retrieved policy sentence.",
        expectation="positive",
        prompt_path=write_file(root / "rag_answer" / "prompt.txt", rag_prompt),
        context_path=write_file(root / "rag_answer" / "context.md", rag_context),
        source_mode="prompt-context",
    )

    log_context = """2026-06-30T14:00:00Z level=INFO worker=ingest shard=01 step=read status=ok duration_ms=18
2026-06-30T14:00:01Z level=INFO worker=ingest shard=01 step=parse status=ok duration_ms=22
2026-06-30T14:00:02Z level=INFO worker=ingest shard=01 step=embed status=ok duration_ms=41
2026-06-30T14:00:03Z level=INFO worker=ingest shard=01 step=store status=ok duration_ms=27
2026-06-30T14:00:04Z level=INFO worker=ingest shard=01 step=commit status=ok duration_ms=12
"""
    log_prompt = (
        "Continue the operational log exactly from the known run below.\n\n"
        "<log>\n"
        + log_context
        + "</log>\n\nNext line:\n2026-06-30T14:00:02Z level=INFO worker=ingest"
    )
    fixtures["log_template"] = Fixture(
        name="log_template",
        workflow="log_generation",
        description="Structured log continuation with repeated tokens and fields.",
        expectation="positive",
        prompt_path=write_file(root / "log_template" / "prompt.txt", log_prompt),
        context_path=write_file(root / "log_template" / "context.log", log_context),
        source_mode="prompt-context",
    )

    json_context = """{
  "profile": "sustained",
  "workload": "llm",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "metrics": {
    "exact_match_rate": 1.0,
    "baseline_decode_tokens_per_second": 33.40,
    "boosted_decode_tokens_per_second": 60.90,
    "selected_draft_tokens": 8,
    "forward_reduction_percent": 56.25
  },
  "safety": {
    "mutates_global_state": false,
    "uploads_telemetry": false
  }
}
"""
    json_prompt = (
        "Continue this JSON configuration exactly from the document below.\n\n"
        "<json>\n"
        + json_context
        + "</json>\n\n{\n  \"profile\": \"sustained\",\n  \"workload\": \"llm\",\n  \"model\":"
    )
    fixtures["json_config"] = Fixture(
        name="json_config",
        workflow="structured_config",
        description="JSON/config continuation with repeated keys and predictable structure.",
        expectation="positive",
        prompt_path=write_file(root / "json_config" / "prompt.txt", json_prompt),
        context_path=write_file(root / "json_config" / "context.json", json_context),
        source_mode="prompt-context",
    )

    tests_context = """def test_profile_sustained_keeps_awake():
    profile = resolve_profile("sustained", "generic")
    assert profile.keep_awake is True
    assert profile.env["MACHBOOST_PROFILE"] == "sustained"

def test_profile_quiet_avoids_keep_awake():
    profile = resolve_profile("quiet", "generic")
    assert profile.keep_awake is False
    assert profile.env["MACHBOOST_PROFILE"] == "quiet"

def test_profile_balanced_uses_safe_defaults():
    profile = resolve_profile("balanced", "generic")
    assert profile.keep_awake is True
    assert profile.env["MACHBOOST_PROFILE"] == "balanced"
"""
    tests_prompt = (
        "You are editing this Python test file. Continue the excerpt exactly from the file below.\n\n"
        "<file>\n"
        + tests_context
        + "</file>\n\n<excerpt>\n"
        + tests_context[:260]
    )
    fixtures["test_boilerplate"] = Fixture(
        name="test_boilerplate",
        workflow="test_generation",
        description="Unit-test boilerplate continuation with repeated assertion patterns.",
        expectation="positive",
        prompt_path=write_file(root / "test_boilerplate" / "prompt.txt", tests_prompt),
        context_path=write_file(root / "test_boilerplate" / "context.py", tests_context),
        source_mode="prompt-context",
    )

    repo_answer_context = """Command examples:
- machboost doctor
- machboost doctor --json
- machboost run --profile sustained --workload generic -- echo ok
- machboost bench command -- sleep 1
- python3 scripts/hf_bench_suite.py --model local-or-hf-model --repeat 5 --local-files-only
"""
    repo_answer_prompt = (
        "Answer by copying the benchmark-suite command from the command examples. Do not explain.\n\n"
        "<examples>\n"
        + repo_answer_context
        + "</examples>\n\nBenchmark-suite command:"
    )
    fixtures["repo_chat_quote"] = Fixture(
        name="repo_chat_quote",
        workflow="repo_chat",
        description="Repo-chat answer that quotes an exact command from local documentation.",
        expectation="positive",
        prompt_path=write_file(root / "repo_chat_quote" / "prompt.txt", repo_answer_prompt),
        context_path=write_file(root / "repo_chat_quote" / "context.md", repo_answer_context),
        source_mode="prompt-context",
    )

    creative_context = "The cache stores benchmark rows, fixture metadata, and JSON output paths."
    fixtures["creative_open"] = Fixture(
        name="creative_open",
        workflow="creative_generation",
        description="Negative control: open-ended generation should not benefit from local lookup.",
        expectation="negative",
        prompt_path=write_file(
            root / "creative_open" / "prompt.txt",
            "Write two fresh sentences about why local tools can make developers feel more capable.",
        ),
        context_path=write_file(root / "creative_open" / "context.txt", creative_context),
        source_mode="context",
    )

    fixtures["short_answer"] = Fixture(
        name="short_answer",
        workflow="short_answer",
        description="Negative control: very short answers should not amortize speculation overhead.",
        expectation="negative",
        prompt_path=write_file(root / "short_answer" / "prompt.txt", "Answer with exactly one word: yes or no?"),
        context_path=write_file(root / "short_answer" / "context.txt", "yes no maybe benchmark profile context"),
        source_mode="context",
    )

    selected = expand_selected(selected, fixtures)
    missing = [name for name in selected if name not in fixtures]
    if missing:
        raise SystemExit(f"unknown or unavailable fixture(s): {', '.join(missing)}")
    return [fixtures[name] for name in selected]


def parse_json_output(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("benchmark command did not emit JSON")
    return json.loads(stdout[start:])


def best_run(result: dict[str, Any]) -> dict[str, Any]:
    runs = result.get("runs")
    if not runs:
        return result
    exact = [run for run in runs if run.get("output_match")]
    candidates = exact if exact else runs
    return max(candidates, key=lambda run: run.get("wall_clock_speedup", 0.0))


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def pct(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def common_int(values: list[int]) -> int:
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def summarize_fixture(fixture: Fixture, records: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = [record["best_run"] for record in records]
    baselines = [record["result"]["baseline"] for record in records]
    specs = [run["speculative"] for run in chosen]
    exact_count = sum(1 for run in chosen if run.get("output_match"))
    decode_speedups = [
        spec.get("decode_tokens_per_second", 0.0) / base.get("decode_tokens_per_second", 1.0)
        for spec, base in zip(specs, baselines)
        if base.get("decode_tokens_per_second", 0.0) > 0
    ]
    return {
        "fixture": fixture.name,
        "workflow": fixture.workflow,
        "expectation": fixture.expectation,
        "description": fixture.description,
        "repeats": len(records),
        "exact_match_rate": pct(exact_count, len(records)),
        "median_baseline_tokens_per_second": median([base["tokens_per_second"] for base in baselines]),
        "median_boosted_tokens_per_second": median([spec["tokens_per_second"] for spec in specs]),
        "median_baseline_decode_tokens_per_second": median(
            [base.get("decode_tokens_per_second", base["tokens_per_second"]) for base in baselines]
        ),
        "median_boosted_decode_tokens_per_second": median(
            [spec.get("decode_tokens_per_second", spec["tokens_per_second"]) for spec in specs]
        ),
        "median_wall_clock_speedup": median([run.get("wall_clock_speedup", 0.0) for run in chosen]),
        "median_decode_speedup": median(decode_speedups),
        "median_forward_reduction_percent": median([run.get("forward_reduction_percent", 0.0) for run in chosen]),
        "median_accepted_draft_tokens": median([spec.get("accepted_draft_tokens", 0) for spec in specs]),
        "median_baseline_model_forwards": median([base.get("model_forwards", 0) for base in baselines]),
        "median_boosted_model_forwards": median([spec.get("model_forwards", 0) for spec in specs]),
        "selected_draft_tokens": common_int([int(run.get("max_draft_tokens", 0)) for run in chosen]),
    }


def command_for_fixture(args: argparse.Namespace, fixture: Fixture) -> list[str]:
    cmd = [
        args.python,
        str(Path(args.script).resolve()),
        "--model",
        args.model,
        "--prompt",
        fixture.prompt_path,
        "--context",
        fixture.context_path,
        "--source-mode",
        fixture.source_mode,
        "--verify-mode",
        args.verify_mode,
        "--anchor-tokens",
        str(args.anchor_tokens),
        "--min-verify-margin",
        str(args.min_verify_margin),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--ngram",
        str(args.ngram),
        "--candidate-limit",
        str(args.candidate_limit),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--max-context-chars",
        str(args.max_context_chars),
        "--auto-draft",
        "--draft-sweep",
        args.draft_sweep,
        "--draft-policy",
        args.draft_policy,
        "--initial-draft-tokens",
        str(args.initial_draft_tokens),
        "--min-draft-tokens",
        str(args.min_draft_tokens),
        "--draft-step",
        str(args.draft_step),
        "--json",
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    return cmd


def run_suite(args: argparse.Namespace, fixture_root: Path) -> dict[str, Any]:
    selected = [item.strip() for item in args.fixtures.split(",") if item.strip()]
    fixtures = build_fixtures(fixture_root, selected)
    if args.list_fixtures:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixtures": [asdict(fixture) for fixture in fixtures],
        }

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.time()

    for fixture in fixtures:
        fixture_records: list[dict[str, Any]] = []
        cmd = command_for_fixture(args, fixture)
        if args.dry_run:
            fixture_records.append(
                {
                    "fixture": fixture.name,
                    "repeat_index": 1,
                    "command": " ".join(shlex.quote(part) for part in cmd),
                }
            )
        else:
            for repeat_index in range(1, args.repeat + 1):
                print(f"[{fixture.name}] repeat {repeat_index}/{args.repeat}", file=sys.stderr, flush=True)
                proc = subprocess.run(cmd, cwd=repo_root(), text=True, capture_output=True)
                if proc.returncode != 0:
                    raise SystemExit(
                        f"benchmark failed for {fixture.name} repeat {repeat_index}\n"
                        f"command: {' '.join(shlex.quote(part) for part in cmd)}\n"
                        f"stderr:\n{proc.stderr[-4000:]}"
                    )
                result = parse_json_output(proc.stdout)
                chosen = best_run(result)
                record = {
                    "fixture": fixture.name,
                    "repeat_index": repeat_index,
                    "command": cmd,
                    "result": result,
                    "best_run": chosen,
                }
                fixture_records.append(record)
                records.append(record)
        if not args.dry_run:
            summaries.append(summarize_fixture(fixture, fixture_records))
        else:
            records.extend(fixture_records)

    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
        "verify_mode": args.verify_mode,
        "anchor_tokens": args.anchor_tokens,
        "draft_policy": args.draft_policy,
        "initial_draft_tokens": args.initial_draft_tokens,
        "min_draft_tokens": args.min_draft_tokens,
        "draft_step": args.draft_step,
        "draft_sweep": args.draft_sweep,
        "fixtures": [asdict(fixture) for fixture in fixtures],
        "summaries": summaries,
        "runs": records,
        "elapsed_seconds": time.time() - started,
        "dry_run": args.dry_run,
    }


def format_table(result: dict[str, Any]) -> str:
    if result.get("dry_run"):
        lines = ["Dry run commands:"]
        for record in result["runs"]:
            lines.append(f"- {record['fixture']}: {record['command']}")
        return "\n".join(lines)
    rows = result.get("summaries", [])
    if not rows:
        return "No benchmark rows."
    headers = [
        "fixture",
        "workflow",
        "expect",
        "match",
        "total tok/s",
        "decode tok/s",
        "speedup",
        "decode speedup",
        "forwards",
        "draft",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["fixture"],
                row["workflow"],
                row["expectation"],
                f"{row['exact_match_rate'] * 100:.0f}%",
                f"{row['median_baseline_tokens_per_second']:.2f}->{row['median_boosted_tokens_per_second']:.2f}",
                (
                    f"{row['median_baseline_decode_tokens_per_second']:.2f}"
                    f"->{row['median_boosted_decode_tokens_per_second']:.2f}"
                ),
                f"{row['median_wall_clock_speedup']:.2f}x",
                f"{row['median_decode_speedup']:.2f}x",
                f"{row['median_baseline_model_forwards']:.0f}->{row['median_boosted_model_forwards']:.0f}",
                str(row["selected_draft_tokens"]),
            ]
        )
    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        " | ".join("-" * width for width in widths),
    ]
    for row in table_rows:
        lines.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def run_self_test() -> dict[str, Any]:
    fake = [
        {
            "result": {
                "baseline": {
                    "tokens_per_second": 10.0,
                    "decode_tokens_per_second": 12.0,
                    "model_forwards": 4,
                }
            },
            "best_run": {
                "output_match": True,
                "wall_clock_speedup": 2.0,
                "forward_reduction_percent": 50.0,
                "max_draft_tokens": 4,
                "speculative": {
                    "tokens_per_second": 20.0,
                    "decode_tokens_per_second": 24.0,
                    "accepted_draft_tokens": 8,
                    "model_forwards": 2,
                },
            },
        },
        {
            "result": {
                "baseline": {
                    "tokens_per_second": 12.0,
                    "decode_tokens_per_second": 13.0,
                    "model_forwards": 4,
                }
            },
            "best_run": {
                "output_match": True,
                "wall_clock_speedup": 1.5,
                "forward_reduction_percent": 25.0,
                "max_draft_tokens": 4,
                "speculative": {
                    "tokens_per_second": 18.0,
                    "decode_tokens_per_second": 20.0,
                    "accepted_draft_tokens": 6,
                    "model_forwards": 3,
                },
            },
        },
    ]
    fixture = Fixture(
        name="fake",
        workflow="self_test",
        description="fake fixture",
        expectation="positive",
        prompt_path="",
        context_path="",
        source_mode="context",
    )
    summary = summarize_fixture(fixture, fake)
    ok = (
        summary["exact_match_rate"] == 1.0
        and summary["selected_draft_tokens"] == 4
        and round(summary["median_wall_clock_speedup"], 2) == 1.75
    )
    return {"schema_version": f"{SCHEMA_VERSION}.self_test", "ok": ok, "summary": summary}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MachBoost HF speculation benchmark suite.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fixtures", default="prompt_visible_readme,prompt_visible_code,controlled_context,hidden_readme")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--draft-sweep", default="2,4,6,8,10")
    parser.add_argument("--draft-policy", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--initial-draft-tokens", type=int, default=2)
    parser.add_argument("--min-draft-tokens", type=int, default=1)
    parser.add_argument("--draft-step", type=int, default=2)
    parser.add_argument("--verify-mode", choices=["block", "hybrid", "sequential"], default="hybrid")
    parser.add_argument("--anchor-tokens", type=int, default=1)
    parser.add_argument("--min-verify-margin", type=float, default=0.0)
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--script", default=str(repo_root() / "scripts" / "hf_corpus_speculate.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--suite-dir", help="Directory for generated fixture files. Defaults to a temporary directory.")
    parser.add_argument("--keep-fixtures", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list-fixtures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", help="Optional path for JSON results.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = run_self_test()
    elif args.suite_dir:
        result = run_suite(args, Path(args.suite_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="machboost-hf-suite-") as tmp:
            result = run_suite(args, Path(tmp))
            if args.keep_fixtures:
                print("warning: --keep-fixtures requires --suite-dir", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json or args.self_test:
        print(json.dumps(result, indent=2))
    else:
        print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
