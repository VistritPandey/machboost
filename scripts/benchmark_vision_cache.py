from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from machboost import MachBoostClient, ensure_server


CASES = (
    ("project", "Return only the project name shown in the image.", "ATLAS"),
    ("status", "Return only the status shown in the image.", "READY"),
    ("budget", "Return only the budget shown in the image.", "$42,700"),
    ("shape", "Return only the color and shape label shown in the image.", "BLUE SQUARE"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark repeated-image feature reuse.")
    parser.add_argument("--model", default="qwen2.5-vl:3b")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    image_path = args.image or Path("results/tmp/vision_cache_fixture.png")
    if args.image is None:
        create_fixture(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    client, started = ensure_server(args.endpoint, timeout=min(30.0, args.timeout))
    client.timeout = args.timeout
    client.stop(args.model)
    loaded = client.load(
        args.model,
        options={"vision_cache_size": 4},
        keep_alive="forever",
    )

    image = str(image_path.resolve())
    run_request(
        client,
        args.model,
        image,
        "Describe the image in one sentence.",
        max_tokens=args.max_tokens,
        cache_enabled=False,
    )
    prime = run_request(
        client,
        args.model,
        image,
        CASES[0][1],
        max_tokens=args.max_tokens,
        cache_enabled=True,
    )

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for case_name, prompt, expected in CASES:
            modes = ("baseline", "cached") if repeat % 2 == 0 else ("cached", "baseline")
            pair: dict[str, dict[str, Any]] = {}
            for mode in modes:
                row = run_request(
                    client,
                    args.model,
                    image,
                    prompt,
                    max_tokens=args.max_tokens,
                    cache_enabled=mode == "cached",
                )
                row.update(
                    {
                        "repeat": repeat + 1,
                        "case": case_name,
                        "mode": mode,
                        "prompt": prompt,
                        "expected": expected,
                        "expected_match": answer_matches(row["output"], expected),
                    }
                )
                pair[mode] = row
                rows.append(row)
            exact = pair["baseline"]["output"] == pair["cached"]["output"]
            pair_speedup = (
                pair["baseline"]["client_total_seconds"]
                / pair["cached"]["client_total_seconds"]
            )
            pair["baseline"]["paired_output_equal"] = exact
            pair["cached"]["paired_output_equal"] = exact
            pair["baseline"]["paired_total_speedup"] = pair_speedup
            pair["cached"]["paired_total_speedup"] = pair_speedup

    summary = summarize(rows)
    output = args.output or Path(
        f"results/vision_cache_qwen25_3b_{datetime.now().strftime('%Y%m%d')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "machboost.vision_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "resolved_model": loaded["instance"]["model"],
        "backend": loaded["instance"]["backend"],
        "image": image,
        "image_bytes": image_path.stat().st_size,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "server_started": started,
        "load_duration_seconds": loaded["load_duration_seconds"],
        "warmup_policy": "one cache-disabled request followed by one unrecorded cache-prime request",
        "machine": machine_data(),
        "packages": package_versions(),
        "prime": prime,
        "summary": summary,
        "rows": rows,
        "resident_models": client.ps(),
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"artifact: {output}")


def run_request(
    client: MachBoostClient,
    model: str,
    image: str,
    prompt: str,
    *,
    max_tokens: int,
    cache_enabled: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_text_at = None
    output: list[str] = []
    final: dict[str, Any] = {}
    events = client.generate(
        model,
        prompt,
        images=[image],
        options={
            "num_predict": max_tokens,
            "temperature": 0.0,
            "no_vision_cache": not cache_enabled,
            "vision_cache_size": 4,
        },
        keep_alive="forever",
        stream=True,
    )
    for event in events:
        text = str(event.get("response") or "")
        if text:
            if first_text_at is None:
                first_text_at = time.perf_counter()
            output.append(text)
        if event.get("done"):
            final = event
    finished = time.perf_counter()
    if not final:
        raise RuntimeError("generation stream ended without final metrics")
    stats = dict((final.get("machboost") or {}).get("stats") or {})
    return {
        "output": "".join(output).strip(),
        "client_total_seconds": finished - started,
        "client_ttft_seconds": None if first_text_at is None else first_text_at - started,
        "server_total_seconds": float(final.get("total_duration", 0)) / 1_000_000_000,
        "generated_tokens": int(final.get("eval_count") or stats.get("generated_tokens") or 0),
        "prompt_tokens": int(stats.get("prompt_tokens") or 0),
        "prompt_tokens_per_second": float(stats.get("prompt_tokens_per_second") or 0.0),
        "generation_tokens_per_second": float(stats.get("generation_tokens_per_second") or 0.0),
        "peak_memory_gb": float(stats.get("peak_memory_gb") or 0.0),
        "visual_cache_enabled": bool(stats.get("visual_cache_enabled")),
        "visual_cache_hit": bool(stats.get("visual_cache_hit")),
        "visual_cache_miss": bool(stats.get("visual_cache_miss")),
        "visual_cache_entries": int(stats.get("visual_cache_entries") or 0),
        "prompt_cache_enabled": bool(stats.get("prompt_cache_enabled")),
        "prompt_cache_prefix_tokens": int(stats.get("prompt_cache_prefix_tokens") or 0),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row for row in rows if row["mode"] == "baseline"]
    cached = [row for row in rows if row["mode"] == "cached"]
    baseline_total = statistics.median(row["client_total_seconds"] for row in baseline)
    cached_total = statistics.median(row["client_total_seconds"] for row in cached)
    baseline_ttft = statistics.median(row["client_ttft_seconds"] for row in baseline)
    cached_ttft = statistics.median(row["client_ttft_seconds"] for row in cached)
    return {
        "rows_per_mode": len(baseline),
        "baseline_median_total_seconds": baseline_total,
        "cached_median_total_seconds": cached_total,
        "median_total_speedup": baseline_total / cached_total,
        "median_paired_total_speedup": statistics.median(
            row["paired_total_speedup"] for row in cached
        ),
        "baseline_median_ttft_seconds": baseline_ttft,
        "cached_median_ttft_seconds": cached_ttft,
        "median_ttft_speedup": baseline_ttft / cached_ttft,
        "baseline_median_prompt_tps": statistics.median(
            row["prompt_tokens_per_second"] for row in baseline
        ),
        "cached_median_prompt_tps": statistics.median(
            row["prompt_tokens_per_second"] for row in cached
        ),
        "paired_output_equal_rate": sum(row["paired_output_equal"] for row in cached) / len(cached),
        "baseline_expected_match_rate": sum(row["expected_match"] for row in baseline) / len(baseline),
        "cached_expected_match_rate": sum(row["expected_match"] for row in cached) / len(cached),
        "cached_hit_rate": sum(row["visual_cache_hit"] for row in cached) / len(cached),
        "cached_prompt_prefix_hit_rate": sum(
            row["prompt_cache_prefix_tokens"] > 0 for row in cached
        )
        / len(cached),
        "cached_median_prompt_prefix_tokens": statistics.median(
            row["prompt_cache_prefix_tokens"] for row in cached
        ),
    }


def answer_matches(output: str, expected: str) -> bool:
    normalized_output = "".join(character.lower() for character in output if character.isalnum())
    normalized_expected = "".join(character.lower() for character in expected if character.isalnum())
    return normalized_expected in normalized_output


def create_fixture(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("The generated fixture requires Pillow.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1024, 768), "#f7f7f2")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=48)
    body = ImageFont.load_default(size=36)
    draw.rectangle((70, 60, 954, 708), outline="#111111", width=6)
    draw.text((120, 110), "MACHBOOST VISION TEST", font=title, fill="#111111")
    draw.text((120, 230), "Project: ATLAS", font=body, fill="#1b4d8f")
    draw.text((120, 305), "Status: READY", font=body, fill="#137333")
    draw.text((120, 380), "Budget: $42,700", font=body, fill="#8a2c0d")
    draw.text((120, 455), "Owner: Vistrit", font=body, fill="#111111")
    draw.rectangle((700, 250, 860, 410), fill="#1976d2")
    draw.text((710, 440), "BLUE SQUARE", font=body, fill="#111111")
    image.save(path)


def machine_data() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("machboost", "mlx", "mlx-lm", "mlx-vlm", "transformers", "Pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


if __name__ == "__main__":
    main()
