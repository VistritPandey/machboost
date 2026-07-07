import argparse
import json

from machboost.adapters import OllamaHTTPAdapter, OllamaHTTPError


def result_dict(result):
    data = result.to_dict()
    return {
        "model": data["model"],
        "response": data["response"],
        "eval_count": data["eval_count"],
        "eval_ms": round(data["eval_ms"], 2),
        "total_ms": round(data["total_ms"], 2),
        "tokens_per_second": round(data["tokens_per_second"], 2),
    }


def print_json(label, value):
    print(label)
    print(json.dumps(value, indent=2, sort_keys=True))


def run_dry(adapter, args):
    options = {"num_predict": args.tokens, "num_ctx": args.ctx}
    if args.draft_num_predict > 0:
        options = OllamaHTTPAdapter.with_draft_options(options, draft_num_predict=args.draft_num_predict)

    print("mode: dry-ollama-http-adapter")
    print("endpoint:", adapter.endpoint)
    print("model:", adapter.model)
    print_json("capabilities:", adapter.capabilities().to_dict())
    print_json("example_options:", options)


def run_once(adapter, args):
    options = {}
    if args.draft_num_predict > 0:
        options = OllamaHTTPAdapter.with_draft_options(options, draft_num_predict=args.draft_num_predict)

    result = adapter.benchmark(args.prompt, tokens=args.tokens, ctx=args.ctx, options=options)
    print("mode: live-ollama-http-adapter")
    print_json("result:", result_dict(result))


def compare_draft(adapter, args):
    baseline = adapter.benchmark(args.prompt, tokens=args.tokens, ctx=args.ctx)
    draft_options = OllamaHTTPAdapter.with_draft_options(draft_num_predict=args.draft_num_predict or 4)
    drafted = adapter.benchmark(args.prompt, tokens=args.tokens, ctx=args.ctx, options=draft_options)
    baseline_tps = baseline.tokens_per_second
    drafted_tps = drafted.tokens_per_second
    ratio = drafted_tps / baseline_tps if baseline_tps > 0 else 0.0

    print("mode: live-ollama-draft-compare")
    print_json(
        "result:",
        {
            "baseline": result_dict(baseline),
            "ollama_draft_option": result_dict(drafted),
            "tokens_per_second_ratio": round(ratio, 3),
            "note": "This compares Ollama's HTTP options, not native MachBoost verifier acceleration.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run the MachBoost Ollama HTTP adapter demo.")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--prompt", default="Write one concise sentence about local inference acceleration.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--draft-num-predict", type=int, default=4)
    parser.add_argument("--run", action="store_true", help="Call a live Ollama server.")
    parser.add_argument("--compare-draft", action="store_true", help="Compare plain Ollama vs draft_num_predict.")
    args = parser.parse_args()

    adapter = OllamaHTTPAdapter(args.model, endpoint=args.endpoint)

    try:
        if args.compare_draft:
            compare_draft(adapter, args)
        elif args.run:
            run_once(adapter, args)
        else:
            run_dry(adapter, args)
    except OllamaHTTPError as exc:
        print("ollama_error:", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
