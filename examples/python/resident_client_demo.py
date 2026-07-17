import argparse
import sys

from machboost import MachBoostClient, ensure_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream chat from the resident MachBoost runtime.")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--context", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()

    client, started = ensure_server()
    assert isinstance(client, MachBoostClient)
    loaded = client.load(args.model, keep_alive="5m", warmup=True)
    print(
        f"server={'started' if started else 'reused'} "
        f"model={loaded['instance']['model']}",
        file=sys.stderr,
    )

    final = {}
    events = client.chat(
        args.model,
        [{"role": "user", "content": args.prompt}],
        context=args.context or None,
        options={"num_predict": args.max_tokens},
        keep_alive="5m",
    )
    for event in events:
        final = event
        text = (event.get("message") or {}).get("content", "")
        if text:
            print(text, end="", flush=True)
    print()

    if final.get("done"):
        seconds = float(final.get("eval_duration", 0)) / 1_000_000_000
        tokens = int(final.get("eval_count", 0))
        rate = tokens / seconds if seconds > 0 else 0.0
        print(f"generated={tokens} eval_rate={rate:.2f} tok/s", file=sys.stderr)

    if args.shutdown:
        client.shutdown()


if __name__ == "__main__":
    main()
