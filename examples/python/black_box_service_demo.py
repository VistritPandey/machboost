import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from machboost import machboost


class BlackBoxStepService:
    def __init__(self, target_tokens, prompt_len):
        self.target_tokens = tuple(target_tokens)
        self.prompt_len = prompt_len
        self.calls = 0

    def next_token(self, prefix_tokens):
        self.calls += 1
        offset = len(prefix_tokens) - self.prompt_len
        if offset >= len(self.target_tokens):
            return None
        return self.target_tokens[offset]


def main():
    prompt = (100, 101, 102)
    target = (1, 2, 3, 4)
    local_context = prompt + target

    service = BlackBoxStepService(target, prompt_len=len(prompt))
    boosted = machboost(
        service,
        corpus_tokens=local_context,
        ngram=3,
        max_draft_tokens=4,
    )

    generated, stats = boosted.generate(prompt, max_tokens=len(target))

    print("generated:", generated)
    print("exact:", generated == target)
    print("baseline target calls:", stats.baseline_target_calls)
    print("boosted target calls:", stats.target_calls)
    print("estimated speedup:", f"{stats.estimated_speedup:.2f}x")
    print("note: black-box services need verifier hooks for real acceleration")


if __name__ == "__main__":
    main()
