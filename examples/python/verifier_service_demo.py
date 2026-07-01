import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from machboost import machboost


class LocalVerifierService:
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

    def verify(self, prefix_tokens, candidate_tokens):
        self.calls += 1
        offset = len(prefix_tokens) - self.prompt_len
        accepted = 0
        for expected, proposed in zip(self.target_tokens[offset:], candidate_tokens):
            if expected != proposed:
                break
            accepted += 1
        return accepted


def main():
    prompt = (100, 101, 102)
    target = (1, 2, 3, 4, 5, 6, 7, 8)
    local_context = prompt + target

    service = LocalVerifierService(target, prompt_len=len(prompt))
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
    print("accepted draft tokens:", stats.accepted_draft_tokens)
    print("estimated speedup:", f"{stats.estimated_speedup:.2f}x")


if __name__ == "__main__":
    main()
