from machboost import Accelerator, GatePolicy
import time


class ScriptedService:
    def __init__(self, prompt, completion):
        self.prompt_len = len(self.encode(prompt))
        self.completion = tuple(self.encode(completion))

    def encode(self, text):
        return tuple(ord(char) for char in text)

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)

    def next_token(self, prefix_tokens):
        time.sleep(0.001)
        offset = max(0, len(prefix_tokens) - self.prompt_len)
        if offset >= len(self.completion):
            return None
        return self.completion[offset]

    def verify(self, prefix_tokens, candidate_tokens):
        offset = max(0, len(prefix_tokens) - self.prompt_len)
        accepted = 0
        for token in candidate_tokens:
            target_pos = offset + accepted
            if target_pos >= len(self.completion) or token != self.completion[target_pos]:
                break
            accepted += 1
        if accepted == len(candidate_tokens):
            return accepted, None
        residual_pos = offset + accepted
        residual = self.completion[residual_pos] if residual_pos < len(self.completion) else None
        return accepted, residual


prompt = "Continue the policy: "
completion = "benchmark results must include exact output match."
service = ScriptedService(prompt, completion)
boost = Accelerator(service, context_texts=[completion], ngram=2, max_draft_tokens=8)

calibration = boost.calibrate(
    prompt,
    max_tokens=len(completion),
    gate_policy=GatePolicy(min_speedup=1.05, min_acceptance_rate=0.50),
)
text, stats = boost.generate(prompt, max_tokens=len(completion))

print(calibration.summary)
print("boost enabled:", boost.boost_enabled)
print("generated:", text)
print("accepted draft tokens:", stats.accepted_draft_tokens)
