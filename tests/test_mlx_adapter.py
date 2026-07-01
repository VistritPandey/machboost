import unittest

from machboost import machboost
from machboost.adapters import MLXCausalLMService


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMX:
    int32 = int

    @staticmethod
    def array(values, dtype=None):
        return values

    @staticmethod
    def eval(*values):
        return None

    @staticmethod
    def argmax(row):
        return Scalar(max(range(len(row)), key=lambda index: row[index]))


class TinyMLXModel:
    def __init__(self, target_tokens, prompt_len):
        self.target_tokens = tuple(target_tokens)
        self.prompt_len = prompt_len
        self.vocab_size = max(self.target_tokens + (0,)) + 8

    def __call__(self, input_ids):
        rows = []
        seqlen = len(input_ids[0])
        for pos in range(seqlen):
            target_offset = pos - self.prompt_len + 1
            token = 0
            if 0 <= target_offset < len(self.target_tokens):
                token = self.target_tokens[target_offset]
            row = [0.0] * self.vocab_size
            row[token] = 10.0
            rows.append(row)
        return [rows]


class MLXAdapterTest(unittest.TestCase):
    def test_next_token_uses_last_logits(self):
        service = MLXCausalLMService(TinyMLXModel((1, 2, 3), prompt_len=3), mx_module=FakeMX)

        self.assertEqual(service.next_token((100, 101, 102)), 1)
        self.assertEqual(service.forward_calls, 1)

    def test_verify_accepts_candidate_in_single_forward(self):
        service = MLXCausalLMService(TinyMLXModel((1, 2, 3, 4), prompt_len=3), mx_module=FakeMX)

        accepted, residual = service.verify((100, 101, 102), (1, 2, 3))

        self.assertEqual(accepted, 3)
        self.assertIsNone(residual)
        self.assertEqual(service.forward_calls, 1)

    def test_verify_returns_residual_on_mismatch(self):
        service = MLXCausalLMService(TinyMLXModel((1, 2, 3), prompt_len=3), mx_module=FakeMX)

        accepted, residual = service.verify((100, 101, 102), (1, 99))

        self.assertEqual(accepted, 1)
        self.assertEqual(residual, 2)

    def test_margin_can_reject_low_confidence_candidate(self):
        service = MLXCausalLMService(
            TinyMLXModel((1, 2, 3), prompt_len=3),
            mx_module=FakeMX,
            min_verify_margin=20.0,
        )

        accepted, residual = service.verify((100, 101, 102), (1, 2))

        self.assertEqual(accepted, 0)
        self.assertEqual(residual, 1)

    def test_machboost_with_mlx_adapter_reduces_target_forwards(self):
        prompt = (100, 101, 102)
        target = (1, 2, 3, 4, 5, 6, 7, 8)
        service = MLXCausalLMService(TinyMLXModel(target, prompt_len=len(prompt)), mx_module=FakeMX)
        boosted = machboost(service, corpus_tokens=prompt + target, ngram=3, max_draft_tokens=4)

        generated, stats = boosted.generate(prompt, max_tokens=len(target))

        self.assertEqual(generated, target)
        self.assertEqual(service.forward_calls, 2)
        self.assertEqual(stats.target_calls, 2)
        self.assertEqual(stats.accepted_draft_tokens, len(target))
        self.assertGreaterEqual(stats.estimated_speedup, 4.0)


if __name__ == "__main__":
    unittest.main()
