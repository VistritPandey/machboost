import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    torch = None
else:
    import torch

from machboost import machboost
from machboost.adapters import HuggingFaceCausalLMService


if torch is not None:
    class Output:
        def __init__(self, logits):
            self.logits = logits


    class TinyCausalModel(torch.nn.Module):
        def __init__(self, target_tokens, prompt_len):
            super().__init__()
            self.target_tokens = tuple(target_tokens)
            self.prompt_len = prompt_len
            self.vocab_size = max(self.target_tokens + (0,)) + 8
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids):
            batch, seqlen = input_ids.shape
            logits = torch.zeros(batch, seqlen, self.vocab_size, device=input_ids.device)
            for pos in range(seqlen):
                target_offset = pos - self.prompt_len + 1
                token = 0
                if 0 <= target_offset < len(self.target_tokens):
                    token = self.target_tokens[target_offset]
                logits[:, pos, token] = 10.0
            return Output(logits)


@unittest.skipIf(torch is None, "torch is not installed")
class HuggingFaceAdapterTest(unittest.TestCase):
    def test_next_token_uses_last_logits(self):
        service = HuggingFaceCausalLMService(TinyCausalModel((1, 2, 3), prompt_len=3))

        self.assertEqual(service.next_token((100, 101, 102)), 1)
        self.assertEqual(service.forward_calls, 1)

    def test_verify_accepts_candidate_in_single_forward(self):
        service = HuggingFaceCausalLMService(TinyCausalModel((1, 2, 3, 4), prompt_len=3))

        accepted, residual = service.verify((100, 101, 102), (1, 2, 3))

        self.assertEqual(accepted, 3)
        self.assertIsNone(residual)
        self.assertEqual(service.forward_calls, 1)

    def test_verify_returns_residual_on_mismatch(self):
        service = HuggingFaceCausalLMService(TinyCausalModel((1, 2, 3), prompt_len=3))

        accepted, residual = service.verify((100, 101, 102), (1, 99))

        self.assertEqual(accepted, 1)
        self.assertEqual(residual, 2)

    def test_margin_can_reject_low_confidence_candidate(self):
        service = HuggingFaceCausalLMService(
            TinyCausalModel((1, 2, 3), prompt_len=3),
            min_verify_margin=20.0,
        )

        accepted, residual = service.verify((100, 101, 102), (1, 2))

        self.assertEqual(accepted, 0)
        self.assertEqual(residual, 1)

    def test_machboost_with_hf_adapter_reduces_target_forwards(self):
        prompt = (100, 101, 102)
        target = (1, 2, 3, 4, 5, 6, 7, 8)
        service = HuggingFaceCausalLMService(TinyCausalModel(target, prompt_len=len(prompt)))
        boosted = machboost(service, corpus_tokens=prompt + target, ngram=3, max_draft_tokens=4)

        generated, stats = boosted.generate(prompt, max_tokens=len(target))

        self.assertEqual(generated, target)
        self.assertEqual(service.forward_calls, 2)
        self.assertEqual(stats.target_calls, 2)
        self.assertEqual(stats.accepted_draft_tokens, len(target))
        self.assertGreaterEqual(stats.estimated_speedup, 4.0)


if __name__ == "__main__":
    unittest.main()
