import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

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


class FakeCache:
    def __init__(self):
        self.tokens = []
        self.trims = []

    @property
    def state(self):
        return tuple(self.tokens)

    def trim(self, n):
        self.trims.append(n)
        if n > 0:
            del self.tokens[-n:]


class CloneableFakeCache:
    def __init__(self, tokens=None):
        self.tokens = list(tokens or [])

    @property
    def state(self):
        return list(self.tokens)

    @property
    def meta_state(self):
        return None

    @classmethod
    def from_state(cls, state, meta_state):
        return cls(state)


class CachedTinyMLXModel(TinyMLXModel):
    layers = [object()]

    def __init__(self, target_tokens, prompt_len):
        super().__init__(target_tokens, prompt_len)
        self.inputs = []

    def __call__(self, input_ids, cache=None):
        tokens = list(input_ids[0])
        self.inputs.append(tuple(tokens))
        offset = len(cache[0].tokens) if cache is not None else 0
        if cache is not None:
            cache[0].tokens.extend(tokens)

        rows = []
        for pos in range(len(tokens)):
            absolute_pos = offset + pos
            target_offset = absolute_pos - self.prompt_len + 1
            token = 0
            if 0 <= target_offset < len(self.target_tokens):
                token = self.target_tokens[target_offset]
            row = [0.0] * self.vocab_size
            row[token] = 10.0
            rows.append(row)
        return [rows]


def cache_service(target, prompt_len):
    return MLXCausalLMService(
        CachedTinyMLXModel(target, prompt_len=prompt_len),
        mx_module=FakeMX,
        cache_factory=lambda model: [FakeCache()],
        cache_trimmer=lambda cache, n: cache[0].trim(n),
        cache_can_trim=lambda cache: True,
    )


def cloneable_cache_service(target, prompt_len):
    return MLXCausalLMService(
        CachedTinyMLXModel(target, prompt_len=prompt_len),
        mx_module=FakeMX,
        cache_factory=lambda model: [CloneableFakeCache()],
        cache_can_trim=lambda cache: False,
    )


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

    def test_cached_next_token_extends_existing_cache(self):
        prompt = (100, 101, 102)
        service = cache_service((1, 2, 3), prompt_len=len(prompt))

        self.assertEqual(service.next_token(prompt), 1)
        self.assertEqual(service.next_token(prompt + (1,)), 2)
        self.assertEqual(service.next_token(prompt + (1, 2)), 3)

        self.assertEqual(service.forward_calls, 3)
        self.assertEqual(service.model.inputs, [prompt, (1,), (2,)])

    def test_generate_tokens_streams_with_cache(self):
        prompt = (100, 101, 102)
        service = cache_service((1, 2, 3, 4), prompt_len=len(prompt))
        chunks = []

        generated = service.generate_tokens(prompt, max_tokens=4, on_tokens=chunks.append)

        self.assertEqual(generated, (1, 2, 3, 4))
        self.assertEqual(tuple(token for chunk in chunks for token in chunk), generated)
        self.assertEqual(service.model.inputs, [prompt, (1,), (2,), (3,)])

    def test_generate_tokens_uses_native_mlx_stream(self):
        observed = {}
        mlx_lm = ModuleType("mlx_lm")

        def stream_generate(model, tokenizer, prompt, *, max_tokens):
            observed.update(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=max_tokens)
            for token in (1, 2, 99):
                yield SimpleNamespace(token=token)

        mlx_lm.stream_generate = stream_generate
        model = object()
        tokenizer = object()
        service = MLXCausalLMService(model, tokenizer)
        chunks = []

        with patch.dict("sys.modules", {"mlx_lm": mlx_lm}):
            generated = service.generate_tokens(
                (100, 101, 102),
                max_tokens=4,
                stop_tokens=(99,),
                on_tokens=chunks.append,
            )

        self.assertEqual(generated, (1, 2))
        self.assertEqual(chunks, [(1,), (2,)])
        self.assertEqual(service.forward_calls, 2)
        self.assertEqual(observed["model"], model)
        self.assertEqual(observed["tokenizer"], tokenizer)
        self.assertEqual(observed["prompt"], [100, 101, 102])
        self.assertEqual(observed["max_tokens"], 4)

    def test_cached_verify_commits_accepted_candidate(self):
        prompt = (100, 101, 102)
        service = cache_service((1, 2, 3, 4, 5), prompt_len=len(prompt))

        accepted, residual = service.verify(prompt, (1, 2, 3, 4))

        self.assertEqual(accepted, 4)
        self.assertIsNone(residual)
        self.assertEqual(service.forward_calls, 2)
        self.assertEqual(service._cache[0].tokens, list(prompt + (1, 2, 3, 4)))
        self.assertEqual(service.next_token(prompt + (1, 2, 3, 4)), 5)
        self.assertEqual(service.forward_calls, 2)

    def test_cached_verify_trims_rejected_tail(self):
        prompt = (100, 101, 102)
        service = cache_service((1, 2, 3, 4), prompt_len=len(prompt))

        accepted, residual = service.verify(prompt, (1, 99, 100))

        self.assertEqual(accepted, 1)
        self.assertEqual(residual, 2)
        self.assertEqual(service._cache[0].tokens, list(prompt + (1,)))
        self.assertEqual(service._cache[0].trims, [2])
        self.assertEqual(service.next_token(prompt + (1,)), 2)
        self.assertEqual(service.forward_calls, 2)

    def test_cached_verify_clones_non_trimmable_cache_before_rejection(self):
        prompt = (100, 101, 102)
        service = cloneable_cache_service((1, 2, 3, 4), prompt_len=len(prompt))

        accepted, residual = service.verify(prompt, (1, 99, 100))

        self.assertEqual(accepted, 1)
        self.assertEqual(residual, 2)
        self.assertEqual(service._cache[0].tokens, list(prompt + (1,)))
        self.assertEqual(service.next_token(prompt + (1,)), 2)
        self.assertEqual(service.forward_calls, 3)


if __name__ == "__main__":
    unittest.main()
