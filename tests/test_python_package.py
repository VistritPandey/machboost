import unittest

from machboost import CorpusDrafter, machboost


class ScriptedVerifierService:
    def __init__(self, target):
        self.target = tuple(target)
        self.calls = 0

    def next_token(self, prefix_tokens):
        self.calls += 1
        offset = self._offset(prefix_tokens)
        if offset >= len(self.target):
            return None
        return self.target[offset]

    def verify(self, prefix_tokens, candidate_tokens):
        self.calls += 1
        offset = self._offset(prefix_tokens)
        accepted = 0
        for expected, proposed in zip(self.target[offset:], candidate_tokens):
            if expected != proposed:
                break
            accepted += 1
        return accepted

    def _offset(self, prefix_tokens):
        return max(0, len(prefix_tokens) - 3)


class ScriptedStepService:
    def __init__(self, target):
        self.target = tuple(target)
        self.calls = 0

    def next_token(self, prefix_tokens):
        self.calls += 1
        offset = max(0, len(prefix_tokens) - 3)
        if offset >= len(self.target):
            return None
        return self.target[offset]


class PythonPackageTest(unittest.TestCase):
    def test_corpus_drafter_uses_observed_history(self):
        drafter = CorpusDrafter([1, 2, 3, 4, 5, 6], ngram=2, max_draft_tokens=4)
        drafter.reset([1, 2])
        drafter.observe([3, 4])

        self.assertEqual(drafter.propose(max_tokens=4), (5, 6))

    def test_verifier_service_accepts_drafts_with_fewer_target_calls(self):
        prompt = (100, 101, 102)
        target = (1, 2, 3, 4, 5, 6, 7, 8)
        context = prompt + target
        service = ScriptedVerifierService(target)

        boosted = machboost(service, corpus_tokens=context, ngram=3, max_draft_tokens=4)
        generated, stats = boosted.generate(prompt, max_tokens=len(target))

        self.assertEqual(generated, target)
        self.assertEqual(service.calls, stats.target_calls)
        self.assertLess(stats.target_calls, stats.baseline_target_calls)
        self.assertGreaterEqual(stats.estimated_speedup, 2.0)
        self.assertEqual(stats.accepted_draft_tokens, len(target))

    def test_black_box_service_stays_exact_without_claiming_speedup(self):
        prompt = (100, 101, 102)
        target = (1, 2, 3, 4)
        context = prompt + target
        service = ScriptedStepService(target)

        boosted = machboost(service, corpus_tokens=context, ngram=3, max_draft_tokens=4)
        generated, stats = boosted.generate(prompt, max_tokens=len(target))

        self.assertEqual(generated, target)
        self.assertEqual(service.calls, stats.target_calls)
        self.assertGreaterEqual(stats.target_calls, stats.baseline_target_calls)
        self.assertLessEqual(stats.estimated_speedup, 1.0)

    def test_candidate_limit_can_try_alternate_context_match(self):
        prompt = (100, 101, 102)
        target = (1, 2, 3, 4)
        wrong_then_right_context = prompt + (9, 9, 9, 9) + prompt + target
        service = ScriptedVerifierService(target)

        boosted = machboost(
            service,
            corpus_tokens=wrong_then_right_context,
            ngram=2,
            max_draft_tokens=4,
            candidate_limit=2,
        )
        generated, stats = boosted.generate(prompt, max_tokens=len(target))

        self.assertEqual(generated, target)
        self.assertEqual(stats.verify_calls, 2)
        self.assertEqual(stats.rejected_candidates, 1)
        self.assertEqual(stats.accepted_draft_tokens, len(target))

    def test_generation_stops_before_stop_token(self):
        prompt = (100, 101, 102)
        target = (1, 2, 99, 3, 4)
        service = ScriptedVerifierService(target)

        boosted = machboost(service, corpus_tokens=prompt + target, ngram=3, max_draft_tokens=5)
        generated, stats = boosted.generate(prompt, max_tokens=len(target), stop_tokens=[99])

        self.assertEqual(generated, (1, 2))
        self.assertEqual(stats.accepted_draft_tokens, 2)

    def test_generation_streams_committed_tokens(self):
        prompt = (100, 101, 102)
        target = (1, 2, 3, 4)
        service = ScriptedVerifierService(target)
        chunks = []

        boosted = machboost(service, corpus_tokens=prompt + target, ngram=3, max_draft_tokens=2)
        generated, _ = boosted.generate(prompt, max_tokens=len(target), on_tokens=chunks.append)

        self.assertEqual(generated, target)
        self.assertEqual(tuple(token for chunk in chunks for token in chunk), target)


if __name__ == "__main__":
    unittest.main()
