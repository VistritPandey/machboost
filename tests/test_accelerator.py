import tempfile
import unittest
from pathlib import Path

from machboost import Accelerator, AcceleratorResult
from machboost.accelerator import read_context_paths, resolve_context


class ScriptedService:
    def __init__(self, prompt, completion):
        self.prompt_len = len(self.encode(prompt))
        self.completion = tuple(self.encode(completion))
        self.reset_count = 0

    def encode(self, text):
        return tuple(ord(char) for char in text)

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)

    def reset_cache(self):
        self.reset_count += 1

    def next_token(self, prefix_tokens):
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


class AcceleratorTests(unittest.TestCase):
    def test_generate_result_uses_context_drafts(self):
        prompt = "Question: ship policy?\nAnswer: "
        completion = "Use the approved rollout checklist."
        service = ScriptedService(prompt, completion)
        accelerator = Accelerator(service, context_texts=[completion], ngram=2, max_draft_tokens=8)

        result = accelerator.generate_result(prompt, max_tokens=len(completion))

        self.assertIsInstance(result, AcceleratorResult)
        self.assertEqual(result.text, completion)
        self.assertEqual(service.reset_count, 1)
        self.assertGreater(result.stats.accepted_draft_tokens, 0)
        self.assertGreater(result.stats.estimated_speedup, 1.0)

    def test_generate_returns_text_and_stats(self):
        prompt = "Complete: "
        completion = "alpha beta"
        service = ScriptedService(prompt, completion)
        accelerator = Accelerator(service, ngram=2, max_draft_tokens=8)

        text, stats = accelerator.generate(prompt, max_tokens=len(completion), context=completion)

        self.assertEqual(text, completion)
        self.assertGreater(stats.accepted_draft_tokens, 0)

    def test_resolve_context_reads_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.md"
            path.write_text("alpha\nbeta\n", encoding="utf-8")

            resolved = resolve_context(str(path), max_chars=100)

        self.assertEqual(len(resolved), 1)
        self.assertIn("# file:", resolved[0])
        self.assertIn("alpha", resolved[0])

    def test_resolve_context_keeps_literal_strings(self):
        self.assertEqual(resolve_context("literal context"), ["literal context"])

    def test_read_context_paths_skips_non_text_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keep.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "skip.bin").write_bytes(b"\x00\x01\x02")

            chunks = read_context_paths(str(root), max_chars=200)

        joined = "\n".join(chunks)
        self.assertIn("keep.py", joined)
        self.assertNotIn("skip.bin", joined)


if __name__ == "__main__":
    unittest.main()
