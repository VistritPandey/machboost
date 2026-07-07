import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from machboost import __version__
from machboost import cli
from machboost.cli import doctor_data, main, self_test_data


class CLITests(unittest.TestCase):
    def test_doctor_data_has_optional_package_statuses(self):
        data = doctor_data()

        self.assertEqual(data["schema_version"], "machboost.doctor.v1")
        self.assertEqual(data["machboost_version"], __version__)
        self.assertIn("transformers", data["optional_packages"])
        self.assertIn("available", data["optional_packages"]["transformers"])

    def test_self_test_uses_verifier_path(self):
        data = self_test_data()

        self.assertTrue(data["ok"])
        self.assertTrue(data["output_match"])
        self.assertGreater(data["accepted_draft_tokens"], 0)
        self.assertGreater(data["estimated_speedup"], 1.0)

    def test_main_version_prints_version(self):
        output = io.StringIO()

        with redirect_stdout(output):
            code = main(["version"])

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue().strip(), __version__)

    def test_main_self_test_json(self):
        output = io.StringIO()

        with redirect_stdout(output):
            code = main(["self-test", "--json"])

        data = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(data["ok"])

    def test_ollama_chat_shortcut_pulls_missing_model_and_streams_response(self):
        output = io.StringIO()
        errors = io.StringIO()
        prompts = iter(["hello", "/bye"])

        with patch.object(cli, "OllamaHTTPAdapter", FakeOllamaAdapter):
            code = cli.run_ollama_chat(
                cli.build_parser().parse_args(["chat", "qwen2.5:3b"]),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
                error_stream=errors,
            )

        self.assertEqual(code, 0)
        self.assertIn("hello from ollama", output.getvalue())
        self.assertIn("pulling", errors.getvalue())
        self.assertEqual(FakeOllamaAdapter.instances[-1].messages[-1][0]["content"], "hello")

    def test_ollama_run_can_skip_pull_for_installed_model(self):
        output = io.StringIO()
        errors = io.StringIO()
        prompts = iter(["/bye"])

        with patch.object(cli, "OllamaHTTPAdapter", InstalledFakeOllamaAdapter):
            code = cli.run_ollama_chat(
                cli.build_parser().parse_args(["ollama", "run", "qwen2.5:3b", "--no-pull"]),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
                error_stream=errors,
            )

        self.assertEqual(code, 0)
        self.assertNotIn("pulling", errors.getvalue())


class FakePullStatus:
    status = "success"
    total = 0
    completed = 0


class FakeChatChunk:
    def __init__(self, content="", done=False):
        self.content = content
        self.done = done


class FakeOllamaAdapter:
    instances = []

    def __init__(self, model, endpoint=None, timeout=300.0):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.messages = []
        self.pulled = False
        self.instances.append(self)

    def has_model(self):
        return False

    def pull(self):
        self.pulled = True
        yield FakePullStatus()

    def chat(self, messages, options=None):
        self.messages.append(messages)
        yield FakeChatChunk("hello ", False)
        yield FakeChatChunk("from ollama", False)
        yield FakeChatChunk("", True)


class InstalledFakeOllamaAdapter(FakeOllamaAdapter):
    def has_model(self):
        return True


if __name__ == "__main__":
    unittest.main()
