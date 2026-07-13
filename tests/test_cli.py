import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
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

    def test_human_self_test_does_not_present_synthetic_metric_as_speedup(self):
        output = io.StringIO()

        with redirect_stdout(output):
            cli.print_human_self_test(self_test_data())

        rendered = output.getvalue()
        self.assertIn("synthetic target-call reduction:", rendered)
        self.assertIn("not a wall-clock benchmark", rendered)
        self.assertNotIn("estimated speedup:", rendered)

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

    def test_model_list_data_detects_cached_native_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            write_cached_model(
                cache_dir,
                "Qwen/Qwen2.5-3B-Instruct",
                {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
            )
            write_cached_model(
                cache_dir,
                "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
            )
            write_cached_model(
                cache_dir,
                "thenlper/gte-base",
                {"architectures": ["BertModel"], "model_type": "bert"},
            )

            data = cli.model_list_data(cache_dirs=[str(cache_dir)])
            names = {model["name"] for model in data["models"]}

        self.assertEqual(data["schema_version"], "machboost.model_list.v1")
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", names)
        self.assertIn("mlx-community/Qwen3.5-0.8B-MLX-4bit", names)
        self.assertNotIn("thenlper/gte-base", names)
        self.assertEqual(data["hidden_unsupported_count"], 1)
        self.assertIn("qwen2.5:3b", {alias["name"] for alias in data["aliases"]})

    def test_main_list_json_can_show_unsupported_cached_models(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            write_cached_model(
                cache_dir,
                "thenlper/gte-base",
                {"architectures": ["BertModel"], "model_type": "bert"},
            )

            with redirect_stdout(output):
                code = main(["list", "--cache-dir", str(cache_dir), "--all", "--json"])

        data = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(data["models"][0]["name"], "thenlper/gte-base")
        self.assertFalse(data["models"][0]["runnable"])

    def test_select_native_backend_prefers_mlx_for_mlx_models(self):
        self.assertEqual(cli.select_native_backend("mlx-community/Qwen3.5-0.8B-MLX-4bit", "auto"), "mlx")
        self.assertEqual(cli.select_native_backend("Qwen/Qwen2.5-3B-Instruct", "auto"), "hf")
        self.assertEqual(cli.select_native_backend("mlx-community/Qwen3.5-0.8B-MLX-4bit", "hf"), "hf")

    def test_short_model_alias_selects_native_mlx_backend(self):
        with patch("machboost.models.native_mlx_available", return_value=True):
            self.assertEqual(cli.select_native_backend("qwen2.5:3b", "auto"), "mlx")

    def test_render_chat_prompt_includes_system_and_history(self):
        prompt = cli.render_chat_prompt(
            "Answer with local context.",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        )

        self.assertIn("System: Answer with local context.", prompt)
        self.assertIn("User: hello", prompt)
        self.assertIn("Assistant: hi", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_native_run_loads_hf_model_and_chats(self):
        output = io.StringIO()
        errors = io.StringIO()
        prompts = iter(["hello", "/bye"])
        FakeAccelerator.reset()

        with patch.object(cli, "Accelerator", FakeAccelerator):
            code = cli.run_native_chat(
                cli.build_parser().parse_args(
                    [
                        "run",
                        "Qwen/Qwen2.5-3B-Instruct",
                        "--backend",
                        "hf",
                        "--context",
                        "README.md",
                        "--show-stats",
                    ]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
                error_stream=errors,
            )

        self.assertEqual(code, 0)
        self.assertIn("machboost run: Qwen/Qwen2.5-3B-Instruct", output.getvalue())
        self.assertIn("native response", output.getvalue())
        self.assertIn("estimated_speedup=2.50x", output.getvalue())
        self.assertEqual(FakeAccelerator.calls[-1][0], "hf")
        self.assertEqual(FakeAccelerator.calls[-1][2]["context_paths"], ["README.md"])
        self.assertEqual(FakeAccelerator.instances[-1].messages[-1][0][0]["role"], "system")
        self.assertEqual(FakeAccelerator.instances[-1].messages[-1][0][-1]["content"], "hello")

    def test_explicit_ollama_wrapper_pulls_missing_model_and_streams_response(self):
        output = io.StringIO()
        errors = io.StringIO()
        prompts = iter(["hello", "/bye"])

        with patch.object(cli, "OllamaHTTPAdapter", FakeOllamaAdapter):
            code = cli.run_ollama_chat(
                cli.build_parser().parse_args(["ollama", "run", "qwen2.5:3b"]),
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

    def test_resident_run_streams_chat_and_forwards_acceleration_options(self):
        output = io.StringIO()
        errors = io.StringIO()
        prompts = iter(["hello", "/bye"])
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    [
                        "run",
                        "mlx-community/Qwen2.5-3B-Instruct-4bit",
                        "--context",
                        "docs",
                        "--ngram",
                        "3",
                        "--show-stats",
                    ]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
                error_stream=errors,
            )

        self.assertEqual(code, 0)
        self.assertIn("resident response", output.getvalue())
        self.assertIn("backend=mlx", output.getvalue())
        self.assertEqual(client.chat_calls[0][2]["context_paths"], ["docs"])
        self.assertEqual(client.chat_calls[0][2]["ngram"], 3)
        self.assertEqual(client.chat_calls[0][3], "forever")

    def test_resident_completion_streams_raw_output(self):
        output = io.StringIO()
        errors = io.StringIO()
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_completion(
                cli.build_parser().parse_args(
                    ["complete", "mlx-community/example", "def add", "--max-tokens", "16"]
                ),
                output_stream=output,
                error_stream=errors,
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "completion\n")
        self.assertEqual(client.generate_calls[0][1], "def add")
        self.assertEqual(client.generate_calls[0][2]["num_predict"], 16)

    def test_ps_prints_resident_model_table(self):
        output = io.StringIO()
        client = FakeResidentClient()

        with patch.object(cli, "MachBoostClient", return_value=client):
            code = cli.run_ps(
                cli.build_parser().parse_args(["ps"]),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertIn("mlx-community/example", output.getvalue())
        self.assertIn("forever", output.getvalue())

    def test_chat_command_uses_native_resident_arguments(self):
        args = cli.build_parser().parse_args(["chat", "mlx-community/example", "--backend", "mlx"])

        self.assertEqual(args.command, "chat")
        self.assertEqual(args.backend, "mlx")
        self.assertEqual(args.keep_alive, "forever")


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


class FakeResidentClient:
    endpoint = "http://127.0.0.1:11435"

    def __init__(self):
        self.chat_calls = []
        self.generate_calls = []

    def is_healthy(self):
        return True

    def chat(self, model, messages, *, options, keep_alive, stream):
        self.chat_calls.append((model, messages, options, keep_alive, stream))
        return iter(
            [
                {"message": {"content": "resident "}, "done": False},
                {"message": {"content": "response"}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "eval_count": 2,
                    "machboost": {
                        "backend": "mlx",
                        "stats": {
                            "generated_tokens": 2,
                            "accepted_draft_tokens": 1,
                            "target_calls": 1,
                            "baseline_target_calls": 2,
                        },
                    },
                },
            ]
        )

    def generate(self, model, prompt, *, options, keep_alive, stream):
        self.generate_calls.append((model, prompt, options, keep_alive, stream))
        return iter(
            [
                {"response": "completion", "done": False},
                {"response": "", "done": True, "eval_count": 1, "machboost": {"backend": "mlx", "stats": {}}},
            ]
        )

    def ps(self):
        return [
            {
                "model": "mlx-community/example",
                "backend": "mlx",
                "requests": 3,
                "idle_seconds": 1.5,
                "keep_alive_seconds": -1.0,
            }
        ]


def write_cached_model(cache_dir, model_id, config):
    snapshot = cache_dir / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")


class FakeStats:
    generated_tokens = 5
    accepted_draft_tokens = 3
    target_calls = 2
    baseline_target_calls = 5
    estimated_speedup = 2.5


class FakeAccelerator:
    calls = []
    instances = []

    def __init__(self, backend, model, kwargs):
        self.backend = backend
        self.model = model
        self.kwargs = kwargs
        self.prompts = []
        self.messages = []
        self.instances.append(self)

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.instances = []

    @classmethod
    def from_huggingface(cls, model, **kwargs):
        cls.calls.append(("hf", model, kwargs))
        return cls("hf", model, kwargs)

    @classmethod
    def from_mlx(cls, model, **kwargs):
        cls.calls.append(("mlx", model, kwargs))
        return cls("mlx", model, kwargs)

    def generate_chat(self, messages, max_tokens=128, on_text=None):
        self.messages.append((messages, max_tokens))
        if on_text is not None:
            on_text("native ")
            on_text("response")
        return "native response", FakeStats()

    def generate(self, prompt, max_tokens=128):
        self.prompts.append((prompt, max_tokens))
        return "native response", FakeStats()


if __name__ == "__main__":
    unittest.main()
