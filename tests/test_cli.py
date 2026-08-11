import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        self.assertIn("mlx_vlm", data["optional_packages"])
        self.assertIn("dflash_mlx", data["optional_packages"])

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

    def test_model_alias_cli_parses_options_and_calls_resident_client(self):
        client = SimpleNamespace(
            create_model=lambda name, source, **kwargs: {
                "model": {"name": name, "source": source, **kwargs}
            }
        )
        args = cli.build_parser().parse_args(
            [
                "create",
                "company-coder:latest",
                "--from",
                "qwen2.5-coder:3b",
                "--system",
                "Use tests.",
                "--option",
                "num_ctx=8192",
                "--option",
                "temperature=0.2",
            ]
        )
        output = io.StringIO()

        with patch("machboost.cli.connect_resident", return_value=client):
            code = cli.run_model_alias_action(args, output_stream=output)

        self.assertEqual(code, 0)
        self.assertIn("created company-coder:latest", output.getvalue())

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
            write_cached_model(
                cache_dir,
                "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                {"architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl"},
            )

            data = cli.model_list_data(cache_dirs=[str(cache_dir)])
            names = {model["name"] for model in data["models"]}

        self.assertEqual(data["schema_version"], "machboost.model_list.v1")
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", names)
        self.assertIn("mlx-community/Qwen3.5-0.8B-MLX-4bit", names)
        self.assertIn("mlx-community/Qwen2.5-VL-3B-Instruct-4bit", names)
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

    def test_native_run_parses_dflash_decoder_options(self):
        args = cli.build_parser().parse_args(
            [
                "run",
                "qwen3.5:9b",
                "--backend",
                "dflash",
                "--draft-model",
                "z-lab/custom-draft",
                "--draft-quant",
                "w4:gs64",
                "--verify-mode",
                "adaptive",
            ]
        )

        options = cli.native_server_options(args)
        self.assertEqual(options["backend"], "dflash")
        self.assertEqual(options["draft_model"], "z-lab/custom-draft")
        self.assertEqual(options["draft_quant"], "w4:gs64")
        self.assertEqual(options["verify_mode"], "adaptive")

    def test_muse_glimmer_run_parses_reasoning_and_context_options(self):
        args = cli.build_parser().parse_args(
            [
                "run",
                "muse-glimmer:30b-mlx",
                "--ctx",
                "131072",
                "--think",
                "high",
                "--show-thinking",
            ]
        )

        resolution = cli.resolve_model(args.model, args.backend)
        options = cli.native_server_options(args)

        self.assertEqual(resolution.backend, "ollama-mlx")
        self.assertEqual(resolution.model, "muse-glimmer:30b-mlx")
        self.assertEqual(options["num_ctx"], 131072)
        self.assertEqual(options["_think"], "high")
        self.assertEqual(options["_reasoning_strength"], "high")
        self.assertTrue(args.show_thinking)

    def test_muse_glimmer_bench_parses_no_speculation_control(self):
        args = cli.build_parser().parse_args(
            [
                "bench",
                "muse-glimmer:30b-mlx",
                "--backend",
                "ollama-mlx",
                "--draft-num-predict",
                "0",
            ]
        )

        self.assertEqual(args.backend, "ollama-mlx")
        self.assertEqual(args.draft_num_predict, 0)

    def test_model_list_includes_cached_muse_glimmer(self):
        row = {
            "name": "muse-glimmer:30b-mlx",
            "backend": "ollama-mlx",
            "cached": True,
            "cached_path": "/tmp/muse-glimmer-manifest",
            "support": "ready",
            "support_reason": "compatible with Ollama's Apple Silicon MLX engine",
        }

        with patch.object(cli, "catalog_rows", return_value=[row]):
            data = cli.model_list_data(cache_dirs=[])

        muse = next(
            model
            for model in data["models"]
            if model["name"] == "muse-glimmer:30b-mlx"
        )
        self.assertEqual(muse["backend"], "ollama-mlx")
        self.assertTrue(muse["runnable"])

    def test_missing_muse_glimmer_is_pulled_before_load(self):
        client = FakeMuseResidentClient(cached=False)
        args = cli.build_parser().parse_args(["run", "muse-glimmer:30b-mlx"])
        output = io.StringIO()

        cli.ensure_resident_model(client, args, stream=output)

        self.assertEqual(client.show_calls, [("muse-glimmer:30b-mlx", True, "ollama-mlx")])
        self.assertEqual(client.pull_calls, [("muse-glimmer:30b-mlx", True)])
        self.assertIn("pulling 21 GB", output.getvalue())
        self.assertIn("pull: success", output.getvalue())

    def test_decode_bench_resolves_bf16_target_and_forwards_suite(self):
        args = cli.build_parser().parse_args(
            [
                "bench-decode",
                "qwen3.5:4b",
                "--prompt-file",
                "benchmarks/unique_decode_prompts.jsonl",
                "--draft-quant",
                "w4:gs64",
                "--max-tokens",
                "256",
                "--runs",
                "2",
                "--no-eos",
                "--output",
                "results/local/decode",
            ]
        )
        benchmark = Mock(side_effect=SystemExit(0))
        package = types.ModuleType("dflash_mlx")
        package.__path__ = []
        benchmark_module = types.ModuleType("dflash_mlx.benchmark")
        benchmark_module.main = benchmark
        model_module = types.ModuleType("dflash_mlx.model")

        class DraftArgs:
            @classmethod
            def from_dict(cls, params):
                return params

        model_module.DFlashDraftModelArgs = DraftArgs

        with patch.dict(
            "sys.modules",
            {
                "dflash_mlx": package,
                "dflash_mlx.benchmark": benchmark_module,
                "dflash_mlx.model": model_module,
            },
        ):
            code = cli.run_decode_bench(args)

        self.assertEqual(code, 0)
        forwarded = benchmark.call_args.args[0]
        self.assertIn("mlx-community/Qwen3.5-4B-MLX-bf16", forwarded)
        self.assertIn("benchmarks/unique_decode_prompts.jsonl", forwarded)
        self.assertIn("--no-eos", forwarded)
        self.assertEqual(forwarded[forwarded.index("--limit") + 1], "3")
        self.assertEqual(forwarded[forwarded.index("--verify-mode") + 1], "adaptive")
        self.assertEqual(args.cooldown, 1)
        benchmark.assert_called_once_with(
            forwarded,
            prog="machboost bench-decode",
        )

    def test_decode_prompt_limit_rejects_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text('{"prompt":"ok"}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                cli._jsonl_row_count(str(path))

    def test_decode_validation_prompt_loader_honors_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                '{"id":"one","prompt":"First"}\n'
                '{"id":"two","prompt":"Second"}\n',
                encoding="utf-8",
            )
            args = SimpleNamespace(prompt=None, prompt_file=str(path), limit=1)

            rows = cli._decode_validation_prompts(args)

        self.assertEqual(rows, [{"id": "one", "prompt": "First"}])

    def test_decode_output_validation_reports_first_token_difference(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

            def encode(self, text):
                return [ord(character) for character in text]

        accelerator = SimpleNamespace(
            model=object(),
            tokenizer=Tokenizer(),
            generate=lambda prompt, max_tokens: (
                "abX",
                SimpleNamespace(acceptance_ratio=0.75),
            ),
            close=Mock(),
        )
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.clear_cache = Mock()
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.generate = lambda *args, **kwargs: "abc"
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = lambda **kwargs: object()

        with (
            patch.dict(
                "sys.modules",
                {
                    "mlx": mlx,
                    "mlx.core": mlx_core,
                    "mlx_lm": mlx_lm,
                    "mlx_lm.sample_utils": sample_utils,
                },
            ),
            patch(
                "machboost.adapters.dflash.DFlashAccelerator.from_pretrained",
                return_value=accelerator,
            ),
        ):
            result = cli.validate_decode_outputs(
                "acme/target",
                [{"id": "fixture", "prompt": "hello"}],
                draft_model=None,
                draft_quant=None,
                verify_mode="adaptive",
                max_tokens=3,
            )

        self.assertEqual(result["exact_matches"], 0)
        self.assertEqual(result["exact_match_rate"], 0.0)
        self.assertEqual(result["results"][0]["common_prefix_tokens"], 2)
        self.assertEqual(
            result["results"][0]["first_difference"],
            {"index": 2, "native_token": ord("c"), "accelerated_token": ord("X")},
        )
        accelerator.close.assert_called_once()

    def test_decode_benchmark_fails_when_output_validation_diverges(self):
        args = cli.build_parser().parse_args(
            [
                "bench-decode",
                "qwen3.5:4b",
                "--prompt",
                "hello",
                "--validation-tokens",
                "16",
            ]
        )
        package = types.ModuleType("dflash_mlx")
        package.__path__ = []
        benchmark_module = types.ModuleType("dflash_mlx.benchmark")
        benchmark_module.main = Mock(return_value=None)
        model_module = types.ModuleType("dflash_mlx.model")
        model_module.DFlashDraftModelArgs = type("DraftArgs", (), {})
        validation = {"rows": 1, "exact_matches": 0}

        with (
            patch.dict(
                "sys.modules",
                {
                    "dflash_mlx": package,
                    "dflash_mlx.benchmark": benchmark_module,
                    "dflash_mlx.model": model_module,
                },
            ),
            patch("machboost.cli.validate_decode_outputs", return_value=validation),
            patch("machboost.adapters.dflash._load_runtime_bundle_compat"),
        ):
            code = cli.run_decode_bench(args, error_stream=io.StringIO())

        self.assertEqual(code, 1)

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
        self.assertEqual(client.chat_calls[0][3], "5m")
        self.assertEqual(client.load_calls[0][2], "5m")
        self.assertTrue(client.load_calls[0][3])

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

    def test_serve_forwards_concurrency_controls(self):
        output = io.StringIO()
        errors = io.StringIO()
        args = cli.build_parser().parse_args(
            [
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "12345",
                "--replicas",
                "3",
                "--max-queue",
                "12",
                "--queue-timeout",
                "7.5",
            ]
        )

        with (
            patch.dict("os.environ", {"MACHBOOST_API_TOKEN": "secret"}),
            patch.object(cli, "serve_runtime") as serve_runtime,
        ):
            code = cli.run_serve(args, output_stream=output, error_stream=errors)

        self.assertEqual(code, 0)
        serve_runtime.assert_called_once_with(
            "0.0.0.0",
            12345,
            replicas=3,
            max_queue=12,
            queue_timeout=7.5,
            api_token="secret",
            require_auth=True,
            team_store=None,
        )
        self.assertIn("Serving 3 replica(s)", output.getvalue())
        self.assertIn("Bearer authentication", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_serve_refuses_lan_bind_without_api_token(self):
        output = io.StringIO()
        errors = io.StringIO()
        args = cli.build_parser().parse_args(["serve", "--host", "0.0.0.0"])

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(cli, "serve_runtime") as serve_runtime,
        ):
            code = cli.run_serve(args, output_stream=output, error_stream=errors)

        self.assertEqual(code, 2)
        serve_runtime.assert_not_called()
        self.assertIn("MACHBOOST_API_TOKEN", errors.getvalue())

    def test_serve_enables_team_store_and_trace_policy(self):
        output = io.StringIO()
        errors = io.StringIO()
        args = cli.build_parser().parse_args(
            [
                "serve",
                "--team",
                "--team-db",
                "/tmp/machboost-team.sqlite3",
                "--trace-mode",
                "redacted",
                "--trace-retention-days",
                "30",
                "--trace-max-mb",
                "512",
            ]
        )

        with (
            patch.object(cli, "TeamStore") as team_store_type,
            patch.object(cli, "serve_runtime") as serve_runtime,
        ):
            code = cli.run_serve(args, output_stream=output, error_stream=errors)

        team_store = team_store_type.return_value
        self.assertEqual(code, 0)
        team_store_type.assert_called_once_with(Path("/tmp/machboost-team.sqlite3"))
        team_store.update_settings.assert_called_once_with(
            trace_mode="redacted",
            retention_days=30,
            max_storage_bytes=512 * 1024 * 1024,
        )
        self.assertIs(serve_runtime.call_args.kwargs["team_store"], team_store)
        self.assertIn("Team gateway enabled", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_warm_preloads_model_through_resident_client(self):
        output = io.StringIO()
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_warm(
                cli.build_parser().parse_args(["warm", "qwen2.5:3b", "--keep-alive", "2h"]),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertIn("mlx-community/Qwen2.5-3B-Instruct-4bit", output.getvalue())
        self.assertIn("model_load=1.25s", output.getvalue())
        self.assertIn("compile_warmup=0.50s", output.getvalue())
        self.assertIn("wall=", output.getvalue())
        self.assertEqual(client.load_calls[0][2], "2h")
        self.assertTrue(client.load_calls[0][3])

    def test_bench_command_prints_latency_summary(self):
        output = io.StringIO()
        client = FakeResidentClient()
        artifact = {
            "config": {"runs": 1, "warmups": 1, "max_tokens": 8},
            "engines": {
                "machboost": {
                    "resolved_model": "mlx-community/example",
                    "backend": "mlx",
                    "load_wall_seconds": 0.1,
                    "model_load_seconds": 0.0,
                    "summary": {
                        "median_wall_seconds": 0.5,
                        "median_client_ttft_seconds": 0.2,
                        "median_tokens_per_second": 20.0,
                    },
                    "rows": [
                        {
                            "run": 1,
                            "wall_seconds": 0.5,
                            "client_ttft_seconds": 0.2,
                            "eval_count": 4,
                            "tokens_per_second": 20.0,
                        }
                    ],
                }
            },
            "comparison": {
                "machboost_total_speedup_vs_ollama": 1.5,
                "machboost_ttft_speedup_vs_ollama": 2.0,
                "median_output_equal": True,
            },
        }

        with (
            patch.object(cli, "connect_resident", return_value=client),
            patch.object(cli, "benchmark_chat_latency", return_value=artifact),
        ):
            code = cli.run_latency_bench(
                cli.build_parser().parse_args(
                    ["bench", "qwen2.5:3b", "--engine", "machboost"]
                ),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertIn("median wall=0.500s", output.getvalue())
        self.assertIn("output_equal=yes", output.getvalue())

    def test_context_bench_uses_one_model_and_reports_valid_speedup(self):
        output = io.StringIO()
        error = io.StringIO()
        accelerator = SimpleNamespace()
        artifact = {
            "config": {
                "runs": 2,
                "warmups": 0,
                "max_tokens": 8,
                "model": "mlx-community/example",
                "backend": "mlx",
            },
            "summary": {
                "valid": True,
                "output_match_rate": 1.0,
                "algorithm_engaged_rate": 1.0,
                "median_native_wall_seconds": 1.0,
                "median_machboost_wall_seconds": 0.5,
                "median_speedup": 2.0,
                "median_accepted_draft_tokens": 6.0,
                "median_target_call_reduction": 0.75,
            },
            "rows": [
                {
                    "run": 1,
                    "order": ["native", "machboost"],
                    "speedup": 2.0,
                    "output_match": True,
                    "accepted_draft_tokens": 6,
                    "native": {"wall_seconds": 1.0},
                    "machboost": {"wall_seconds": 0.5},
                }
            ],
        }
        args = cli.build_parser().parse_args(
            [
                "bench-context",
                "mlx-community/example",
                "--backend",
                "mlx",
                "--prompt",
                "Complete: ",
                "--context-text",
                "Complete: exact continuation",
                "--runs",
                "2",
                "--warmups",
                "0",
            ]
        )

        with (
            patch.object(cli.Accelerator, "from_mlx", return_value=accelerator) as load,
            patch.object(cli, "benchmark_context_acceleration", return_value=artifact) as benchmark,
        ):
            code = cli.run_context_bench(
                args,
                output_stream=output,
                error_stream=error,
            )

        self.assertEqual(code, 0)
        self.assertIn("VALID same-model speedup: 2.000x", output.getvalue())
        self.assertIn("loading 'mlx-community/example' once", error.getvalue())
        self.assertEqual(load.call_count, 1)
        self.assertEqual(
            load.call_args.kwargs["context"],
            ["Complete: exact continuation"],
        )
        self.assertEqual(benchmark.call_args.kwargs["runs"], 2)

    def test_context_bench_rejects_missing_context_before_model_load(self):
        error = io.StringIO()
        args = cli.build_parser().parse_args(
            [
                "bench-context",
                "qwen2.5:3b",
                "--prompt",
                "Complete: ",
            ]
        )

        with patch.object(cli.Accelerator, "from_mlx") as load:
            code = cli.run_context_bench(args, error_stream=error)

        self.assertEqual(code, 2)
        self.assertIn("provide --context", error.getvalue())
        load.assert_not_called()

    def test_chat_command_uses_native_resident_arguments(self):
        args = cli.build_parser().parse_args(["chat", "mlx-community/example", "--backend", "mlx"])

        self.assertEqual(args.command, "chat")
        self.assertEqual(args.backend, "mlx")
        self.assertEqual(args.keep_alive, "5m")

    def test_verbose_alias_enables_resident_timings(self):
        args = cli.build_parser().parse_args(["run", "qwen2.5:3b", "--verbose"])

        self.assertTrue(args.show_stats)

    def test_control_d_unloads_resident_model(self):
        output = io.StringIO()
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(["run", "qwen2.5:3b"]),
                input_func=lambda prompt: (_ for _ in ()).throw(EOFError()),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertEqual(client.stop_calls, ["qwen2.5:3b"])
        self.assertIn("unloaded 1 model instance", output.getvalue())

    def test_control_c_stops_reply_and_keeps_chat_open(self):
        output = io.StringIO()
        prompts = iter(["hello", "/bye"])
        client = InterruptingResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(["run", "qwen2.5:3b"]),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertIn("generation stopped", output.getvalue())

    def test_resident_visual_chat_forwards_images_and_cache_options(self):
        output = io.StringIO()
        errors = io.StringIO()
        prompts = iter(["What is shown?", "/bye"])
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    [
                        "run",
                        "qwen2.5-vl:3b",
                        "--image",
                        "fixture.png",
                        "--no-vision-cache",
                        "--vision-cache-size",
                        "7",
                        "--cold-vision",
                        "adaptive",
                        "--vision-max-edge",
                        "512",
                        "--show-stats",
                    ]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
                error_stream=errors,
            )

        self.assertEqual(code, 0)
        self.assertEqual(client.chat_calls[0][5], ["fixture.png"])
        self.assertTrue(client.chat_calls[0][2]["no_vision_cache"])
        self.assertEqual(client.chat_calls[0][2]["vision_cache_size"], 7)
        self.assertEqual(client.chat_calls[0][2]["cold_vision"], "adaptive")
        self.assertEqual(client.chat_calls[0][2]["vision_max_edge"], 512)
        self.assertIn("vision_cache=off", output.getvalue())
        self.assertIn("cold_vision=adaptive:512px", output.getvalue())

    def test_muse_glimmer_chat_streams_reasoning_answer_and_tool_calls(self):
        output = io.StringIO()
        prompts = iter(["What is the weather?", "/bye"])
        client = FakeMuseResidentClient(cached=True)

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    [
                        "run",
                        "muse-glimmer:30b-mlx",
                        "--think",
                        "high",
                        "--show-thinking",
                        "--ctx",
                        "32768",
                    ]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("thinking> I should use the weather tool.", rendered)
        self.assertIn("assistant> Checking now.", rendered)
        self.assertIn('"name": "get_weather"', rendered)
        options = client.chat_calls[0][2]
        self.assertEqual(options["_think"], "high")
        self.assertEqual(options["_reasoning_strength"], "high")
        self.assertEqual(options["num_ctx"], 32768)

    def test_visual_chat_can_attach_image_interactively(self):
        output = io.StringIO()
        client = FakeResidentClient()
        prompts = iter(["/image second.png", "Describe both.", "/clear-images", "/bye"])

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    ["run", "qwen2.5-vl:3b", "--image", "first.png"]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertEqual(client.chat_calls[0][5], ["first.png", "second.png"])
        self.assertIn("images cleared", output.getvalue())

    def test_resident_visual_chat_forwards_post_fusion_options(self):
        output = io.StringIO()
        prompts = iter(["Read this.", "/bye"])
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    [
                        "run",
                        "qwen3-vl:8b",
                        "--image",
                        "fixture.png",
                        "--vision-tokens",
                        "adaptive",
                        "--vision-token-ratio",
                        "0.35",
                        "--show-stats",
                    ]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertEqual(client.chat_calls[0][2]["vision_tokens"], "adaptive")
        self.assertEqual(client.chat_calls[0][2]["vision_token_ratio"], 0.35)
        self.assertIn("vision_tokens=adaptive:35%", output.getvalue())

    def test_resident_visual_chat_forwards_automatic_policy_controls(self):
        prompts = iter(["Read this.", "/bye"])
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    [
                        "run",
                        "qwen3-vl:8b",
                        "--image",
                        "fixture.png",
                        "--vision-tokens",
                        "auto",
                        "--vision-token-layer",
                        "6",
                        "--vision-token-bucket",
                        "32",
                        "--vision-calibration",
                        "vision-calibration.json",
                    ]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=io.StringIO(),
            )

        self.assertEqual(code, 0)
        options = client.chat_calls[0][2]
        self.assertEqual(options["vision_tokens"], "auto")
        self.assertEqual(options["vision_token_layer"], 6)
        self.assertEqual(options["vision_token_bucket"], 32)
        self.assertEqual(options["vision_calibration"], "vision-calibration.json")

    def test_visual_completion_forwards_image(self):
        output = io.StringIO()
        client = FakeResidentClient()

        with patch.object(cli, "connect_resident", return_value=client):
            code = cli.run_resident_completion(
                cli.build_parser().parse_args(
                    ["complete", "qwen2.5-vl:3b", "Describe this.", "--image", "fixture.png"]
                ),
                output_stream=output,
            )

        self.assertEqual(code, 0)
        self.assertEqual(client.generate_calls[0][5], ["fixture.png"])

    def test_resident_visual_chat_samples_video_frames(self):
        prompts = iter(["What changes?", "/bye"])
        client = FakeResidentClient()
        selection = SimpleNamespace(images=("frame-1.jpg", "frame-4.jpg"))

        with (
            patch.object(cli, "connect_resident", return_value=client),
            patch.object(cli, "sample_video", return_value=selection) as sample_video,
        ):
            code = cli.run_resident_chat(
                cli.build_parser().parse_args(
                    ["run", "qwen3-vl:8b", "--video", "clip.mp4"]
                ),
                input_func=lambda prompt: next(prompts),
                output_stream=io.StringIO(),
            )

        self.assertEqual(code, 0)
        sample_video.assert_called_once()
        self.assertEqual(client.chat_calls[0][5], ["frame-1.jpg", "frame-4.jpg"])
        self.assertIn(
            "chronological frames selected from a video",
            client.chat_calls[0][1][-1]["content"],
        )

    def test_visual_completion_forwards_sampled_video_frames(self):
        client = FakeResidentClient()
        selection = SimpleNamespace(images=("frame-1.jpg", "frame-3.jpg"))

        with (
            patch.object(cli, "connect_resident", return_value=client),
            patch.object(cli, "sample_video", return_value=selection),
        ):
            code = cli.run_resident_completion(
                cli.build_parser().parse_args(
                    ["complete", "qwen3-vl:8b", "Describe changes.", "--video", "clip.mp4"]
                ),
                output_stream=io.StringIO(),
            )

        self.assertEqual(code, 0)
        self.assertEqual(client.generate_calls[0][5], ["frame-1.jpg", "frame-3.jpg"])
        self.assertIn(
            "chronological frames selected from a video",
            client.generate_calls[0][1],
        )


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
        self.load_calls = []
        self.stop_calls = []

    def is_healthy(self):
        return True

    def chat(self, model, messages, *, options, keep_alive, stream, images=None):
        self.chat_calls.append((model, messages, options, keep_alive, stream, images))
        backend = "mlx-vlm" if images else "mlx"
        stats = {
            "generated_tokens": 2,
            "accepted_draft_tokens": 1,
            "target_calls": 1,
            "baseline_target_calls": 2,
        }
        if images:
            stats.update(
                image_count=len(images),
                visual_cache_hit=False,
                visual_cache_miss=not options.get("no_vision_cache", False),
            )
            if options.get("cold_vision") != "off":
                stats["cold_vision"] = {
                    "enabled": True,
                    "mode": options["cold_vision"],
                    "target_max_edge": options.get("vision_max_edge") or 512,
                }
            if options.get("vision_tokens") != "off":
                stats["post_fusion_vision"] = {
                    "enabled": True,
                    "mode": options["vision_tokens"],
                    "actual_visual_retention_ratio": options.get(
                        "vision_token_ratio", 0.35
                    ),
                }
        return iter(
            [
                {"message": {"content": "resident "}, "done": False},
                {"message": {"content": "response"}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "eval_count": 2,
                    "machboost": {
                        "backend": backend,
                        "stats": stats,
                    },
                },
            ]
        )

    def generate(self, model, prompt, *, options, keep_alive, stream, images=None):
        self.generate_calls.append((model, prompt, options, keep_alive, stream, images))
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

    def load(self, model, *, options, keep_alive, warmup=False):
        self.load_calls.append((model, options, keep_alive, warmup))
        return {
            "status": "success",
            "load_duration_seconds": 1.25,
            "warmup_duration_seconds": 0.5 if warmup else 0.0,
            "instance": {
                "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                "backend": "mlx",
            },
        }

    def stop(self, model):
        self.stop_calls.append(model)
        return {"unloaded": 1}


class FakeMuseResidentClient(FakeResidentClient):
    def __init__(self, *, cached):
        super().__init__()
        self.cached = cached
        self.show_calls = []
        self.pull_calls = []

    def show(self, model, *, preflight, backend):
        self.show_calls.append((model, preflight, backend))
        return {
            "preflight": {
                "runtime_available": True,
                "cached": self.cached,
                "supported": True,
            }
        }

    def pull(self, model, *, stream):
        self.pull_calls.append((model, stream))
        self.cached = True
        return iter([{"status": "pulling model"}, {"status": "success", "done": True}])

    def load(self, model, *, options, keep_alive, warmup=False):
        self.load_calls.append((model, options, keep_alive, warmup))
        return {
            "status": "success",
            "load_duration_seconds": 0.25,
            "warmup_duration_seconds": 0.1 if warmup else 0.0,
            "instance": {
                "model": "muse-glimmer:30b-mlx",
                "backend": "ollama-mlx",
            },
        }

    def chat(self, model, messages, *, options, keep_alive, stream, images=None):
        self.chat_calls.append((model, messages, options, keep_alive, stream, images))
        return iter(
            [
                {
                    "message": {"content": "", "thinking": "I should use the weather tool."},
                    "done": False,
                },
                {"message": {"content": "Checking now."}, "done": False},
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Chicago"},
                                }
                            }
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {"content": ""},
                    "done": True,
                    "eval_count": 12,
                    "machboost": {
                        "backend": "ollama-mlx",
                        "stats": {"generated_tokens": 12},
                    },
                },
            ]
        )


class InterruptingResidentClient(FakeResidentClient):
    def chat(self, model, messages, *, options, keep_alive, stream, images=None):
        self.chat_calls.append((model, messages, options, keep_alive, stream, images))

        def rows():
            raise KeyboardInterrupt
            yield None

        return rows()


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
