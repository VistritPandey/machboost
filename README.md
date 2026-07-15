# machboost

[![CI](https://github.com/VistritPandey/machboost/actions/workflows/ci.yml/badge.svg)](https://github.com/VistritPandey/machboost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MachBoost is an experimental resident local-inference server and Python package for MLX, MLX-VLM, and Hugging Face models. It provides an Ollama-like model workflow, keeps models warm between requests, streams text and visual chat, and applies backend-specific acceleration when a request contains reusable work. An opt-in Qwen3-VL path also reduces first-view visual prefill by compressing visual hidden states after the model's early fusion layers.

For text, it drafts candidate tokens from nearby prompts, retrieved chunks, repo files, policies, configs, and docs, then verifies them with the target model. For repeated-image VLM requests, it uses architecture-aware caches: Qwen2.5-VL and Qwen3.5 can reuse projected vision features, while Qwen3-VL conservatively reuses only complete visual prompt state. Qwen3.5's hybrid recurrent/attention state is restored from whole-prefix checkpoints rather than an unsafe KV-only trim. None of these paths changes model weights.

MachBoost is local-first and alpha-stage. It does not upload telemetry, mutate global runtime settings, change model weights, or claim universal speedups. Ordinary open-ended chat follows the backend's native generation path; context-backed verification is enabled only when a candidate exists.

## Install

From a local checkout:

```sh
git clone https://github.com/VistritPandey/machboost.git
cd machboost
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional backends as needed:

```sh
pip install -e ".[mlx]"
pip install -e ".[hf]"
pip install -e ".[vision]"
pip install -e ".[all]"
```

After publishing this repository on GitHub, users can install directly from GitHub:

```sh
pip install "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git"
pip install "machboost[vision] @ git+https://github.com/VistritPandey/machboost.git"
```

Update an existing install:

```sh
pip install --upgrade "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git"
```

Check the install:

```sh
machboost doctor
machboost self-test
machboost list
python -m machboost self-test --json
```

## Quick Start

Start a native local chat from the command line. Short aliases select MLX on a compatible Apple Silicon installation and Hugging Face elsewhere:

```sh
machboost list
machboost pull qwen2.5:3b
machboost run qwen2.5:3b
```

`machboost run` starts a local server automatically when needed. The server keeps the model in unified memory until `machboost stop`, `machboost shutdown`, or process exit, so subsequent commands avoid model reload latency. Preload a model before the first user request with:

```sh
machboost warm qwen2.5:3b
machboost ps
```

Use full repository IDs when a model has no short alias. If the model is not cached, the selected backend may download it through its normal Hugging Face or MLX loader. Use `--local-files-only` with Hugging Face to require an existing cache.

Stream a raw completion for an editor or code tool:

```sh
machboost complete qwen2.5-coder:3b "def fibonacci(n):" --max-tokens 128
machboost complete qwen2.5:3b --file ./prompt.txt --context ./docs --show-stats
```

The resident server also exposes Ollama-compatible and OpenAI-compatible HTTP endpoints on `http://127.0.0.1:11435`:

```sh
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

### Visual Chat

Install the vision extra and run a supported MLX-VLM model with a local image:

```sh
pip install -e ".[vision]"
machboost pull qwen2.5-vl:3b
machboost run qwen2.5-vl:3b --image ./invoice.png --show-stats
machboost run qwen3-vl:4b --image ./invoice.png --show-stats
machboost run qwen3.5:4b --image ./invoice.png --show-stats
```

The interactive session keeps the model and attached image warm. Use `/image PATH` to attach another image, `/images` to inspect attachments, and `/clear-images` to remove them. The same path is available to Python applications:

```python
from machboost import MachBoostClient, ensure_server

client, _ = ensure_server()
response = client.chat(
    "qwen2.5-vl:3b",
    [{"role": "user", "content": "Return only the invoice total."}],
    images=["./invoice.png"],
    options={"temperature": 0.0, "num_predict": 32},
    stream=False,
)
print(response["message"]["content"])
```

Image reuse is content-addressed: changing the file bytes creates a new cache identity. Set `--no-vision-cache` or `options={"no_vision_cache": True}` for an uncached control. This optimization targets repeated questions over the same image; the first image request still performs normal vision encoding and prefill. Cache capabilities are model-specific, and MachBoost disables projected-feature reuse when a model requires additional visual tensors that cannot be cached safely.

### Experimental First-View Acceleration

Qwen3-VL can opt into adaptive post-fusion visual-token compression for a new image and question:

```sh
machboost run qwen3-vl:8b \
  --image ./document.png \
  --vision-tokens adaptive \
  --vision-token-ratio 0.35 \
  --show-stats
```

The image still passes through the full-resolution vision encoder. Qwen3-VL then processes every visual token through its three early language layers and required deep-stack injections. MachBoost groups the resulting visual states spatially, preserves the highest-variance groups, merges the rest with query-weighted pooling, and sends the shorter sequence through the remaining language layers. The request bypasses visual and prompt caches, so the reported gain is independent of prior images or prompts.

This path is approximate and disabled by default. It currently supports batch-one Qwen3-VL requests only, cannot be combined with `--cold-vision`, and can change wording or answers. On a 10-image TextVQA pilot with Qwen3-VL 8B, 35% visual retention produced a 1.67x aggregate wall-time speedup and 1.70x median paired speedup. Baseline and compressed paths each matched the dataset answer on 8 of 10 questions; normalized outputs were equal on 7 of 10. A 30% follow-up retained the same task score but was slower, so 35% remains the measured profile rather than assuming that more pruning is always better.

Use the high-level `Accelerator` when you want MachBoost to load a model and build the draft corpus from strings, files, or directories. Calibrate before enabling the boosted path for a workflow:

```python
from machboost import Accelerator, GatePolicy

boost = Accelerator.from_mlx(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    context_paths=["./docs", "./src"],
    ngram=2,
    max_draft_tokens=8,
    candidate_limit=1,
)

calibration = boost.calibrate(
    [
        "Copy the exact release checklist item from the local docs:",
        "Emit the deployment JSON field from the local config:",
    ],
    max_tokens=32,
    gate_policy=GatePolicy(min_speedup=1.05, min_acceptance_rate=0.10),
)

print(calibration.summary)

if calibration.enabled:
    text, stats = boost.generate(
        "Copy the exact release checklist item from the local docs:",
        max_tokens=64,
    )
    print(text)
    print(stats.estimated_speedup)
```

For chat models, use the chat-aware API. It applies the tokenizer's chat template and stops on special end-of-turn tokens:

```python
from machboost import Accelerator

boost = Accelerator.from_huggingface(
    "Qwen/Qwen2.5-3B-Instruct",
    context_paths=["./docs", "./src"],
    local_files_only=True,
)

text, stats = boost.generate_chat(
    [{"role": "user", "content": "Summarize the local release checklist in one sentence."}],
    max_tokens=64,
)

print(text)
```

For custom runtimes, wrap a verifier-capable service:

```python
from machboost import machboost

boosted = machboost(
    some_service,
    corpus_tokens=local_context_tokens,
    ngram=4,
    max_draft_tokens=8,
)

tokens, stats = boosted.generate(prompt_tokens, max_tokens=128)
```

Real speedups require the wrapped service to expose a verifier hook such as:

```python
verify(prefix_tokens, candidate_tokens) -> accepted_count
```

A black-box service with only `next_token(prefix_tokens)` remains exact, but it cannot skip target-model work.

Applications can control a resident server directly:

```python
from machboost import MachBoostClient

client = MachBoostClient()
client.load("qwen2.5:3b", keep_alive="forever")

for chunk in client.chat(
    "qwen2.5:3b",
    [{"role": "user", "content": "Summarize the deployment policy."}],
    options={"context_paths": ["./docs"], "num_predict": 128},
    stream=True,
):
    print(chunk.get("message", {}).get("content", ""), end="", flush=True)
```

## When It Helps

MachBoost is most useful when the model is likely to continue with text already present nearby:

- repo or source-code continuation
- config and JSON generation
- policy or documentation copying
- RAG answers that quote retrieved context
- repeated logs, templates, checklists, and structured artifacts
- repeated extraction, QA, or agent turns over the same image
- short Qwen3-VL first-view requests where visual prefill dominates and approximate token merging is acceptable

It is usually neutral or slower for open-ended creative writing, one-word answers, and prompts where the next tokens are not recoverable from local context. The package exposes benchmark and calibration APIs so applications can turn the boosted path on only when it helps.

## Command Line

The Python package installs an Ollama-style resident model workflow:

```sh
machboost list
machboost list --json
machboost pull qwen2.5:3b
machboost warm qwen2.5:3b
machboost run qwen2.5:3b
machboost run qwen2.5-vl:3b --image ./image.png
machboost chat qwen2.5:3b
machboost complete qwen2.5-coder:3b "def parse_config(text):"
machboost ps
machboost show qwen2.5:3b
machboost stop qwen2.5:3b
machboost shutdown
```

`machboost list` shows cached Hugging Face and MLX models, backend readiness, and available short aliases. `machboost run MODEL` connects to the resident server, loads the model once, builds a draft corpus from any `--context` files or directories, and opens a streaming interactive chat. Inside chat, use `/bye`, `/exit`, `/quit`, EOF, or Ctrl-C to leave, and `/clear` to reset history. Leaving chat does not unload the model.

Run the server in the foreground when integrating it with another application or process manager:

```sh
machboost serve --host 127.0.0.1 --port 11435
```

By default, models remain warm indefinitely. A finite lifetime can be selected per load or run:

```sh
machboost warm qwen2.5:3b --keep-alive 1h
machboost run qwen2.5:3b --keep-alive 10m
```

Plain open-ended chat without local context uses a fast serial greedy path and should report `estimated_speedup=1.00x`. MachBoost speedups require useful `--context` that predicts upcoming tokens.

Useful native options:

```sh
machboost list --backend mlx
machboost list --all
machboost run qwen2.5:3b --show-stats
machboost run qwen2.5:3b --direct
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --device mps --max-tokens 128
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --dtype float16 --show-stats
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --local-files-only
machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx --strict
machboost run mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx --context ./docs --ngram 1 --reentry-probe-tokens 1
machboost run qwen3-vl:8b --image ./document.png --vision-tokens adaptive --vision-token-ratio 0.35 --show-stats
```

`--reentry-probe-tokens` is experimental and disabled by default. `--direct` restores the one-process behavior for debugging. On Apple Silicon, a short alias prefers the MLX 4-bit model; explicit Hugging Face models default to `--device auto --dtype auto`, which selects MPS with float16 when available.

The package also includes install checks:

```sh
machboost doctor --json
machboost self-test --json
machboost version
```

An explicit Ollama wrapper remains available for compatibility with an existing Ollama installation:

```sh
machboost ollama run qwen2.5:3b
```

If the model is missing, MachBoost asks the local Ollama server to pull it first, then opens an interactive chat. Inside the chat, use `/bye`, `/exit`, `/quit`, EOF, or Ctrl-C to leave, and `/clear` to reset chat history.

Useful options:

```sh
machboost ollama run qwen2.5:3b --ctx 4096 --temperature 0
machboost ollama run llama3.2 --system "Answer concisely." --no-pull
```

This wrapper uses Ollama's public HTTP API and is not native MachBoost verifier acceleration. `machboost run` and `machboost chat` use the MachBoost resident runtime.

The repository also includes the original Go CLI for diagnostics, command wrapping, and local benchmark experiments:

```sh
go run ./cmd/machboost doctor
go run ./cmd/machboost doctor --json
go run ./cmd/machboost run --profile sustained --workload generic -- echo ok
go run ./cmd/machboost bench command -- sleep 1
go run ./cmd/machboost overlap --prompt prompt.txt --output output.txt --context .
```

The Go CLI is useful for local systems experiments. The Python package is the product-facing inference layer.

## Backends

| Backend | Status | Notes |
|---|---|---|
| MLX / `mlx-lm` | native adapter | Best Mac-first path. Strict evidence mode can disable prompt cache for clean exactness checks. |
| MLX-VLM | native visual adapter | Uses model-safe cache reuse for unchanged images and opt-in post-fusion compression for uncached Qwen3-VL requests. |
| Hugging Face Transformers | native adapter | Useful for research and broad model coverage. |
| MachBoost resident server | native control plane | Keeps MLX/HF models warm and exposes Ollama/OpenAI-compatible streaming APIs. |
| Custom Python service | native if verifier exists | Implement `next_token`, `verify`, `encode`, and `decode` as needed. |
| Ollama HTTP | wrapper only | Useful for benchmarking/capability detection; public HTTP does not expose logits/token IDs/KV hooks needed for exact acceleration. |

## Evidence

Public benchmark artifacts live in [results](results/), with a summary in [results/README.md](results/README.md). The current native-baseline evidence is:

| Artifact | Model | Path | Repeats | Exact Match | Median Paired Speedup |
|---|---|---:|---:|---:|---|
| `mlx_native_default_qwen25_3b_20260713.json` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | default code continuation | 5 | 100% | 1.96x |
| `mlx_native_reentry_qwen25_3b_20260713.json` | same | experimental RAG re-entry | 5 | 100% | 1.58x |
| `mlx_native_reentry_qwen25_3b_20260713.json` | same | open-ended native fallback | 5 | 100% | 1.08x |
| `vision_cache_qwen25_3b_20260714.json` | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | repeated questions over one image | 12 | 100% | 18.33x |
| `vision_cache_qwen3vl_2b_20260714.json` | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | repeated questions over one image | 12 | 100% | 11.41x |
| `vision_cache_qwen3vl_4b_20260714.json` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | same | 12 | 100% | 12.73x |
| `vision_cache_qwen3vl_8b_20260714.json` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | same | 12 | 100% | 16.69x |
| `vision_cache_qwen35_08b_20260714.json` | `mlx-community/Qwen3.5-0.8B-MLX-4bit` | same | 12 | 75% | 5.14x |
| `vision_cache_qwen35_4b_20260714.json` | `mlx-community/Qwen3.5-4B-MLX-4bit` | same | 12 | 100% | 14.29x |
| `vision_cache_qwen35_9b_20260714.json` | `mlx-community/Qwen3.5-9B-MLX-4bit` | same | 12 | 100% | 17.44x |
| `cold_vision_qwen3vl_8b_postfusion_20260715.json` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | unique-image TextVQA, 35% visual retention | 10 | 70% normalized output equality; 80%/80% task match | 1.70x |

Resident-server latency is tracked separately in `resident_qwen25_3b_20260713.json`. On the same M1 Max, five warm forced 64-token requests reached a 0.657-second median wall time and 97.5 end-to-end tok/s. Five short streaming chats reached a 0.298-second median time to first text and 0.358-second median total latency. These requests used native fallback with zero accepted draft tokens, so they measure warm serving rather than the context speculation algorithm.

The default code path accepted a median 51 of 64 tokens and reduced logical target forwards by 76.6%. One-token re-entry broadens coverage to copied RAG answers, accepting a median 30 tokens. The current repeated medians are below 2x and remain workload-specific. Older `strict` and 9B artifacts compared against synchronous or cache-disabled baselines and remain available only as diagnostics; they do not establish an improvement over optimized `mlx-lm` or Ollama.

The visual artifact compares 12 uncached requests with 12 accelerated requests on the same resident Qwen2.5-VL 3B model. Median wall time fell from 2.818 seconds uncached to 0.152 seconds on the accelerated path; median paired speedup was 18.33x and median TTFT speedup was 19.45x. All paired outputs and expected fixture answers matched. Eleven of 12 accelerated rows reused a 1,018-token visual prefix. The remaining row deliberately repeated the priming prompt, so it only reused projected image features and reached 1.33x. These are warm repeated-image results on one machine and model, not a claim about first-view latency, decode throughput, changed images, or arbitrary visual workloads.

The cross-model artifact `vision_cache_qwen_matrix_20260714.json` applies the same image, prompts, resident-process policy, generation settings, and alternating pair order to six Qwen models. Across 72 pairs, both modes answer every fixture correctly. The median of the six model-level paired medians is 13.51x, ranging from 5.14x to 17.44x. Literal output equality is 95.83%: the only drift is three Qwen3.5 0.8B rows that differ by a semicolon inside a JSON fence while returning the same expected answer. The median reusable-prefix pair is 12.96x. Qwen3-VL's three genuine no-prefix cache controls have a 0.99x median, confirming that cache reuse alone does not improve first-view work. Every Qwen3.5 row uses a guarded whole-state checkpoint, including the repeated priming prompt. Qwen3.6 is excluded because its official releases are 28B and 36B total parameters. Variant names are not always total multimodal size: Qwen3-VL-8B is listed as 9B total and Qwen3.5-9B as 10B total.

The first-view artifact `cold_vision_qwen3vl_8b_postfusion_20260715.json` instead uses 10 unique public TextVQA images, disables both visual and prompt caches, alternates pair order, and excludes a held-out warm-up. Adaptive post-fusion compression retains a median 35.12% of visual states after layer 3. Median wall time falls from 4.078 to 2.368 seconds; aggregate speedup is 1.67x and median paired speedup is 1.70x. Both paths match an accepted dataset answer on 80% of rows, but normalized output equality is 70% and literal equality is 50%. This is a quality-neutral task-score pilot on one model and machine, not exact decoding evidence or a universal first-view claim.

The research paper source and PDF are included in [paper](paper/). Keeping `paper/` and `results/` in the public repository is intentional: they make the claims auditable. They are not imported by the package at runtime.

## Reproduce Benchmarks

Run the paired native-MLX suite:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures code,rag,creative_open \
  --repeat 5 \
  --max-new-tokens 64 \
  --max-draft-tokens 32 \
  --ngram 3 \
  --source-mode context \
  --mlx-model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --output results/local/mlx_native_default.json
```

Test the opt-in one-token re-entry profile:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures code,rag,creative_open \
  --repeat 5 \
  --max-new-tokens 64 \
  --ngram 1 \
  --max-draft-tokens 32 \
  --reentry-probe-tokens 1 \
  --source-mode context \
  --mlx-model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --output results/local/mlx_native_reentry.json
```

The harness includes prompt processing in both paths, alternates baseline-first and boosted-first ordering, uses fresh nonces, and records environment provenance. For historical comparison with Hugging Face prompt lookup:

```sh
python3 scripts/hf_prompt_lookup_compare.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --fixtures real_readme_api,real_core_code,policy,json,rag,code \
  --max-new-tokens 32 \
  --prompt-lookup-sweep 4,8,16 \
  --machboost-source-modes prompt,context,prompt-context \
  --output results/local/hf_prompt_lookup_compare.json
```

Use `results/local/` for new local runs; it is ignored by git.

Run the repeated-image VLM benchmark:

```sh
python3 -m scripts.benchmark_vision_cache \
  --model qwen2.5-vl:3b \
  --repeats 3 \
  --max-tokens 16 \
  --output results/local/vision_cache_qwen25_3b.json
```

Run additional models with aliases such as `qwen3-vl:2b`, `qwen3-vl:4b`, `qwen3-vl:8b`, `qwen3.5:0.8b`, `qwen3.5:4b`, and `qwen3.5:9b`. Consolidate compatible artifacts with:

```sh
python3 scripts/summarize_vision_matrix.py results/local/vision_cache_*.json \
  --output results/local/vision_cache_matrix.json
```

Run the unique-image Qwen3-VL post-fusion benchmark:

```sh
python3 scripts/benchmark_cold_vision.py \
  --model qwen3-vl:8b \
  --datasets textvqa \
  --samples-per-dataset 10 \
  --max-tokens 16 \
  --cold-mode off \
  --vision-tokens adaptive \
  --vision-token-ratio 0.35 \
  --output results/local/cold_vision_qwen3vl_8b_postfusion.json
```

## Examples

Runnable examples live in [examples/python](examples/python/):

```sh
python3 examples/python/verifier_service_demo.py
python3 examples/python/black_box_service_demo.py
python3 examples/python/accelerator_calibration_demo.py
python3 examples/python/hf_adapter_demo.py
python3 examples/python/mlx_adapter_demo.py
python3 examples/python/vision_client_demo.py --image ./image.png
python3 examples/python/ollama_adapter_demo.py
```

The HF, MLX, and vision examples require the matching optional dependencies and locally available models.

## Development

Run tests:

```sh
python3 -m unittest discover -s tests
go test ./...
```

On macOS 26 or newer, use Go 1.24 or newer for `go test ./...`; earlier Go linkers do not emit the Mach-O `LC_UUID` command that modern `dyld` requires.

Check packaging:

```sh
python3 -m pip install --dry-run .
python3 -m pip wheel . -w /tmp/machboost-wheel --no-deps
```

Render the paper:

```sh
tectonic paper/machboost.tex --outdir paper
```

## Safety And Scope

MachBoost v1 does not:

- change global shell config
- mutate `launchctl`
- change Ollama service state
- change Docker Desktop settings
- change system power settings
- upload telemetry
- modify model weights

It only accelerates paths where the backend can verify candidate tokens against the target model.

## License

MIT. See [LICENSE](LICENSE).
