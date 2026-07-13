# machboost

[![CI](https://github.com/VistritPandey/machboost/actions/workflows/ci.yml/badge.svg)](https://github.com/VistritPandey/machboost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MachBoost is an experimental Python package for exact local-context speculative acceleration of local LLM inference.

It drafts candidate tokens from nearby text such as prompts, retrieved chunks, repo files, policies, configs, and docs. A backend verifier then accepts only the tokens that match the target model's greedy continuation. When the local context predicts the next tokens well, MachBoost can reduce target-model calls without changing the generated token sequence.

MachBoost is local-first and alpha-stage. It does not upload telemetry, mutate global runtime settings, change model weights, or claim universal speedups.

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
pip install -e ".[all]"
```

After publishing this repository on GitHub, users can install directly from GitHub:

```sh
pip install "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git"
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

Start a native local chat from the command line:

```sh
machboost list
machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx --context ./docs --context ./src
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --show-stats
```

If the model is not already cached, the selected backend may download it through its normal Hugging Face or MLX loader. Use `--local-files-only` with the Hugging Face backend to require an existing local cache.

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

## When It Helps

MachBoost is most useful when the model is likely to continue with text already present nearby:

- repo or source-code continuation
- config and JSON generation
- policy or documentation copying
- RAG answers that quote retrieved context
- repeated logs, templates, checklists, and structured artifacts

It is usually neutral or slower for open-ended creative writing, one-word answers, and prompts where the next tokens are not recoverable from local context. The package exposes benchmark and calibration APIs so applications can turn the boosted path on only when it helps.

## Command Line

The Python package installs a native model runner:

```sh
machboost list
machboost list --json
machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --context ./docs --context ./src --show-stats
```

`machboost list` shows cached Hugging Face and MLX models that the native runner can likely load, plus backend readiness. `machboost run MODEL` loads a Hugging Face or MLX model, builds a MachBoost draft corpus from any `--context` files or directories, and opens a streaming interactive chat. Inside the chat, use `/bye`, `/exit`, `/quit`, EOF, or Ctrl-C to leave, and `/clear` to reset chat history.

Plain open-ended chat without local context uses a fast serial greedy path and should report `estimated_speedup=1.00x`. MachBoost speedups require useful `--context` that predicts upcoming tokens.

Useful native options:

```sh
machboost list --backend mlx
machboost list --all
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --device mps --max-tokens 128
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --dtype float16 --show-stats
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --local-files-only
machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx --strict
```

On Apple Silicon, the Hugging Face backend defaults to `--device auto --dtype auto`, which selects MPS with float16 when available. Ollama can still be faster for general chat because it uses optimized quantized runners; MachBoost's native path is mainly for exact local-context acceleration and research/debuggable verifier hooks.

The package also includes install checks:

```sh
machboost doctor --json
machboost self-test --json
machboost version
```

An Ollama-compatible chat wrapper is available for compatibility:

```sh
machboost ollama run qwen2.5:3b
machboost chat qwen2.5:3b
```

If the model is missing, MachBoost asks the local Ollama server to pull it first, then opens an interactive chat. Inside the chat, use `/bye`, `/exit`, `/quit`, EOF, or Ctrl-C to leave, and `/clear` to reset chat history.

Useful options:

```sh
machboost ollama run qwen2.5:3b --ctx 4096 --temperature 0
machboost ollama run llama3.2 --system "Answer concisely." --no-pull
```

This wrapper uses Ollama's public HTTP API. It is useful for a familiar local UX, model pulling, and chat, but it is not native MachBoost verifier acceleration.

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
| Hugging Face Transformers | native adapter | Useful for research and broad model coverage. |
| Custom Python service | native if verifier exists | Implement `next_token`, `verify`, `encode`, and `decode` as needed. |
| Ollama HTTP | wrapper only | Useful for benchmarking/capability detection; public HTTP does not expose logits/token IDs/KV hooks needed for exact acceleration. |

## Evidence

Public benchmark artifacts live in [results](results/), with a summary in [results/README.md](results/README.md). The current native-baseline evidence is:

| Artifact | Model | Path | Repeats | Exact Match | Median Paired Speedup |
|---|---|---:|---:|---:|---|
| `mlx_native_adaptive_qwen25_3b_20260713.json` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | adaptive code continuation | 3 | 100% | 2.36x |
| `mlx_native_adaptive_qwen25_3b_20260713.json` | same | native RAG fallback | 3 | 100% | 0.96x |
| `mlx_native_adaptive_qwen25_3b_20260713.json` | same | native open-ended fallback | 3 | 100% | 1.00x |

The accelerated code path accepted 51 of 64 tokens in every repeat and reduced logical target forwards from 64 to 14. This is a conditional result, not a universal 2x claim. Older `strict` and 9B artifacts compared against synchronous or cache-disabled baselines and remain available only as diagnostics; they do not establish an improvement over optimized `mlx-lm` or Ollama.

The research paper source and PDF are included in [paper](paper/). Keeping `paper/` and `results/` in the public repository is intentional: they make the claims auditable. They are not imported by the package at runtime.

## Reproduce Benchmarks

Run the paired native-MLX suite:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures code,rag,creative_open \
  --repeat 3 \
  --max-new-tokens 64 \
  --max-draft-tokens 32 \
  --ngram 3 \
  --source-mode context \
  --mlx-model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --output results/local/mlx_native_adaptive.json
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

## Examples

Runnable examples live in [examples/python](examples/python/):

```sh
python3 examples/python/verifier_service_demo.py
python3 examples/python/black_box_service_demo.py
python3 examples/python/accelerator_calibration_demo.py
python3 examples/python/hf_adapter_demo.py
python3 examples/python/mlx_adapter_demo.py
python3 examples/python/ollama_adapter_demo.py
```

The HF and MLX examples require the matching optional dependencies and locally available models.

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
