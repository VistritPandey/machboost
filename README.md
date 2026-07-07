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
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --context ./docs --show-stats
```

If the model is not already cached, the selected backend may download it through its normal Hugging Face or MLX loader. Use `--local-files-only` with the Hugging Face backend to require an existing local cache.

Use the high-level `Accelerator` when you want MachBoost to load a model and build the draft corpus from strings, files, or directories:

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
        "Continue the rollout checklist from the local docs:",
        "Copy the JSON deployment policy from the local config:",
    ],
    max_tokens=32,
    gate_policy=GatePolicy(min_speedup=1.05, min_acceptance_rate=0.10),
)

text, stats = boost.generate(
    "Continue the rollout checklist from the local docs:",
    max_tokens=128,
)

print(text)
print(stats.estimated_speedup)
print(calibration.summary)
```

Hugging Face causal language models use the same shape:

```python
from machboost import Accelerator

boost = Accelerator.from_huggingface(
    "Qwen/Qwen2.5-3B-Instruct",
    context_paths=["./docs", "./src"],
    local_files_only=True,
)

text, stats = boost.generate("Continue from the local context:", max_tokens=64)
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

`machboost list` shows cached Hugging Face and MLX models that the native runner can likely load, plus backend readiness. `machboost run MODEL` loads a Hugging Face or MLX model, builds a MachBoost draft corpus from any `--context` files or directories, and opens an interactive chat. Inside the chat, use `/bye`, `/exit`, `/quit`, EOF, or Ctrl-C to leave, and `/clear` to reset chat history.

Useful native options:

```sh
machboost list --backend mlx
machboost list --all
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --device mps --max-tokens 128
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --local-files-only
machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx --strict
```

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

Public benchmark artifacts live in [results](results/), with a summary in [results/README.md](results/README.md). The current headline artifacts are:

| Artifact | Model | Rows | Exact Match | Median Speedup | Notes |
|---|---|---:|---:|---:|---|
| `mlx_evidence_v2_strict_aggregate_20260706.json` | `mlx-community/Qwen3.5-0.8B-MLX-4bit` | 90 | 100% | 3.00x | repeated strict MLX run |
| `hf_prompt_lookup_compare_qwen25_3b.json` | `Qwen/Qwen2.5-3B-Instruct` | 42 method rows | 100% over requested budget | 2.47x for MachBoost context | direct comparison with HF prompt lookup |
| `mlx_qwen35_9b_strict_smoke.json` | `mlx-community/Qwen3.5-9B-MLX-4bit` | 2 | 100% | 7.50x | larger-model smoke test |

The research paper source and PDF are included in [paper](paper/). Keeping `paper/` and `results/` in the public repository is intentional: they make the claims auditable. They are not imported by the package at runtime.

## Reproduce Benchmarks

Run the strict MLX suite:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures real_readme_api,real_core_code,real_paper_method,policy,json,rag,code,repo_quote,creative_open,short_answer \
  --repeat 3 \
  --max-new-tokens 64 \
  --max-draft-tokens 8 \
  --ngram 2 \
  --candidate-limit 1 \
  --source-mode context \
  --mlx-disable-cache \
  --mlx-model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --output results/local/mlx_strict.json
```

Compare against Hugging Face prompt lookup:

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
