# MachBoost Acceleration Layer

MachBoost is an exact local-context speculative acceleration layer. It drafts candidate tokens from nearby text and asks the target runtime to verify those tokens before committing them.

The central question is:

> Can local context safely draft tokens that the target model would have generated anyway?

If yes, MachBoost can reduce target-model work. If no, the package falls back to normal generation and records why.

## Non-Goals

- No quality tradeoff.
- No quantization requirement.
- No global system mutation.
- No hosted service dependency.
- No claim that black-box inference can be accelerated without runtime support.

## Runtime Classes

### Native Acceleration

This is where real speedups happen. The runtime must expose enough internals to:

- tokenize and detokenize text
- compute next-token logits or greedy decisions
- verify a candidate token span against the target model
- advance or rebuild KV/cache state after accepting a verified prefix

Current package adapters:

- Hugging Face Transformers
- MLX / `mlx-lm`
- custom Python services that implement `next_token` and `verify`

Future native targets:

- llama.cpp / llama-server
- an Ollama runner patch or fork

### Resident Runtime

MachBoost 0.2 adds a long-running control plane around the native adapters. It:

- loads each model once and retains it in unified memory
- streams generated text without re-decoding the entire prefix per token
- applies finite or indefinite model keep-alive policies
- serializes generation per model while serving independent models concurrently
- exposes both Ollama-compatible and OpenAI-compatible HTTP endpoints
- supports explicit preload, inspection, stop, and shutdown operations

Resident serving removes repeated model-loading costs and makes MachBoost usable by editors, chat clients, scripts, and internal assistants. It is an operational latency improvement; it is separate from speculative token acceleration and should be measured separately.

### Wrapper / Policy Mode

Black-box local servers can still be wrapped for diagnostics, benchmarking, and option management. They cannot receive exact MachBoost speedups unless they expose a verifier API.

Examples:

- Ollama HTTP
- OpenAI-compatible local servers
- existing long-running model daemons

For these, MachBoost can still provide:

- workload classification
- context-overlap analysis
- benchmark comparison
- capability reports
- native-adapter recommendations

## Architecture

```text
Prompt + local context
        |
        v
Context Router ---> Policy Gate ------ no ----> serial generation
        |              |
        |             yes
        v              v
Candidate Drafter -> Runtime Verifier -> accepted prefix -> output
        |              |
        v              v
Results Recorder <----+
```

## Core Interfaces

### Candidate Drafter

The current drafter indexes token n-grams from a caller-provided corpus. At generation time it matches suffixes of the current prefix and proposes the tokens that followed the best matching span.

Current behavior:

- n-gram local-context lookup
- longest suffix preference
- configurable maximum draft length
- optional multiple candidate attempts

Likely improvements:

- suffix arrays or suffix automata for faster lookup
- source locality scoring
- prompt-visible source priority
- trie/tree candidate packing
- retrieval-score weighting

### Runtime Verifier

The verifier checks candidate tokens against the target model.

Current behavior:

- greedy exact-match verification
- accepted-prefix commits
- residual-token fallback on mismatch where supported
- MLX strict/stateless mode for clean evidence runs

Important limitation: sampling-compatible verification is not claimed in v1. Current public evidence is for greedy decoding and exact token equality.

### Policy Gate

The policy gate decides whether speculation should run.

Inputs:

- benchmark speedup
- exact-match status
- acceptance rate
- accepted draft span length
- target-call or forward-call reduction

Outputs:

- `enabled`: speculation is likely useful
- `disabled`: serial generation is safer or faster

The policy gate is product-critical because local-context drafting is not universal. It should accelerate grounded workflows and stay out of the way for open-ended prompts.

### Results Recorder

Evidence and benchmark rows should be machine-readable:

- model and backend
- fixture or workload type
- baseline and boosted tokens/sec
- exact-match rate
- accepted draft tokens
- target-model forward reduction
- policy decision
- warnings and mode flags

This keeps future dashboards, CI budgets, and technical reports possible without adding SaaS to v1.

## Public Interfaces

### Python API

```python
from machboost import Accelerator

accel = Accelerator.from_mlx(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    context_paths=["README.md", "docs/"],
)

text, stats = accel.generate(prompt, max_tokens=128)
```

### Resident Client

```python
from machboost import MachBoostClient

client = MachBoostClient()
client.load("qwen2.5:3b", keep_alive="forever")

for chunk in client.chat(
    "qwen2.5:3b",
    [{"role": "user", "content": "Explain the retry logic."}],
):
    print(chunk, end="", flush=True)
```

### Custom Service

```python
from machboost import machboost

boosted = machboost(
    service,
    corpus_tokens=local_context_tokens,
    ngram=4,
    max_draft_tokens=8,
)

tokens, stats = boosted.generate(prompt_tokens, max_tokens=128)
```

### CLI

The Python package exposes a resident native model runner:

```sh
machboost list
machboost pull qwen2.5:3b
machboost warm qwen2.5:3b --keep-alive forever
machboost run qwen2.5:3b --context ./docs --show-stats
machboost complete qwen2.5-coder:3b --file ./prompt.txt
machboost ps
machboost stop qwen2.5:3b
```

`machboost list` reports cached Hugging Face and MLX models plus portable short aliases. `machboost run MODEL` auto-starts the local server, resolves the best available backend, loads the model when needed, builds a draft corpus from local context files or directories, and opens an interactive streaming chat. Leaving chat does not unload the model. Use `--direct` for the earlier one-process behavior.

The package also exposes lightweight install checks:

```sh
machboost doctor
machboost self-test
machboost version
```

The native server listens on `http://127.0.0.1:11435` and implements Ollama-compatible chat, generate, model-listing, and lifecycle endpoints plus OpenAI-compatible chat and text completion endpoints. The external Ollama wrapper remains available separately:

```sh
machboost ollama run qwen2.5:3b
```

That explicit wrapper can pull and chat with models owned by an Ollama daemon, but it is not the native verifier-accelerated path. `machboost chat` is an alias for native `machboost run`.

The Go CLI remains available from source for local systems experiments:

```sh
go run ./cmd/machboost doctor
go run ./cmd/machboost bench command -- sleep 1
```

## Adapter Capability Matrix

| Backend | Resident serving | Native speculation | Status |
|---|---:|---:|---|
| Hugging Face | yes | yes | native adapter, streaming server, and benchmarks exist |
| MLX / `mlx-lm` | yes | yes | native adapter, fast text streaming, and strict evidence mode exist |
| Custom Python service | caller-owned | yes, if verifier exists | supported through `machboost(...)` |
| External Ollama HTTP | already resident | no | compatibility wrapper and benchmarks only |
| llama.cpp | planned | possible | needs verifier/KV hooks or patch |

Protocol compatibility does not imply full feature parity. Version 0.2 does not yet provide Ollama model creation/copy/deletion, embeddings, or multimodal requests.

## Milestones

### P0: Evidence Runner

Status: done.

- Hugging Face verifier prototype.
- MLX package adapter.
- Repeatable benchmark suites.
- Direct Hugging Face prompt-lookup comparison.
- Strict MLX evidence artifacts.
- Exact-match JSON reports.

### P1: Package Layer

Status: done for 0.2.

- Public Python API.
- Optional backend extras.
- Install doctor and self-test CLI.
- Examples and package docs.
- Calibration and gate policy APIs.

### P2: Resident Serving

Status: done for 0.2.

- Long-running Hugging Face/MLX model ownership.
- Streaming CLI chat and raw completion.
- Model alias resolution and backend selection.
- Pull, preload, inspect, unload, and shutdown lifecycle.
- Ollama-compatible and OpenAI-compatible local APIs.
- End-to-end resident latency evidence.

### P3: Runtime Expansion

Next targets:

- repeated larger-model evaluations
- cache-enabled MLX exactness work
- llama.cpp verifier hook investigation
- Ollama MLX runner patch or fork

### P4: Multimodal Runtime

Next product target after text-runtime hardening:

- image and video request schemas
- multimodal model capability discovery
- image preprocessing and prompt-cache reuse
- time-to-first-token and end-to-end visual-task benchmarks
- exact input/output comparisons where deterministic evaluation is possible

## Product Principle

MachBoost should feel boring:

- If it can help, it speeds up exact output.
- If it cannot help, it stays out of the way.
- It always leaves an audit trail explaining what happened.
