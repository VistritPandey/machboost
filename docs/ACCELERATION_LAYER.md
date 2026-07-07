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
- an OpenAI-compatible sidecar backed by a native runtime

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

The Python package currently exposes lightweight install checks:

```sh
machboost doctor
machboost self-test
machboost version
```

The Go CLI remains available from source for local systems experiments:

```sh
go run ./cmd/machboost doctor
go run ./cmd/machboost bench command -- sleep 1
```

## Adapter Capability Matrix

| Backend | Wrapper | Native Speedup | Status |
|---|---:|---:|---|
| Hugging Face | yes | yes | package adapter and benchmark scripts exist |
| MLX / `mlx-lm` | yes | yes | package adapter and strict evidence mode exist |
| Custom Python service | yes | yes, if verifier exists | supported through `machboost(...)` |
| Ollama HTTP | yes | no | benchmark/capability wrapper only |
| llama.cpp | planned | possible | needs verifier/KV hooks or patch |
| OpenAI-compatible servers | yes | only if owned | sidecar can wrap, but black-box acceleration is not claimed |

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

Status: in progress.

- Public Python API.
- Optional backend extras.
- Install doctor and self-test CLI.
- Examples and package docs.
- Calibration and gate policy APIs.

### P2: Runtime Expansion

Next targets:

- repeated larger-model evaluations
- cache-enabled MLX exactness work
- llama.cpp verifier hook investigation
- Ollama MLX runner patch or fork

### P3: Sidecar Server

Future target:

- OpenAI-compatible local endpoint
- streaming output
- per-request policy decisions
- native acceleration only when MachBoost owns or patches the runtime

## Product Principle

MachBoost should feel boring:

- If it can help, it speeds up exact output.
- If it cannot help, it stays out of the way.
- It always leaves an audit trail explaining what happened.
