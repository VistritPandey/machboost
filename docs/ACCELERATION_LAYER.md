# MachBoost Acceleration Layer

## Goal

Turn the current Hugging Face prototype into a local inference acceleration layer that can sit beside developer workflows without changing model output.

The layer should make one decision repeatedly:

> Can local context safely draft tokens that the target model would have generated anyway?

If yes, MachBoost verifies and accepts those tokens faster than serial decoding. If no, MachBoost falls back to normal generation and records why.

## Non-Goals

- No quality tradeoff.
- No quantization requirement.
- No global system mutation.
- No hosted service dependency.
- No claim that black-box inference can be accelerated without runtime support.

## Runtime Reality

There are two integration classes.

### Native Acceleration

This is where real speedups happen. The runtime must expose enough internals to:

- Tokenize and detokenize text.
- Prefill a prompt and retain KV cache state.
- Read logits or greedy next-token decisions.
- Verify a candidate token span against the target model.
- Crop or advance KV cache after accepting a verified prefix.

Good targets:

- Hugging Face Transformers: current prototype.
- MLX / mlx-lm: best Mac-first next target.
- llama.cpp: likely production target if verifier hooks are available or patchable.
- Custom local Python runtimes.

### Wrapper / Policy Mode

Black-box local servers can still be wrapped for diagnostics, benchmarking, and policy reports, but not true verifier acceleration unless they expose a draft/verify API.

Examples:

- Existing Ollama HTTP API.
- OpenAI-compatible local servers.
- Existing long-running model daemons.

For these, MachBoost can still provide:

- Workload classification.
- Context-overlap analysis.
- Benchmark comparison.
- “Acceleration likely / unlikely” reports.
- Native-adapter recommendations.

## Layer Architecture

```text
Prompt + local context
        |
        v
Context Router ---> Policy Gate ------ no ----> normal generation
        |              |
        |             yes
        v              v
Candidate Drafter -> Runtime Verifier -> accepted prefix -> stream/output
        |              |
        v              v
Results Recorder <----+
```

## Core Interfaces

### Context Router

Collects possible draft sources:

- Prompt text.
- Retrieved RAG chunks.
- Local files.
- Recent transcript or generated output.
- Structured templates.

The router must score source quality and avoid adding irrelevant context just because it exists.

### Candidate Drafter

Produces candidate continuations from local context.

Current implementation:

- N-gram local-context lookup.
- Longest suffix match.
- Multiple candidate lengths.

Likely next improvements:

- Source locality scoring.
- Prompt-visible source priority.
- Trie/tree candidate packing.
- Repetition/template detection.
- Retrieval-score weighting.

### Runtime Verifier

Checks draft tokens against the target model.

Current implementation:

- Greedy exact-match verification.
- Hybrid verification: step-verify an anchor token, then block-verify the tail.
- Verified-prefix acceptance when a later token mismatches.

Future verifier work:

- Tree/trie multi-candidate verification.
- Runtime-specific KV cache optimizations.
- Sampling-compatible verification modes where possible.

### Policy Gate

Decides whether speculation should run.

Inputs:

- Workload type.
- Context overlap.
- Early acceptance rate.
- Draft span length.
- Verification overhead.
- Exact-match status.

Outputs:

- `enable`: speculation is likely useful.
- `neutral`: safe but not clearly useful.
- `disable`: use normal generation.

The policy gate is the product layer. It prevents the research mechanism from becoming an always-on slowdown.

### Results Recorder

Every run should emit machine-readable results:

- Model and backend.
- Fixture or workload type.
- Baseline total/decode tokens per second.
- Boosted total/decode tokens per second.
- Exact-match rate.
- Accepted draft tokens.
- Target-model forward reduction.
- Policy decision.
- Warnings.

This keeps future dashboards, CI budgets, and technical reports possible without adding SaaS to v1.

## Proposed Public Interfaces

### CLI

```sh
machboost accel probe --context . --prompt prompt.txt --json
machboost accel bench --backend hf --model Qwen/Qwen2.5-3B-Instruct --fixtures use_cases --repeat 5
machboost accel serve --backend mlx --model ./model --port 11435
```

### Python

```python
from machboost import Accelerator

accel = Accelerator.from_hf(
    model="Qwen/Qwen2.5-3B-Instruct",
    context_paths=["README.md", "docs/"],
    policy="auto",
)

for token in accel.generate(prompt, max_new_tokens=128):
    print(token, end="", flush=True)
```

### OpenAI-Compatible Sidecar

```sh
machboost accel serve --backend mlx --model ./model --openai-compatible
```

Apps can then point at MachBoost as a local OpenAI-compatible endpoint. This only provides true acceleration when MachBoost owns the runtime or the backend exposes verification hooks.

## Adapter Capability Matrix

| Backend | Wrapper | Native Speedup | Notes |
|---|---:|---:|---|
| Hugging Face | yes | yes | Prototype exists. Good for research, slower absolute Mac speed. |
| MLX | planned | planned | Best Mac-first product target. |
| llama.cpp | planned | possible | Needs verifier/KV hooks or patch. |
| Ollama HTTP | yes | not yet | Existing API is black-box; native integration would need deeper support. |
| OpenAI-compatible local servers | yes | only if owned | Sidecar can expose API, but cannot accelerate arbitrary black-box servers. |

## Milestones

### P0: Evidence Runner

Status: done.

- HF verifier prototype.
- Repeatable benchmark suite.
- Use-case and negative-control fixtures.
- Exact-match JSON reports.

### P1: Policy Gate

Build a short probe that classifies workloads as `enable`, `neutral`, or `disable`.

Acceptance criteria:

- Keeps strong use-case wins.
- Avoids slowdowns on negative controls.
- Emits stable JSON.

### P2: MLX Adapter

Port the verifier to MLX/`mlx-lm`.

Acceptance criteria:

- Same exact-match guarantees.
- Better absolute tokens per second than HF/MPS.
- Repeatable gains on at least RAG, config, docs, tests, and logs.

### P3: Sidecar Server

Expose an OpenAI-compatible local endpoint backed by a native adapter.

Acceptance criteria:

- Existing local clients can switch base URL.
- Streaming works.
- Policy gate can disable speculation per request.

### P4: llama.cpp / Ollama Path

Investigate whether verifier hooks can land in llama.cpp or be exposed through an adapter.

Acceptance criteria:

- No output quality regression.
- Clear answer on whether Ollama can support native verification without service patching.

## Product Principle

MachBoost should feel boring:

- If it can help, it speeds up exact output.
- If it cannot help, it stays out of the way.
- It always leaves an audit trail explaining what happened.
