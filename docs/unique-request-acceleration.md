# Unique-request acceleration

MachBoost treats first-request latency as two separate problems:

1. **Prefill** processes the prompt, retrieved repository chunks, and visual inputs.
2. **Decode** produces output tokens autoregressively.

A method that improves one phase must not be reported as if it improved the other.
Prefix caches reduce repeated prefill. They do not make a new output sequence decode
faster. DFlash reduces target decode passes on supported text models. It does not
remove the first prefill, and it does not currently accelerate vision decoding.

## Verified DFlash decoding

The optional `dflash` backend uses a small block-diffusion draft model to propose a
block of future tokens in parallel. The target model verifies those proposals and
only target-approved tokens are emitted. MachBoost keeps the target and draft
resident behind the same bounded, concurrent API used by its other native backends.

Install the optional runtime and run a supported alias:

```sh
pip install -e ".[dflash]"
machboost pull qwen3.5:4b-dflash
machboost run qwen3.5:4b-dflash --show-stats
```

The `-dflash` suffix is a normal MachBoost model name. OpenAI- and
Ollama-compatible clients select it through their standard `model` field; they do
not need a MachBoost-specific request extension. Pulling the alias downloads both
the target and its paired draft model, and catalog cache state is ready only when
both repositories are present.

```json
{
  "model": "qwen3.5:4b-dflash",
  "messages": [{"role": "user", "content": "Explain a mutex."}],
  "temperature": 0,
  "stream": true
}
```

Benchmark fresh prompts against ordinary autoregressive generation from the same
target weights:

```sh
machboost bench-decode qwen3.5:4b \
  --prompt-file benchmarks/unique_decode_prompts.jsonl \
  --runs 3 \
  --max-tokens 512 \
  --no-eos \
  --output results/local/qwen35-4b-decode
```

`bench-decode` runs every non-empty JSONL row unless `--limit` is supplied. It also
compares native and accelerated greedy outputs for the first 128 generated tokens,
stores only token hashes and mismatch metadata, and exits nonzero if any sequence
differs. Set `--validation-tokens 0` only when intentionally collecting a
non-equivalent throughput diagnostic. The `--no-eos` setting is useful for
steady-state throughput comparisons; it is not a realistic chat-latency
measurement. Run without it when measuring complete answers.

The default adaptive verifier reduces the proposed block when recent acceptance is
too low. A fixed `--verify-mode dflash` can be faster for highly predictable output,
but it can regress workloads with lower acceptance. Always calibrate against the
same target weights and representative prompts before selecting it.

## Current Apple M5 Pro evidence

The checked-in three-prompt suite covers mathematical reasoning, code generation,
and an operations plan. Each throughput row uses 512 generated tokens, three
measured repetitions, greedy decoding, and the same loaded target weights for native
and accelerated legs.

| Target | Verifier | Native | DFlash | Median | Prompt range | Output gate |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B BF16 | adaptive | 29.36 tok/s | 48.52 tok/s | 1.65x | 1.31-2.43x | 3/3 exact at 128 tokens |
| Qwen3.5 9B 4-bit | adaptive | 54.63 tok/s | 68.32 tok/s | 1.32x | 1.22-1.73x | divergence observed; experimental |
| Qwen3.5 9B 4-bit | fixed 16 | 51.43 tok/s | 44.14 tok/s | 0.84x | 0.76-1.63x | not promoted |

The 4B row was reproduced with the shippable `dflash-mlx==0.1.8` wheel. The 9B
adaptive result is useful as a practical quantized control, but its greedy
output did not always match native MLX. The 4B BF16 path passed the strict sequence
gate on this suite. Neither result establishes quality or equivalence for unseen
prompts, another runtime version, or a different quantization.

## Boundaries

- DFlash is explicit opt-in and only supports model/draft pairs published for the
  target architecture. Unsupported models continue to use their native backend.
- The current integration is greedy-only and rejects custom stop strings. These
  restrictions avoid presenting an unverified sampling path as equivalent.
- Target verification means no draft-only token is emitted. It does not guarantee
  byte equality with a different runtime, prompt template, quantization, or floating
  point dispatch path.
- `mlx-lm` currently has an open report of greedy speculative decoding diverging
  from plain generation. MachBoost therefore measures token equality instead of
  inferring it from the algorithm's verifier contract.
- Full-precision DFlash and quantized native generation answer different practical
  questions. Same-weight speedup measures the decoding algorithm; absolute tokens
  per second against a quantized runtime measures user-visible performance.
- Short answers can see little benefit because model load, prefill, and first-token
  latency dominate. Long generations expose decode throughput more clearly.
- Additional draft weights increase disk and unified-memory use.

## Phase-aware roadmap

No single mechanism covers all first requests. The intended runtime policy is:

| Request phase | Exact acceleration path | Current state |
|---|---|---|
| New long prompt | Hardware-specific prefill kernels | research; not claimed by MachBoost |
| New text output | Target-verified DFlash decode | implemented for selected Qwen targets |
| Reused repository or system prefix | Exact prompt-state reuse | implemented |
| Identical deterministic request | Scoped exact-response reuse | implemented, opt-in |
| Concurrent team traffic | Continuous admission, fairness, and replicas | implemented; aggregate throughput differs from single-request latency |
| New visual input | Model-specific token reduction | experimental and approximate |

Two recent directions inform the next experiments. BaseRT reports M5 Metal 4 tensor
kernels that improve prompt processing by up to 3.9x over MLX while its decode gain
is smaller. Speculative Speculative Decoding overlaps drafting and verification on
separate compute resources and reports up to 5x over autoregressive generation in a
multi-GPU system. Neither result transfers automatically to one Apple GPU. A useful
MachBoost implementation would need a reproduced same-model benchmark, an explicit
license boundary, and a no-regression routing gate.

## Related work

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [DFlash MLX runtime](https://github.com/bstnxbt/dflash-mlx)
- [MLX-LM greedy speculative divergence report](https://github.com/ml-explore/mlx-lm/issues/1470)
- [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251)
- [BaseRT on M5 neural accelerators](https://arxiv.org/abs/2607.19438)
- [MLX-LM EAGLE-3 Apple Silicon analysis](https://github.com/ml-explore/mlx-lm/discussions/890)
