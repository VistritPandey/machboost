# MachBoost Evidence Runs

This directory stores public benchmark artifacts for the local-context speculative decoding prototype.

## Current Native-Baseline Evidence, July 13 2026

Artifacts:

- `mlx_native_default_qwen25_3b_20260713.json`
- `mlx_native_reentry_qwen25_3b_20260713.json`

The current harness compares MachBoost against `mlx-lm` native streaming generation, includes prompt processing in both wall-clock measurements, alternates baseline-first and boosted-first order, records fresh nonces, and preserves every raw row. Each artifact also records package versions, hardware, memory, timestamps, and thermal status.

Model: `mlx-community/Qwen2.5-3B-Instruct-4bit`

Hardware: Apple M1 Max, 32 GB unified memory

Generation: greedy, 64 requested tokens, five fresh-nonce repeats per fixture

### Default profile

Settings: 3-gram context lookup, 32-token drafts, no native-token re-entry.

| Fixture | Selected Path | Exact Match | Median Speedup | Baseline tok/s | MachBoost tok/s |
|---|---|---:|---:|---:|---:|
| `code` | adaptive context verifier | 100% | 1.96x | 87.46 | 167.28 |
| `rag` | native fallback | 100% | 1.04x | 89.98 | 91.61 |
| `creative_open` | native fallback | 100% | 1.00x | 99.95 | 98.27 |

The code fixture accepted a median 51 draft tokens and reduced logical target forwards by 76.6%. Its paired speedups were 2.44x, 1.15x, 2.08x, 1.96x, and 1.91x, showing substantial short-run variance.

### Experimental one-token re-entry

Settings: 1-gram context lookup, 32-token drafts, one native seed token before re-entry.

| Fixture | Selected Path | Exact Match | Median Speedup | Baseline tok/s | MachBoost tok/s |
|---|---|---:|---:|---:|---:|
| `code` | adaptive context verifier | 100% | 1.62x | 72.54 | 121.38 |
| `rag` | adaptive context verifier | 100% | 1.58x | 88.92 | 140.09 |
| `creative_open` | native fallback | 100% | 1.08x | 94.25 | 101.57 |

Re-entry broadens useful coverage: RAG accepts a median 30 draft tokens after one native seed and reduces logical target forwards by 43.8%. A longer 3-token re-entry probe produced one mismatch in a separate 15-row exploratory run, so re-entry is opt-in in 0.1.4.

The current repeated default result is close to, but below, 2x. This is not a universal acceleration result. The implementation fuses verifier continuation with the next draft block, rewinds the MLX KV cache in place, matches native MLX prompt prefill, and resumes native asynchronous decoding when context candidates end.

## Runtime Suitability Probe, July 13 2026

Artifact: `runtime_probe_qwen25_3b_20260713.json`

A separate forced 64-token decode probe compared Qwen2.5 3B runtimes on the same M1 Max. Native `mlx-lm` reached a median 127.41 tok/s, while Ollama reached 94.68 tok/s, a 1.35x MLX advantage. `vllm-mlx` 0.4.0 reached 15.80 tok/s for five serial prompts and 93.19 tok/s aggregate for five concurrent prompts in its built-in benchmark. The latter improves concurrent completion time but does not improve single-request decode latency.

This is not an exact-weight comparison: MLX and Ollama use different 4-bit formats. The artifact records raw rates, versions, commands' workload shape, and additional caveats. It supports selecting native MLX for Apple Silicon single-stream chat; it does not support a universal 2x claim.

## Legacy Diagnostic Artifacts

Artifacts below this point predate the native MLX baseline and the adaptive fallback path. In particular, the July 6 "strict" MLX runs compared against a synchronous/stateless path near 10 tok/s and must not be used to claim a 3x to 8x improvement over optimized `mlx-lm`, Ollama, or another production runtime. They remain tracked for implementation history and regression analysis.

## Qwen2.5-3B Use Cases, Repeat 3

Command:

```sh
python3 scripts/hf_bench_suite.py \
  --runner in-process \
  --repeat 3 \
  --fixtures use_cases,negative_controls \
  --draft-policy fixed \
  --local-files-only \
  --output results/qwen25_3b_use_cases_r3.json
```

Model: `Qwen/Qwen2.5-3B-Instruct`

Runner: in-process Hugging Face/MPS, one model load reused across fixtures.

Generation: greedy, 32 new tokens, exact-match checked against baseline output.

Artifacts:

- `qwen25_3b_use_cases_r3.json`: full machine-readable results.
- `qwen25_3b_use_cases_r3.txt`: terminal summary table.

| Fixture | Workflow | Expected | Exact Match | Decode Speedup | Forwards |
|---|---|---:|---:|---:|---:|
| `prompt_visible_readme` | docs continuation | positive | 100% | 1.78x | 32 -> 9 |
| `prompt_visible_code` | code completion | positive | 100% | 1.44x | 32 -> 18 |
| `rag_answer` | RAG answer | positive | 100% | 2.32x | 32 -> 9 |
| `log_template` | log generation | positive | 100% | 1.55x | 32 -> 15 |
| `json_config` | structured config | positive | 100% | 2.25x | 32 -> 9 |
| `test_boilerplate` | test generation | positive | 100% | 1.75x | 32 -> 15 |
| `repo_chat_quote` | repo chat | positive | 100% | 1.09x | 32 -> 27 |
| `controlled_context` | controlled | positive | 100% | 1.97x | 32 -> 13 |
| `hidden_readme` | hidden context | negative | 100% | 1.01x | 32 -> 32 |
| `creative_open` | creative generation | negative | 100% | 1.00x | 32 -> 32 |
| `short_answer` | short answer | negative | 100% | 1.04x | 32 -> 32 |

Takeaways:

- Strongest repeatable wins are RAG, structured config, controlled/reference-heavy continuation, docs, logs, and raw test/code continuation.
- Negative controls stayed neutral, which supports the policy goal: enable speculation only for context-grounded workflows.
- Repo-chat quote is weak in this fixture because the model does not reliably copy the intended command.

## Hugging Face Prompt Lookup Comparison, Qwen2.5-3B

Command:

```sh
python3 scripts/hf_prompt_lookup_compare.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --fixtures real_readme_api,real_core_code,policy,json,rag,code \
  --repeat 1 \
  --max-new-tokens 32 \
  --prompt-lookup-sweep 4,8,16 \
  --machboost-source-modes prompt,context,prompt-context \
  --output results/hf_prompt_lookup_compare_qwen25_3b.json
```

Model: `Qwen/Qwen2.5-3B-Instruct`

Runner: in-process Hugging Face/MPS, one model load reused across fixtures.

Generation: greedy, 32 requested new tokens. Output match compares token IDs over the requested 32-token budget. The artifact also records `raw_generated_tokens`; Hugging Face prompt lookup emitted extra tail tokens in some rows after the matching 32-token prefix.

Artifact:

- `hf_prompt_lookup_compare_qwen25_3b.json`: full machine-readable comparison against Hugging Face `prompt_lookup_num_tokens`.

Overall:

| Method | Rows | Exact Match | Median Speedup | P90 Speedup | Median tok/s | Median Forwards | Forward Reduction | Accepted Draft Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hf_serial_generate` | 6 | 100% | 1.00x | 1.00x | 21.74 | 32.0 | 0.0% | 0.0 |
| `hf_prompt_lookup_4` | 6 | 100% | 1.83x | 1.96x | 38.68 | 8.5 | 73.4% | 0.0 |
| `hf_prompt_lookup_8` | 6 | 100% | 1.86x | 2.42x | 41.13 | 5.5 | 82.8% | 0.0 |
| `hf_prompt_lookup_16` | 6 | 100% | 2.11x | 3.61x | 46.59 | 3.0 | 90.6% | 0.0 |
| `machboost_prompt` | 6 | 100% | 2.14x | 2.41x | 45.77 | 10.5 | 67.2% | 30.5 |
| `machboost_context` | 6 | 100% | 2.47x | 2.70x | 51.06 | 10.5 | 67.2% | 30.5 |
| `machboost_prompt-context` | 6 | 100% | 2.26x | 2.44x | 47.63 | 11.0 | 65.6% | 30.0 |

Selected per-fixture comparison:

| Fixture | HF Prompt Lookup 16 | MachBoost Context | Note |
|---|---:|---:|---|
| `real_readme_api` | 2.03x | 2.64x | local README continuation favors context corpus |
| `real_core_code` | 2.12x | 2.19x | roughly tied |
| `policy` | 2.10x | 2.43x | context corpus wins |
| `json` | 3.91x | 2.50x | HF prompt lookup wins |
| `rag` | 1.95x | 1.68x | HF prompt lookup wins |
| `code` | 3.32x | 2.77x | HF prompt lookup wins |

Takeaways:

- Hugging Face prompt lookup is a strong baseline, not a straw man. On this 3B run, `prompt_lookup_num_tokens=16` reaches a 2.11x median speedup with exact 32-token prefix agreement.
- MachBoost context mode is still competitive and wins the median in this fixture mix at 2.47x, especially on local README and policy continuation.
- The defensible MachBoost product difference is not "n-gram lookup exists." It is local-corpus source control, adapter packaging, calibration/gating, and machine-readable evidence around when to enable the layer.
- The comparison should be repeated before making paper-grade claims; this artifact is a first direct baseline check with one repeat.

## MLX Strict Evidence V2, 64 Tokens, Repeat 3

Command:

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
  --output results/mlx_evidence_v2_strict_c1_20260706.json
```

Model: `mlx-community/Qwen3.5-0.8B-MLX-4bit`

Runner: MLX package adapter with prompt cache disabled. This strict mode compares boosted generation against the same stateless greedy baseline, avoiding cache-trajectory drift seen in longer cache-enabled diagnostic runs.

Generation: greedy, 64 new tokens, exact token-match checked against baseline output.

Artifact:

- `mlx_evidence_v2_strict_c1_20260706.json`: full machine-readable results.

Overall:

| Rows | Exact Match | Median Speedup | Mean Speedup | Baseline tok/s | Boosted tok/s | Median Accepted Draft Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 100% | 3.03x | 3.55x | 10.53 | 26.17 | 49.0 |

Per-fixture medians:

| Fixture | Workflow | Expected | Exact Match | Speedup | Accepted Draft Tokens |
|---|---|---:|---:|---:|---:|
| `real_readme_api` | real README continuation | positive | 100% | 5.68x | 64 |
| `real_core_code` | real code continuation | positive | 100% | 5.93x | 64 |
| `real_paper_method` | real paper continuation | positive | 100% | 8.54x | 64 |
| `json` | structured config | positive | 100% | 3.79x | 57 |
| `code` | code completion | positive | 100% | 3.72x | 55 |
| `policy` | policy quote | positive | 100% | 2.31x | 41 |
| `rag` | RAG answer | positive | 100% | 1.91x | 27 |
| `repo_quote` | repo quote | positive | 100% | 1.78x | 24 |
| `creative_open` | creative generation | negative | 100% | 1.09x | 0 |
| `short_answer` | short answer | negative | 100% | 1.01x | 0 |

Takeaways:

- Real local artifacts are the strongest v2 use case: README, core code, and paper-source continuation all accept the full 64-token draft window and exceed 5.6x speedup in strict mode.
- Strict stateless MLX mode is not the fastest raw runtime path, but it provides clean exactness evidence for longer generations.
- Cache-enabled diagnostic runs were faster in raw tok/s but showed boundary mismatches on longer synthetic quote fixtures, so cache-mode evidence should remain diagnostic until cache trajectory handling is improved further.

## MLX Strict Evidence V2, Three Independent Runs

Additional strict runs:

- `mlx_evidence_v2_strict_run2_20260706.json`
- `mlx_evidence_v2_strict_run3_20260706.json`
- `mlx_evidence_v2_strict_aggregate_20260706.json`

The aggregate combines the original strict v2 run with two fresh runs using different seeds.

| Runs | Rows | Exact Match | Median Speedup | Mean Speedup | P10 Speedup | P90 Speedup | Baseline tok/s | Boosted tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 90 | 100% | 3.00x | 3.56x | 1.05x | 6.93x | 10.61 | 25.66 |

Per-run medians:

| Run | Exact Match | Median Speedup | Mean Speedup | Baseline tok/s | Boosted tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 3.03x | 3.55x | 10.53 | 26.17 |
| 2 | 100% | 3.15x | 3.43x | 11.03 | 25.57 |
| 3 | 100% | 3.00x | 3.70x | 10.15 | 25.53 |

Aggregate per-fixture medians:

| Fixture | Exact Match | Median Speedup | P10-P90 Speedup | Accepted Draft Tokens |
|---|---:|---:|---:|---:|
| `real_paper_method` | 100% | 8.54x | 6.50x-9.57x | 64 |
| `real_core_code` | 100% | 6.61x | 5.22x-6.99x | 64 |
| `real_readme_api` | 100% | 5.65x | 4.23x-5.97x | 64 |
| `json` | 100% | 3.89x | 3.44x-4.11x | 57 |
| `code` | 100% | 3.71x | 3.57x-3.82x | 55 |
| `policy` | 100% | 2.48x | 2.29x-2.96x | 41 |
| `rag` | 100% | 1.66x | 1.55x-2.49x | 27 |
| `repo_quote` | 100% | 1.60x | 1.31x-1.86x | 24 |
| `creative_open` | 100% | 1.07x | 1.04x-1.19x | 0 |
| `short_answer` | 100% | 1.01x | 0.98x-1.03x | 0 |

Takeaways:

- The strict v2 median speedup is stable across independent runs: 3.03x, 3.15x, and 3.00x.
- Exactness held across all 90 rows.
- Real artifact continuations remain the strongest evidence: the README, core-code, and paper-source fixtures accepted the full 64-token draft budget in aggregate.
- Negative controls stayed close to neutral and accepted zero draft tokens, which supports the benchmark gate design.

## MLX Qwen3.5-9B Strict Smoke

Command:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --mlx-model mlx-community/Qwen3.5-9B-MLX-4bit \
  --source-mode context \
  --mlx-disable-cache \
  --fixtures policy,json \
  --repeat 1 \
  --max-new-tokens 16 \
  --output results/mlx_qwen35_9b_strict_smoke.json
```

Model: `mlx-community/Qwen3.5-9B-MLX-4bit`

Runner: MLX package adapter with prompt cache disabled, matching the strict evidence mode.

Generation: greedy, 16 new tokens, exact token-match checked against baseline output.

Artifact:

- `mlx_qwen35_9b_strict_smoke.json`: two-row larger-model smoke artifact.

| Rows | Exact Match | Median Speedup | Baseline tok/s | Boosted tok/s | Accepted Draft Tokens | Forward Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100% | 7.50x | 1.77 | 13.26 | 16.0 | 87.5% |

Per-fixture:

| Fixture | Exact Match | Speedup | Baseline tok/s | Boosted tok/s | Accepted Draft Tokens |
|---|---:|---:|---:|---:|---:|
| `policy` | 100% | 6.99x | 1.86 | 13.00 | 16 |
| `json` | 100% | 8.01x | 1.69 | 13.51 | 16 |

Takeaway:

- This is only a smoke test, but it supports the expected scaling behavior: when accepted draft spans are long, larger/slower target models benefit more because each avoided serial target step is more expensive.
