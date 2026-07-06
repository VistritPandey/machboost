# MachBoost Evidence Runs

This directory stores public benchmark artifacts for the local-context speculative decoding prototype.

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
