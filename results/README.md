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
