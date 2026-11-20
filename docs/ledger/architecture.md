# Implementation architecture

Read this before writing any code. A job that adds a module, flag or schema updates the
relevant table here in the same PR. The layout below is the plan. If a job needs a
different split, it changes the table in the same PR and explains why in its status log.

## 1. Where code goes

| Path | What | Who edits |
| --- | --- | --- |
| `src/original_drag/`, `src/reproduction_changes/`, `results/` | Frozen baseline and evidence | Nobody (CI guard from J00) |
| `improvements/drag_plus/` | The team's pipeline package | J02 onward |
| `improvements/tests/` | Unit and integration tests, fakes, the `flashrag` test stub | Every code job |
| `improvements/configs/` | Variant JSON files (one per experiment arm) | Code and E-jobs |
| `scripts/` | Runner, overlay builder, offline checks, code-limit checker | Jobs listed in the job file |
| `experiments/<JOB-ID>/` | Compact, committed experiment evidence (E-jobs) | E-jobs |
| `new-runs/` (ignored) | Raw run output | Runner |
| `docs/` | Ledger, jobs, prompts, research notes | Every job, own files only |

**Planned package layout** (each file stays within the code limits in §6):

```
improvements/drag_plus/
  flags.py            variant schema, defaults = baseline, validation (J02)
  state.py            per-question state record (J02; extended by J15)
  adapters.py         generator / retriever / prompt-template protocols (J02)
  pipeline.py         orchestration only; delegates to stages (J02)
  query_stage.py      query debate (J02; J07, J13, J14 extend)
  answer_stage.py     response debate and moderator (J02; J06, J08, J18 extend)
  prompts.py          all prompt text in one place (J02; J10, J11, J18 extend)
  evidence/           store.py, budget.py, layout.py (J10, J11)
  judge/              logprob.py, actions.py (J12, J13)
  gates/              escalation.py, router.py, conformal.py (J14, J20, J21)
  retrieval/          cache.py, hybrid.py (J04, J17)
  scheduling/         barrier.py, refill.py (J15, J16)
  instrument/         timing.py, logs.py, manifest.py (J03)
  analysis/           extract.py, stats.py, report.py (J01)
```

## 2. Hard constraints

1. **`drag_plus` imports without FlashRAG, vLLM, torch or FAISS.** Those objects are
   injected through `adapters.py` by the runner. CI does not install them.
2. **Defaults equal the baseline.** With every flag at its default, the pipeline must
   reproduce v1.0.1 exactly: test **T0**, the golden transcript (§5).
3. **One flag, one behaviour.** Every flag is independently switchable, validated (an
   unknown flag is an error), and recorded in the run manifest.
4. **No module-level mutable state.** Per-question data lives in the state record (§4);
   nothing persists across questions on `self`.
5. **No network, GPU or wall-clock dependence in tests.** Inject time sources and
   tokenizers.
6. **Frozen paths stay untouched.** Baseline defects (findings.md, minor defects) are
   fixed only in `drag_plus`, never in the frozen copy.

## 3. Interfaces the runner injects

| Protocol | Methods | Real implementation | Test fake |
| --- | --- | --- | --- |
| Generator | `generate(prompts: list[str], **params) -> list[str]`; optional `generate_with_scores(...)` returning text and token log-probs | FlashRAG `VLLMGenerator`; per-call `**params` are merged into its `SamplingParams` | Scripted outputs; records every call (prompts, params, batch size) |
| Retriever | `search(query) -> list[doc]`; `batch_search(queries) -> list[list[doc]]`; a doc is `{id, contents}` | FlashRAG `DenseRetriever` | Dict from query to docs; records calls |
| Prompt template | `get_string(messages) -> str` | FlashRAG `PromptTemplate` (chat template) | Identity (returns messages), as in the existing Judge tests |
| Token counter | `count(text) -> int` | Llama-3.1 tokenizer | Whitespace count |
| Clock | `now() -> float` | `time.monotonic` | Deterministic counter |

**FlashRAG facts** (from `main`; re-check the pinned `flashrag-dev==0.1.4.dev20250111`
on the Pod) [P]:
- `VLLMGenerator` builds `vllm.LLM` with `gpu_memory_utilization` from config (default
  0.85) and **without** `enable_prefix_caching`, so that flag needs a custom
  construction.
- It does not truncate prompts.
- `DenseRetriever` with `faiss_gpu` clones the index with `useFloat16 = True` and
  `shard = True`.

## 4. Per-question state record

One record per question, owned by the pipeline. Stages read and write only this record.

| Field | Content |
| --- | --- |
| `key` | `(dataset, split, id)` |
| `question` | Question text |
| `queries` | Ordered list of `{qid, text, source: original \| optimize \| expand, round}` |
| `evidence` | Map `doc_id → {contents, first_qid, rank, score}`; the original question's docs are anchored (J10) |
| `messages` | Message histories per stage and role |
| `stage`, `round`, `status` | `query \| answer \| done`; round counter; `active \| done \| failed` |
| `decisions` | Judge verdicts, escalation decisions, router route, with scores |
| `calls` | Timing and log rows (§5) for this question |
| `pred`, `raw_pred` | Final answer under the unified extractor, and the raw text |

## 5. Logs and manifests

| File (per run, in `new-runs/<run>/`) | One row per | Required fields |
| --- | --- | --- |
| `stage-timings.jsonl` | LLM or retrieval call | run_id, variant, dataset, split, id, stage (`retrieval` \| `query_agent` \| `query_judge` \| `query_judge_retry` \| `answer_agent` \| `answer_moderator` \| `rerank` \| `gate`), round, role, batch_size, prompt_tokens, output_tokens, finish_reason, cache_hit, t_start, elapsed_ms |
| `transcripts.jsonl` | Question | Every prompt and raw output in order, plus a sha256 of each; `pred` under both extractors |
| `retrieval-cache.jsonl` | Unique query | key = (index checksum, encoder, top-k, normalised query); passage IDs; scores |
| `run-manifest.json` | Run | Commit, variant JSON, model and index identifiers, GPU, vLLM/FlashRAG/FAISS versions, thread settings, seeds, start, end, exit status |

**Tests.**
- **T0 golden transcript (J02):** with default flags, `drag_plus` produces the same
  prompt sequence, call order and final `pred` as `src/reproduction_changes/model/drag.py`
  on scripted fake outputs. To import the frozen file without FlashRAG, tests put
  `improvements/tests/stubs/` (a minimal fake `flashrag` package) on `sys.path`. Tests
  only, never the runtime.
- **T-flag (each job):** the exact call-count or structure change the job file specifies,
  with all other behaviour unchanged (T0 still passes).
- **No skips:** tests under `improvements/tests/` must never skip. CI fails if any do.

## 6. Code limits and CI (J00)

| Rule | Limit | Enforcement |
| --- | --- | --- |
| Function length | ≤ 40 lines | `scripts/check_code_limits.py` |
| File length | ≤ 250 lines | same |
| Cyclomatic complexity | ≤ 10 | same (or ruff C901) |
| Existing over-limit files (e.g. `scripts/run_evaluation.py`) | No growth: may shrink, never grow | Ratchet allowlist with recorded current sizes |
| Frozen paths | No diff | CI guard on PRs |
| Skipped tests in `improvements/tests/` | 0 | CI |
| Committed file size | ≤ 1 MB unless the job file allows it | CI |
| Gate runtime and memory | ≤ 5 min total, ≤ 10 s per test, ≤ 2 GB peak RSS, on the team laptop (8 threads, 23 GiB RAM shared with a desktop) | `scripts/gate.sh` prints its duration; the J00 status log records `/usr/bin/time -v` |
| Python version for the gate | 3.11 (CI, `.venv` from pyenv, or Docker `python:3.11-slim`) | `gate.sh` refuses any other version |

The limits come from the team's guardrails (40-line functions, complexity under 10,
250-line files). CI runs on every PR and on every push to `main`, without GPU, FlashRAG
or vLLM. "Main builds" means this CI is green on the latest `main` commit.

## 7. Feature flags (planned)

Defaults reproduce the baseline. The job that implements a flag owns its row.

| Flag | Default | Job |
| --- | --- | --- |
| `answer_extractor` (`native` \| `unified`) | `native` | J01 |
| `profile_calls` | `false` | J03 |
| `retrieval_cache` (path or null) | `null` | J04 |
| `moderator_final_only`, `batch_agents_in_round` | `false`, `false` | J06 |
| `skip_duplicate_query`, `zero_novelty_exit` | `false`, `false` | J07 |
| `response_rounds`, `agent_max_tokens`, `moderator_max_tokens`, `concise_agents` | `3`, `256`, `256`, `false` | J08 |
| `faiss_gpu`, `gpu_memory_utilization`, `omp_threads` | `false`, `0.85`, `12` | J09 |
| `evidence_store` (`dict` \| `anchored`), `evidence_delimiters`, `rewrite_novelty_audit` | `dict`, `false`, `false` | J10 |
| `context_token_budget`, `prompt_layout` (`role_first` \| `evidence_first`), `enable_prefix_caching` | `null`, `role_first`, `false` | J11 |
| `judge_scoring` (`generative` \| `logprob`), `judge_order_balance`, `judge_tau` | `generative`, `false`, `0.0` | J12 |
| `opponent_action_schema`, `judge_sees_refined_query` | `false`, `false` | J13 |
| `escalation_gate` (`none` \| `risk`), `escalation_tau`, `lazy_query_opponent` | `none`, null, `false` | J14 |
| `question_batch_size` | `1` | J15 |
| `refill_window` | `false` | J16 |
| `retriever` (`e5` \| `hybrid`), `rerank_k`, `per_title_cap`, `score_floor` | `e5`, null, null, null | J17 |
| `opponent_mode` (`closed_book` \| `grounded`), `verdict_sequestration`, `aggregator` (`moderator` \| `vote` \| `self_consistency`) | `closed_book`, `false`, `moderator` | J18 |
| `faithfulness_scorer`, `aggregator = weighted` | null | J19 |
| `router` (null \| path to a fitted router) | null | J20 |
| `conformal_alpha` | null | J21 |
| `decompose` | `false` | J22 |
| `judge_adapter` (LoRA path) | null | J23 |
| `quantization`, `speculative` | `none`, `none` | E03 variants (config only) |
