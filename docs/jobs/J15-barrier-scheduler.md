---
id: J15
title: Per-question state and barrier scheduler
status: todo
env: cpu
size: M
depends: [J06, J04]
---
# J15 Per-question state and barrier scheduler

**Goal.** Restructure `drag_plus` so each stage step turns a question's state into
**requests** (prompts or queries) and later consumes their results. Add a barrier
scheduler, `question_batch_size = B`, that groups the requests of B questions at the
same (stage, round, role) into one `generate` call and one `batch_search` call.

**Why.** B4: questions run one at a time, at batch size 1. Upstream `mad.py:34-90`
already batches across questions; this is the throughput foundation for J16 and E03.

## Research questions (save to `docs/research/J15-barrier-scheduler.md`)
1. Request/response ("step function") designs for batched LLM pipelines; how batched
   agent frameworks avoid cross-request state leakage.
2. vLLM offline-batch behaviour for mixed prompt lengths (continuous batching inside
   one `generate` call).

## Scope
- **In:**
  - `scheduling/barrier.py`;
  - stage refactor into request and consume steps;
  - per-question error isolation;
  - checkpoint flush per finished question;
  - tests.
- **Out:** the refill window (J16); any behaviour change for B = 1.

## Files
- **Create:**
  - `improvements/drag_plus/scheduling/{__init__,barrier}.py`
  - `improvements/tests/test_barrier_scheduler.py`
- **Modify:** `pipeline.py`, `query_stage.py`, `answer_stage.py`, `state.py`,
  `runner/run.py`, `flags.py`, architecture.md §4 and §7.

## Design notes and edge cases
- **Equivalence.** B = 1 must produce exactly the T0 transcripts. B > 1 must produce,
  per question, the same transcript as B = 1 under a deterministic fake generator.
- **Ragged finishes.** Questions finish at different rounds. Finished questions leave
  the barrier, and the remaining ones continue.
- **Error isolation.** An exception in one question marks it `failed` (recorded with the
  traceback summary) without stopping the others. The run's exit status reports
  failures.
- **No shared mutable objects.** Test identity: no two states share a list or dict
  object.
- **Output order.** Batched calls return results in request order. Assert the counts
  match.

## Acceptance
- [ ] B = 1 passes T0. For B ∈ {2, 4, 7}, per-question transcripts equal B = 1's (tests;
  B = 7 checks a non-power-of-two remainder).
- [ ] The number of `generate` calls equals the number of barriers, not questions ×
  calls (test).
- [ ] The error-isolation test passes; the shared-reference test passes.
- [ ] Each checkpoint line is written when its question finishes (test).

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
