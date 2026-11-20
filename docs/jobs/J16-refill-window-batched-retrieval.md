---
id: J16
title: Refill window and batched retrieval
status: todo
env: cpu
size: M
depends: [J15]
---
# J16 Refill window and batched retrieval

**Goal.** Add `refill_window`. It keeps B slots full by admitting pending questions as
others finish, and it gathers every ready query from every active question into one
`batch_search` call. Add a throughput simulator (fake backend with a cost model) to
check occupancy offline.

**Why.** With a fixed lockstep batch, occupancy collapses: up to 62% of questions leave
the query stage after its first round (B16). Batched retrieval amortises the CPU
FlatIP scan (B5).

## Research questions (save to `docs/research/J16-refill-window-batched-retrieval.md`)
1. Continuous batching and refill scheduling, as in vLLM or Orca-style iteration-level
   scheduling, applied one level up (pipeline requests).
2. FAISS batched-search efficiency (one GEMM pass) and sensible batch caps.

## Scope
- **In:**
  - `scheduling/refill.py`;
  - batched-retrieval gathering;
  - `scripts/simulate_throughput.py` (cost model: per-call cost = a + b × batch size;
    reports occupancy and modelled questions/hour);
  - tests.
- **Out:** GPU measurement (E03).

## Files
- **Create:**
  - `improvements/drag_plus/scheduling/refill.py`
  - `scripts/simulate_throughput.py`
  - `improvements/tests/test_refill.py`
- **Modify:** `scheduling/barrier.py`, `runner/run.py`, `flags.py`, architecture.md §7,
  findings.md B4.

## Design notes and edge cases
- **Determinism.** Refill must keep per-question transcripts identical to B = 1 under a
  deterministic fake. Only the composition of each batch changes.
- **Mixed stages.** Admission order is manifest order. At any step, a batch may mix
  stages and roles. If that complicates the prompt-template interface, group by
  (stage, role) per step. Either way, document the choice.
- **Batch caps.** Cap concurrent prompts with `question_batch_size` and cap the size of
  each retrieval batch. Both are configurable.
- **Resume.** Under refill, admitted-but-unfinished questions are re-run from scratch on
  resume; finished ones are skipped.

## Acceptance
- [ ] Per-question transcripts equal B = 1's with refill on, for B ∈ {2, 8} (tests).
- [ ] In the simulator, with a skewed finish distribution, refill keeps mean occupancy
  ≥ 90% of B until the queue empties, against lockstep's lower figure. Numbers go in the
  status log as simulation results, not measurements.
- [ ] Batched retrieval issues one `batch_search` per step with all pending queries
  (test).
- [ ] The resume test passes under refill.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
