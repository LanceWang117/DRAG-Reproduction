---
id: J22
title: Multi-hop decomposition route
status: todo
env: cpu
size: L
depends: [J10, J12]
---
# J22 Multi-hop decomposition route

**Goal.** Add `decompose`: a route for multi-hop questions.
- A planner prompt produces a typed plan of at most 3 sub-questions with placeholders,
  for example "q1: Who was Junia Tertia's husband? → #E1" and "q2: Where did #E1 die?".
- An external ledger records `(sub_q, entity, confidence, doc_ids)`.
- Each hop is retrieved and answered, and the next hop starts only if the bound entity
  appears in that hop's passages.
- With a low margin, keep two candidate bindings (beam of 2).
- Finish with a composed answer.

**Why.**
- B7: even correct parametric hop resolution loses the evidence (`dev_1245`).
- B17: retrieval misses on multi-hop questions.
- DRAG's gains concentrate on 2Wiki and HotpotQA, so this route targets them.

## Research questions (save to `docs/research/J22-multihop-decomposition.md`)
1. Decomposition methods (Self-Ask, IRCoT, plan-and-solve) and their failure modes:
   hop-1 errors propagating, and drift from hop 3 (arXiv 2604.10745).
2. FlashRAG's `IRCoT` and `SelfAsk` pipelines (`src/original_drag/model/baseline.py`):
   how to run them as comparison arms through the variant runner.

## Scope
- **In:**
  - `decompose/plan.py` (prompt, parser, validator; depth ≤ 3; acyclic);
  - `decompose/execute.py` (per-hop retrieval and answer, grounding check, beam of 2);
  - `decompose/compose.py`;
  - route integration with J20;
  - runner support for the IRCoT and Self-Ask arms (baseline comparison only);
  - tests.
- **Out:** training a planner.

## Files
- **Create:**
  - `improvements/drag_plus/decompose/{__init__,plan,execute,compose}.py`
  - `improvements/tests/test_decompose_plan.py`
  - `improvements/tests/test_decompose_execute.py`
- **Modify:** `pipeline.py`, `runner/backends.py`, `flags.py`, architecture.md §7,
  findings.md B7 and B17.

## Design notes and edge cases
- **Invalid plans.** Cycles, a depth above 3, unknown placeholders, or an empty plan all
  fall back to the normal DRAG route. Log the fallback.
- **Grounding failure.** If a hop's bound entity is not found in its passages, try one
  rewrite; then fall back. Never compose from an ungrounded binding without flagging it.
- **Beam.** Bounded at 2. The total call budget per question is capped and logged.
- **Anchoring.** All hop passages go into the anchored store (J10); the original
  question's passages stay.

## Acceptance
- [ ] Parser and validator tests cover every invalid-plan type.
- [ ] Scripted execution: grounded success, grounding failure with fallback, beam
  selection, and the call-cap enforcement (tests).
- [ ] The IRCoT and Self-Ask arms are constructible through the runner with the fake
  backend (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
