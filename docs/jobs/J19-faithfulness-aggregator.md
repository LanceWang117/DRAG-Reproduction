---
id: J19
title: Faithfulness signal and weighted aggregator
status: todo
env: cpu
size: M
depends: [J18]
---
# J19 Faithfulness signal and weighted aggregator

**Goal.** Add a faithfulness-scorer interface (`faithfulness_scorer`) with two
implementations:
- a default CI-testable heuristic: does a cited passage contain the answer string;
- an optional MiniCheck adapter, imported lazily.

Add `aggregator = "weighted"`: `a* = argmax_a [w₁·conf(a) + w₂·s_faith(a) + w₃·agr(a)]`,
with weights loaded from JSON. Add a fitting script for the weights.

**Why.** B11. Fact-checking a candidate against the evidence is an exogenous signal.
Agreement between agents is not (Dai, 2026). NLI-style checkers have known failure
modes, so the score is soft, never a veto.

## Research questions (save to `docs/research/J19-faithfulness-aggregator.md`)
1. MiniCheck (Tang et al., 2024): the model sizes, licence, input limits and expected
   latency. How to use it per claim and per passage.
2. Two failure modes of NLI checkers: "blind faithfulness" to a misleading passage, and
   penalising correct commonsense answers. Mitigations reported in the literature.

## Scope
- **In:**
  - `grounding/scorer.py` (the interface and the heuristic);
  - `grounding/minicheck.py` (lazy optional adapter; not used in CI);
  - the `weighted` option in `aggregators.py`;
  - `scripts/fit_aggregator.py`;
  - tests.
- **Out:** downloading MiniCheck weights in CI.

## Files
- **Create:**
  - `improvements/drag_plus/grounding/{__init__,scorer,minicheck}.py`
  - `scripts/fit_aggregator.py`
  - `improvements/tests/test_faithfulness.py`
  - `improvements/tests/test_weighted_aggregator.py`
- **Modify:** `aggregators.py`, `gates/features.py` (connect evidence support), `flags.py`,
  architecture.md §7.

## Design notes and edge cases
- **Heuristic score.** The answer, normalised by J01's scorer, is matched against
  passages. For yes/no answers the heuristic returns "unknown" (0.5), never 1: yes/no
  strings appear in every passage.
- **Weights file.** It records the training split and manifest hash. The loader refuses
  `test`.
- **MiniCheck adapter.** Import errors become a clear configuration error. There is no
  silent fallback when the user asked for MiniCheck.

## Acceptance
- [ ] Heuristic tests, including the yes/no case.
- [ ] The weighted aggregator picks the expected candidate on scripted inputs; the
  weights loader refuses test-split weights (tests).
- [ ] The fit script recovers known weights on synthetic data (test).
- [ ] No MiniCheck import unless configured (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
