---
id: J18
title: Grounded answer stage and vote baselines
status: todo
env: cpu
size: M
depends: [J02, J01]
---
# J18 Grounded answer stage and vote baselines

**Goal.** Add three flags:
- `opponent_mode = "grounded"`: a Conflict Auditor that sees the evidence pool replaces
  the closed-book Opponent. The Proponent becomes an Evidence Synthesizer that cites
  `[E…]` labels.
- `verdict_sequestration`: when one agent's answer is relayed to the other, the
  reasoning is kept and the "The answer is: X" line is withheld.
- `aggregator` ∈ {`moderator`, `vote`, `self_consistency`}: the required non-debate
  baselines.

**Why.**
- B11: DRAG switched to wrong parametric answers on identical evidence (`dev_3381`,
  `dev_5657`, `test_12352`).
- Debate gains often come from ensembling (Choi et al., 2025), and agents conform to
  each other (Free-MAD; "Peacemaker or Troublemaker"; Dai, 2026).

## Research questions (save to `docs/research/J18-grounded-answer-vote-baselines.md`)
1. Verdict sequestration and calibrated debate (Dai, 2026, "Prevent"): what exactly is
   hidden, and the reported effect.
2. Self-consistency at matched compute: how to match generated tokens to DRAG's budget
   fairly.
3. Prompting for conflicts between the model's own knowledge and the context ("DRAGged
   into Conflicts"; Astute RAG): prompt patterns that help 8B models.

## Scope
- **In:**
  - grounded role prompts in `prompts.py`;
  - sequestration (strip the answer line from relayed messages);
  - `aggregators.py` (`vote`, `self_consistency`; plurality over `E(·)`-normalised
    answers, with a documented tie-break);
  - tests.
- **Out:** the faithfulness scorer and weighted aggregation (J19).

## Files
- **Create:**
  - `improvements/drag_plus/aggregators.py`
  - `improvements/tests/test_grounded_answer.py`
  - `improvements/tests/test_aggregators.py`
- **Modify:** `answer_stage.py`, `prompts.py`, `flags.py`, architecture.md §7,
  findings.md B11.

## Design notes and edge cases
- **Sequestration.** It removes every occurrence of the marker line, matched
  case-insensitively, and keeps the rest. If nothing would remain, it relays
  "[reasoning withheld]" rather than an empty message.
- **Self-consistency.**
  - `m` samples at the configured temperature: a separate generation-parameter set,
    never mutating the shared configuration;
  - it records the token count so matching can be reported;
  - `do_sample` must be true only for these calls.
- **Vote tie-break.** Prefer the answer supported by the evidence (containment),
  otherwise the Proponent's answer. Document this and test it.
- **Determinism in tests.** Use scripted samples.

## Acceptance
- [ ] Grounded mode: the Opponent prompt contains the evidence block; the closed-book
  text is absent (test).
- [ ] The sequestration edge cases are tested.
- [ ] `vote` and `self_consistency` produce the expected answers on scripted cases, and
  their token accounting is logged (tests).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
