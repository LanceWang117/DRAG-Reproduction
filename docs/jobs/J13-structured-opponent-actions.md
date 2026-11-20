---
id: J13
title: Structured Opponent actions
status: todo
env: cpu
size: M
depends: [J10, J12]
---
# J13 Structured Opponent actions

**Goal.** Add two flags:
- `opponent_action_schema`: the query-stage Opponent ends its reasoning with a JSON
  action `{action ∈ {optimize, expand, no_op}, source_query_id, new_query, reason}`.
  The action is parsed, validated and committed transactionally, with stable query IDs
  replacing the Levenshtein match.
- `judge_sees_refined_query`: an A/B switch for whether the Judge sees the original
  question or the latest refined query.

**Why.**
- B7: free-text parsing plus a Levenshtein "closest key" can delete the wrong query.
- B14: the Judge never sees the refined query, and the paper doesn't say whether it
  should.

## Research questions (save to `docs/research/J13-structured-opponent-actions.md`)
1. Reasoning-then-constrained-slot output versus fully constrained output ("Let Me Speak
   Freely?", CRANE; the .txt rebuttal). Which applies to an 8B model producing a short
   action?
2. Is vLLM 0.6.6 guided decoding (xgrammar 0.1.10 is pinned) safe to use? Ledger issues
   #12005, #12122, #11289. Default design: prompt the format plus a strict parser; no
   grammar engine.
3. Validation rules for rewritten queries (entity preservation, length): practitioner
   guidance.

## Scope
- **In:** `judge/actions.py` (schema, parser, validator, transactional apply), Opponent
  prompt text for schema mode, the visibility flag, tests.
- **Out:** guided decoding (optional later variant; not in this job).

## Files
- **Create:**
  - `improvements/drag_plus/judge/actions.py`
  - `improvements/tests/test_opponent_actions.py`
- **Modify:** `query_stage.py`, `prompts.py`, `evidence/store.py` (query IDs),
  `flags.py`, architecture.md §7, findings.md B7 and B14.

## Design notes and edge cases
- **Parser.** Take the **last** JSON object in the output. It must tolerate code fences
  and trailing text, and reject:
  - multiple actions;
  - unknown fields;
  - a wrong type;
  - an empty `new_query`.
- **Validation:**
  - `new_query` is 3–30 tokens;
  - it is not a duplicate (reuse J07's normaliser);
  - `optimize` needs an existing `source_query_id`;
  - entities from the J10 audit must not be dropped from the source query without a
    reason;
  - `no_op` is always valid.
- **Transactions.** Build the candidate pool, validate, then commit. On any failure,
  keep the pool unchanged and log `{raw, parsed, error}`.
- **Invalid actions.** An invalid action counts as `no_op` for control flow. Record it
  in the decisions so E02b can report the invalid-action rate.

## Acceptance
- [ ] Parser and validator tests cover every rejection reason and the code-fence and
  trailing-text cases.
- [ ] A transactional failure leaves the pool identical (test).
- [ ] Optimize with a stable ID never deletes the wrong query (a regression test built
  from a Levenshtein near-tie that the baseline gets wrong).
- [ ] The visibility flag changes only the Judge prompt's query line (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
