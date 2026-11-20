---
id: J07
title: Duplicate-query guard and zero-novelty exit
status: todo
env: cpu
size: S
depends: [J02]
---
# J07 Duplicate-query guard and zero-novelty exit

**Goal.** Add two flags, both small method changes:
- `skip_duplicate_query`: skip the search, and end the query stage, when a proposed
  query is effectively already in the pool.
- `zero_novelty_exit`: end the query stage when a search returns no passage ID that the
  pool does not already have.

**Why.** B15: 8 of 76 changed pools retrieved no new passages, yet the debate kept
going. Example: `PopQA/test_8583`, where the third query ("was" → "is") retrieved
exactly the second query's passages.

## Research questions (save to `docs/research/J07-duplicate-guard-novelty-exit.md`)
1. Query-normalisation practice for duplicate detection in IR, specifically stop-word
   and auxiliary-verb handling. Keep the rule simple and documented.
2. Do fixed-point or no-new-evidence stopping rules for iterative retrieval appear in
   the literature (for example in IRCoT, Self-Ask, or adaptive retrieval)? One or two
   sources are enough.

## Scope
- **In:** a normalisation helper, both flags in `query_stage.py`, fixtures built from
  the 8 recorded zero-novelty cases, tests.
- **Out:** changes to how the Opponent proposes queries (J13).

## Files
- **Modify:** `improvements/drag_plus/query_stage.py`, `flags.py`, architecture.md §7,
  findings.md B15.
- **Create:**
  - `improvements/drag_plus/query_norm.py`
  - `improvements/tests/test_novelty_exit.py`
  - `improvements/tests/fixtures/zero_novelty.json`, built by a small committed script
    from `results/retrieval-200.jsonl`, holding IDs only

## Design notes and edge cases
- **Normalisation.** Lowercase, Unicode NFC, strip punctuation, collapse whitespace, and
  drop a fixed small stop list. The list covers articles, auxiliaries (is / was / are /
  were / be / been) and "of", "the" and "for". Duplicate means equal token sets. Write
  the list in code and in the research note.
- **What "already in the pool" means.** Under the baseline `dict` store, an Optimization
  first deletes the old query's passages. The novelty check must compare with the IDs
  that were present **before** the action, including the deleted ones. Otherwise
  deletion masks duplication. Under J10's anchored store this happens naturally.
- **Exit semantics.** Exiting returns the current pool. It must not append the duplicate
  passages, and it must record a decision `query_exit: duplicate | zero_novelty` in the
  state record.
- **Combinations.** Both flags work with `max_query_debate_rounds` ∈ {1, 2, 3} and with
  J06's batching.

## Acceptance
- [ ] For each of the 8 recorded zero-novelty cases, the exit triggers at the right
  step (fixture test).
- [ ] A query with novel IDs never triggers an exit (test).
- [ ] `test_8583`-style query pairs differing only in auxiliary verbs count as
  duplicates; queries that differ in entities do not (tests).
- [ ] T0 still passes with the flags off.
- [ ] findings.md B15: "Addressed by J07 (validation: E02a)".

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
