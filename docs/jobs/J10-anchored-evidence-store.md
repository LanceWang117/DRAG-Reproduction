---
id: J10
title: Anchored evidence store
status: todo
env: cpu
size: M
depends: [J02]
---
# J10 Anchored evidence store

**Goal.** Add `evidence_store = "anchored"`, an append-only store keyed by `doc_id`.
- The original question's passages are never evicted.
- Duplicates are removed, and passages get global labels `E{qid}-{rank}`.
- `evidence_delimiters` wraps passages in `<document id=…>` blocks and tells the model
  that they are data, not instructions.
- `rewrite_novelty_audit` flags entities and numbers invented by rewrites.

Also add an offline replay estimating the recall effect from stored passages.

**Why.**
- B7: Optimization deleted the original evidence on 53/200 questions (`dev_4016`,
  `dev_1245`).
- B8: 4.3% of slots held duplicate passages.
- B13: raw passages sit in system prompts with no boundary.

## Research questions (save to `docs/research/J10-anchored-evidence-store.md`)
1. Prompt-injection defences for RAG (delimiting retrieved text as data; instruction
   hierarchy): the current guidance and its limits.
2. Query-rewrite drift, and mitigations like "retrieve with the original and the
   rewrite" (ledger references: arXiv 2603.13301 and 2604.10745). Extend with at most 2
   more sources.
3. Lightweight entity and number extraction without spaCy in CI (regex heuristics):
   precision trade-offs.

## Scope
- **In:**
  - `evidence/store.py`;
  - prompt formatting with global labels and delimiters (in `prompts.py` or
    `evidence/format.py`);
  - `evidence/audit.py`;
  - `scripts/replay_evidence.py` (approximate, from stored passages);
  - tests.
- **Out:** the token budget and layout (J11); structured Opponent actions (J13).

## Files
- **Create:**
  - `improvements/drag_plus/evidence/{__init__,store,format,audit}.py`
  - `scripts/replay_evidence.py`
  - `improvements/tests/test_evidence_store.py`
  - `improvements/tests/test_replay_evidence.py`
- **Modify:** `query_stage.py`, `prompts.py`, `flags.py`, architecture.md §4 and §7,
  findings.md B7, B8 and B13.

## Design notes and edge cases
- **Optimization under the anchored store.** The new query's passages are added and the
  old non-original query is marked superseded. Its passages stay unless a future budget
  (J11) drops them. The original's passages always stay.
- **Ordering.** Iteration order is deterministic: by first-seen round, then rank. That
  is a precondition for prefix caching in J11.
- **Output contract.** The `QueryStage_QueryPool` output key must still exist for
  FlashRAG's saved data. Map the store back to `{query: [docs]}` for that key.
- **Replay.** The replay can only approximate. Naive's passages stand in for the deleted
  original retrieval, and that proxy is imperfect (B2). The script must say so in its
  output. It reports gold-string recall for the baseline pool versus the anchored union.
- **Audit.** A log-only signal; it never blocks a query in this job.

## Acceptance
- [ ] Under the anchored store, the original passages survive any Optimization or
  Expansion sequence (property-style test over generated action sequences).
- [ ] There are no duplicate `doc_id`s in formatted prompts, labels are unique and
  stable across rounds, and delimiters are present when the flag is on (tests).
- [ ] The replay prints recall for both policies on the 175 non-StrategyQA dev
  questions, with the proxy caveat. Numbers are recorded in the status log as [D].
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
