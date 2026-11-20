---
id: J04
title: Single-path retrieval cache
status: todo
env: cpu
size: S
depends: [J03]
---
# J04 Single-path retrieval cache

**Goal.** Build a caching retriever proxy. Its `search` and `batch_search` both resolve
through **one** code path, and it persists results to `retrieval-cache.jsonl`. The
variant runner hands it to both arms: DRAG through the adapter, and Naive through
FlashRAG's `SequentialPipeline`.

**Why.** B2: 32 of 124 unchanged pools got different passages from Naive for the same
query. Paired comparisons need identical retrieval for identical queries.

## Research questions (save to `docs/research/J04-retrieval-cache.md`)
1. At the **pinned** FlashRAG version, how do `search`, `batch_search` and FlashRAG's own
   `cache_manager` / `use_retrieval_cache` work? On `main`, both appear to route misses
   through `_batch_search_with_rerank`. If the pinned version differs, that may explain
   B2. Report what you find and mark the root cause [C] or [IC].
2. The signature and return shape of `search` and `batch_search`, including
   `return_score`.
3. Is E5-base-v2 uncased? This decides whether lowercasing is safe in the cache key.
   Default: do not lowercase; normalise whitespace and Unicode (NFC) only.

## Scope
- **In:** `retrieval/cache.py` (the proxy plus the persistent store), runner
  integration, tests.
- **Out:** any change to ranking.

## Files
- **Create:**
  - `improvements/drag_plus/retrieval/{__init__,cache}.py`
  - `improvements/tests/test_retrieval_cache.py`
- **Modify:**
  - `improvements/drag_plus/runner/backends.py`
  - `improvements/drag_plus/flags.py` (`retrieval_cache`)
  - architecture.md §7

## Design notes and edge cases
- **Cache key:** `(index identifier or checksum, encoder name, top_k, normalised
  query)`. A different index or top-k must never hit a stale entry.
- **Misses.** On a miss, call the underlying `batch_search` once for all missing queries
  in the request. Preserve request order, and deduplicate repeated queries within one
  batch.
- **Copies.** Return deep copies; callers must not be able to mutate cached docs.
- **Scores.** Store them, because J20 needs retrieval margins. Support both the scored
  and unscored call shapes.
- **Persistence.** JSONL with one line per query, written only after a successful
  search. A corrupt last line on load is skipped with a warning.

## Acceptance
- [ ] For the same query, `search` and `batch_search` return identical IDs, scores and
  order (test with a fake retriever whose batch and single paths deliberately differ).
- [ ] The underlying retriever is called exactly once per unique key across a run
  (test).
- [ ] The persistence round trip works, and a corrupt-line recovery test passes.
- [ ] Both runner arms receive the same proxy instance (test).
- [ ] findings.md B2: "Addressed by J04; root cause: <finding> [C|IC]".

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
