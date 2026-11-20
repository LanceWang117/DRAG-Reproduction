---
id: J17
title: Hybrid retrieval code
status: todo
env: cpu
size: M
depends: [J04]
---
# J17 Hybrid retrieval code

**Goal.** Add `retriever = "hybrid"`. It combines E5 dense top-K with BM25 top-K, fuses
them by weighted reciprocal rank fusion (RRF) with a per-retriever `score_floor`, can
rerank the fused list with a cross-encoder (`rerank_k`), and caps passages per article
(`per_title_cap`). Add a retrieval-only evaluation script. Everything runs offline
behind adapters; E04 builds the real index.

**Why.** B17: 69% of wrong answers had no gold in context, and gold containment is
about 58% for both methods. Retrieval is the largest shared ceiling.

## Research questions (save to `docs/research/J17-hybrid-retrieval.md`)
1. RRF (Cormack et al., 2009, currently unverified in the ledger: verify it): the
   formula and the usual k; weighted variants.
2. `bm25s` 0.2.0 (installed): index build, memory mapping, and query API for about 21M
   passages; expected RAM and disk.
3. Cross-encoder rerankers that fit next to vLLM on an A100 (for example
   `bge-reranker-base` or `bge-reranker-v2-m3`): licences, size and latency. Also the
   large-K degradation found in "Drowning in Documents".

## Scope
- **In:**
  - `retrieval/hybrid.py` (fusion, floor, cap);
  - `retrieval/bm25.py` (adapter interface; lazy import of `bm25s`);
  - `retrieval/rerank.py` (interface; lazy import of the cross-encoder);
  - `scripts/eval_retrieval.py` (Containment@k, MRR, nDCG@k, unique-title ratio;
    retrieval latency from the timing logs);
  - tests with fake indexes.
- **Out:** building the index (E04); changing the default retriever.

## Files
- **Create:**
  - `improvements/drag_plus/retrieval/{hybrid,bm25,rerank}.py`
  - `scripts/eval_retrieval.py`
  - `improvements/tests/test_hybrid_retrieval.py`
  - `improvements/tests/test_eval_retrieval.py`
- **Modify:** `runner/backends.py`, `retrieval/cache.py` (the key includes the retriever
  configuration), `flags.py`, architecture.md §7, findings.md B17.

## Design notes and edge cases
- **Cache key.** The J04 cache key must include the retriever type, k values, weights,
  floor, cap and reranker ID. Otherwise hybrid runs would hit dense entries.
- **Ties and stability.** Deterministic tie-breaking (by doc_id). When one retriever
  returns fewer results than k, fuse what exists.
- **Title cap.** Extract the title from the first line of `contents`, as
  `_format_reference` does in the baseline.
- **Reranker.** Optional: when `rerank_k` is null, there is no reranker import at all.
- **Metrics.** `eval_retrieval.py` must compute metrics from candidate lists alone, so
  it can score stored baseline passages, with gold strings normalised by J01's scorer.

## Acceptance
- [ ] RRF, floor and cap produce the expected rankings on hand-built fixtures (tests).
- [ ] `eval_retrieval.py` reproduces gold containment 114–115/200 from the stored
  `results/retrieval-200.jsonl` passages (Naive and DRAG) [D] (test).
- [ ] No heavy import unless the corresponding flag is set (test).
- [ ] Cache-key separation between dense and hybrid configurations (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
