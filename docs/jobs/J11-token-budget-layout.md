---
id: J11
title: Token budget and evidence-first layout
status: todo
env: cpu
size: M
depends: [J10, J03]
---
# J11 Token budget and evidence-first layout

**Goal.** Add three flags:
- `context_token_budget`: select passages by relevance per token, with an MMR-style
  redundancy penalty and at most 2 chunks per article, under a token budget.
- `prompt_layout = "evidence_first"`: all three roles in a round share one long prompt
  prefix (header + question + evidence) before their role instructions.
- `enable_prefix_caching`: plumb vLLM prefix caching into backend construction.

**Why.**
- B8: there is no budget, and FlashRAG never truncates (`max_model_len = 4096`).
- B9: role-first prompts defeat KV-cache reuse.

## Research questions (save to `docs/research/J11-token-budget-layout.md`)
1. vLLM 0.6.6 automatic prefix caching: how it is enabled, block granularity, and known
   output-difference issues (ledger: #5543, #7845, PR #65962, #63743).
2. How the Llama-3.1 chat template orders system and user messages, and where the
   shared prefix can live without changing the role semantics.
3. How the pinned FlashRAG `VLLMGenerator` constructs `vllm.LLM`, and the cleanest way
   to pass `enable_prefix_caching` (subclass or construct directly). Only the backend
   builder may know about this.

## Scope
- **In:** `evidence/budget.py`, `evidence/layout.py`, backend support for prefix
  caching, tests.
- **Out:** changing what evidence is retrieved.

## Files
- **Create:**
  - `improvements/drag_plus/evidence/{budget,layout}.py`
  - `improvements/tests/test_budget_layout.py`
- **Modify:** `prompts.py`, `runner/backends.py`, `flags.py`, architecture.md §7,
  findings.md B8 and B9.

## Design notes and edge cases
- **Token counting.** Counting uses the injected token counter: the Llama tokenizer in
  real runs, whitespace in tests.
- **What the budget covers.** Header, question, evidence and history together. It drops
  the lowest-utility passages first and never drops the system instructions, the
  question or any anchored original passage. If the anchored passages alone exceed the
  budget, truncate the lowest-rank anchored passage text **with a logged warning**.
- **Shared prefix.** The prefix must be byte-identical across roles within a round.
  Evidence is append-only across rounds, so round `r+1`'s prefix extends round `r`'s.
- **Prefix caching.** Default off; it is enabled only in E-job variants that pass the
  A/A floor.

## Acceptance
- [ ] Token counts never exceed the budget, and anchored passages survive (property
  test).
- [ ] With the evidence-first layout, the three role prompts share an identical prefix
  that ends after the evidence block (test). Round `r`'s prefix is a prefix of round
  `r+1`'s (test).
- [ ] The backend passes `enable_prefix_caching=True` only when the flag is on (test
  with a fake LLM constructor).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
