---
id: J08
title: Round and output-length budgets
status: todo
env: cpu
size: S
depends: [J02]
---
# J08 Round and output-length budgets

**Goal.** Add four flags, all method changes:
- `response_rounds` (1–3; 0 keeps the answer-only path);
- `agent_max_tokens` and `moderator_max_tokens`, passed per call;
- `concise_agents`: agents are told "at most 2 sentences, then 'The answer is: …'".

**Why.**
- B6: every question pays for 3 response rounds and up to 256 tokens per agent turn.
- The paper's Table 6 has 1 round beating 3 on 5 of 6 datasets (the exception is 2Wiki).
- Generation dominates DRAG's LLM time [E].

## Research questions (save to `docs/research/J08-round-length-budgets.md`)
1. Evidence on how shorter or concise reasoning affects multi-hop QA accuracy for
   8B-class models (for example "Chain of Draft" or concise-CoT work). Cite only what
   you can verify.
2. vLLM `SamplingParams.max_tokens` and `finish_reason == "length"` semantics, and how
   FlashRAG passes per-call `**params`.

## Scope
- **In:** the flags, per-role parameter passing, finish-reason logging, extractor
  fallback for truncated outputs, tests.
- **Out:** adaptive stopping (J14).

## Files
- **Modify:** `answer_stage.py`, `prompts.py`, `flags.py`, `instrument/timing.py`
  (finish_reason), architecture.md §7.
- **Create:** `improvements/tests/test_budgets.py`.

## Design notes and edge cases
- **Round count.** Round semantics follow the baseline: rounds run 0..R−1.
  - `response_rounds = 1` with `moderator_final_only` gives 2 agent calls plus 1
    moderator call.
  - Without that flag it gives 2 + 1 as well, because the baseline calls the moderator
    once per round.
- **Truncated outputs.** An output cut off before "The answer is:" must still produce a
  prediction through `E(·)`'s first-line fallback. Log `truncated: true` in the state
  record.
- **Prompt changes.** `concise_agents` changes only the agent instruction text. Keep the
  original text as the default, and put both versions in `prompts.py`.
- **Parameter passing.** Pass `max_tokens` per call, never by mutating the generator's
  shared configuration (FlashRAG copies `generation_params` per call). Test that one
  call's parameters do not leak into the next.

## Acceptance
- [ ] Call counts for R ∈ {0, 1, 2, 3}, each with and without `moderator_final_only`
  (test matrix).
- [ ] Per-role `max_tokens` reaches the fake generator; there is no leakage between
  calls (test).
- [ ] The truncated-output fallback is tested, and `finish_reason` is logged.
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
