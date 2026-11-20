---
id: J12
title: Log-prob Judge with order balancing
status: todo
env: cpu
size: M
depends: [J02]
---
# J12 Log-prob Judge with order balancing

**Goal.** Add three flags:
- `judge_scoring = "logprob"`: decide the query-stage verdict from label
  log-probabilities, `s_O = log p(Opponent) − log p(Proponent)`, instead of parsing
  generated text. No retries and no wasted round.
- `judge_order_balance`: score both presentation orders in one batched call and average
  them (`s̄_O`).
- `judge_tau`: the decision threshold.

Record `s̄_O` in the state record for J14 and J20.

**Why.** B10: an unresolved verdict burns a round, the retry costs a call, and
Proponent is always listed first. LLM judges are known to have position bias (Shi et
al., 2025).

## Research questions (save to `docs/research/J12-logprob-judge.md`)
1. vLLM 0.6.6 `SamplingParams(logprobs=k)`, `prompt_logprobs`, and the return format;
   how FlashRAG surfaces scores (`return_scores`); the cheapest way to get next-token
   log-probs for two candidate labels.
2. How Llama-3.1 tokenizes "Proponent" and "Opponent" with and without a leading space.
   Do the first tokens differ? Compute it with the tokenizer, or mark [IC] and design
   for multi-token labels.
3. Position-bias mitigation by order swapping (Shi et al., 2025); other recent
   evidence.

## Scope
- **In:** `judge/logprob.py`, a generator-adapter extension
  (`generate_with_scores`), the order-balanced prompt pair, decision logic, fallback,
  tests.
- **Out:** changing the Judge's instructions beyond what label scoring needs.

## Files
- **Create:**
  - `improvements/drag_plus/judge/{__init__,logprob}.py`
  - `improvements/tests/test_logprob_judge.py`
- **Modify:** `adapters.py`, `query_stage.py`, `prompts.py`, `runner/backends.py`,
  `flags.py`, architecture.md §3 and §7, findings.md B10.

## Design notes and edge cases
- **Scoring.** Score the full label token sequences (summed log-prob) when the first
  tokens are not unique. Use teacher-forced scoring through `prompt_logprobs`, or two
  continuation prompts, whichever research shows is cheapest. Keep the label set
  explicit.
- **Missing labels.** If a label's log-prob is unavailable (for example, outside the
  top-k), fall back to the generative verdict for that round, and log
  `judge_fallback`. Never silently treat a missing value as zero.
- **Order balance.** Build the PO and OP prompts with the agent arguments swapped,
  **and** with the instruction text naming the agents in that order. Batch both in one
  call.
- **Decision.** `s̄_O > judge_tau` means Opponent wins; ties go to Proponent
  (conservative: stop retrieving). The decision is deterministic given the scores.

## Acceptance
- [ ] With scripted fake scores: decisions match the expected ones for τ ∈ {−1, 0, 1};
  ties are handled; the fallback path is logged (tests).
- [ ] With `judge_scoring = "logprob"`, there are zero retry calls and no round wasted
  on an unresolved verdict (test call counts).
- [ ] Order balance makes one 2-prompt call per round, and the two prompts differ only
  in agent order (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
