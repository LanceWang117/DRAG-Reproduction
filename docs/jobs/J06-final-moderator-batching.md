---
id: J06
title: Final-only moderator and intra-round batching
status: todo
env: cpu
size: S
depends: [J02]
---
# J06 Final-only moderator and intra-round batching

**Goal.** Add two flags, both answer-preserving:
- `moderator_final_only`: run the answer moderator once, after the last round.
- `batch_agents_in_round`: send the Proponent and Opponent prompts of each round in one
  2-prompt `generate` call, in both stages.

**Why.**
- B3: the moderator is called every round, and rounds 0–1 are overwritten
  (`drag.py:183`). That is 9 response calls against the paper's 7 (Eq. 13).
- B4: the two agents' prompts in a round are independent (`drag.py:259` reads only
  round `r−1`).
- Calls drop from `3·Q_r + 3·R` to `3·Q_r + 2·R + 1`, and per-question sequential depth
  roughly halves.

## Research questions (save to `docs/research/J06-final-moderator-batching.md`)
1. Do vLLM's `LLM.generate` and FlashRAG's `VLLMGenerator.generate` return outputs in
   input order? (Cite the docs or source.)
2. The paper's response-stage definition (§3, Eq. 13): confirm that the moderator
   judges only the last round.
3. Known effects of batch size on greedy outputs. Keep it brief; protocol.md already
   covers the A/A floor.

## Scope
- **In:** the two flags, the stage code, tests.
- **Out:** prompt text changes; cross-question batching (J15).

## Files
- **Modify:** `improvements/drag_plus/answer_stage.py`, `query_stage.py`, `flags.py`,
  architecture.md §7, findings.md B3 and B4.
- **Create:** `improvements/tests/test_final_moderator_batching.py`.

## Design notes and edge cases
- **Final moderator prompt.** With `moderator_final_only`, it must be byte-identical to
  the baseline's last-round moderator prompt (same `agents_responses` construction).
  The `update_output` keys for rounds 0..R−2 moderators are simply absent; document
  that in architecture.md.
- **`max_answer_debate_rounds = 0`.** Keep the answer-only path unchanged.
- **Batching.**
  - Map outputs back to roles by position.
  - Assert that the output count equals the prompt count, and raise a clear error if
    not.
  - The query stage must record `agents_messages[role] = [prompt, output]` exactly as
    the baseline does, because the Judge prompt is built from it.
- **Combinations.** Both flags combine with each other and with every later flag. Test
  the 4 on/off combinations.

## Acceptance
- [ ] `moderator_final_only` on: the answer stage makes exactly `2R + 1` generations;
  the final prompt and `pred` equal the baseline's (test, R ∈ {1, 2, 3}).
- [ ] `batch_agents_in_round` on: `generate` receives one 2-element list per round in
  both stages; transcripts are otherwise identical to the baseline (test).
- [ ] T0 still passes with both flags off.
- [ ] findings.md: B3 "Addressed by J06 (validation: E01)"; B4 partially addressed.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
