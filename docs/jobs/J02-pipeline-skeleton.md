---
id: J02
title: Pipeline skeleton, flags and golden transcript
status: todo
env: cpu
size: M
depends: [J00]
---
# J02 Pipeline skeleton, flags and golden transcript

**Goal.** Re-implement the v1.0.1 DRAG pipeline as `improvements/drag_plus/`. It has
injected adapters, a validated flag registry and a per-question state record. Prove,
with the golden-transcript test **T0**, that with default flags it behaves exactly like
`src/reproduction_changes/model/drag.py`.

**Why.** Every improvement is a flag on this package (architecture.md §2–4). The frozen
file is 334 lines with long functions, so it cannot be extended in place.

## Research questions (save to `docs/research/J02-pipeline-skeleton.md`)
1. Which `item.update_output` keys the baseline writes, and which ones FlashRAG's
   evaluator and saved intermediate data rely on. `drag_plus` must write the same keys.
2. How FlashRAG's `PromptTemplate.get_string(messages=...)` is called by the baseline.
   The adapter must use the identical call.
3. Python `typing.Protocol` versus duck typing for the adapters; pick one and justify
   it.

## Scope
- **In:**
  - `flags.py` (schema, defaults equal to the baseline, an unknown flag is an error,
    load from JSON, export to the manifest);
  - `state.py`;
  - `adapters.py`;
  - `prompts.py` (all baseline prompt text, verbatim);
  - `query_stage.py`;
  - `answer_stage.py`;
  - `pipeline.py`;
  - fakes and tests.
- **Out:** runner integration (J03); any behaviour change.

## Files
- **Create:**
  - `improvements/drag_plus/{flags,state,adapters,prompts,query_stage,answer_stage,pipeline}.py`
  - `improvements/tests/fakes.py`
  - `improvements/tests/test_flags.py`
  - `improvements/tests/test_golden_transcript.py`
- **Modify:** `docs/ledger/architecture.md`, if the layout changes.

## Design notes and edge cases
- **Behaviour.** Replicate baseline behaviour **including its quirks**: dead moderator
  calls, Levenshtein pool deletion, the `continue` on unresolved verdicts, and the
  StrategyQA prompt branches. Fixes come later, behind flags.
- **Allowed deviations:**
  - The baseline prints and re-raises on pool errors (`drag.py:307-312`); `drag_plus`
    raises a typed error without printing. Document this; T0 does not cover crash paths.
  - The baseline's `agents_num` precedence bug (`drag.py:30`) is not copied.
- **T0 comparison.** Import the frozen class with `improvements/tests/stubs` on
  `sys.path` (J00 stub). Drive both implementations with the same scripted fake
  generator, retriever and identity prompt template. Compare:
  - the ordered list of generator inputs, including params;
  - the retriever calls;
  - every `update_output` key and value;
  - the final query pool and `pred`.
- **T0 scenarios:**
  - (a) valid Proponent at round 0;
  - (b) Opponent expansion, then Proponent;
  - (c) optimization replacing the original query;
  - (d) an ambiguous verdict, then a retry;
  - (e) unresolved twice;
  - (f) a StrategyQA prompt;
  - (g) the Opponent output has no action (pool returned unchanged);
  - (h) `max_query_debate_rounds = 0`;
  - (i) `max_answer_debate_rounds = 0`.
- **State.** No state on `self` between questions. Each call builds its own state
  record.
- **Limits.** Keep each module ≤ 250 lines and each function ≤ 40; split the prompt
  builders.

## Acceptance
- [ ] T0 passes for every scenario (a)–(i).
- [ ] `import improvements.drag_plus` succeeds with FlashRAG, vLLM and torch absent
  (tested without the stub on the path).
- [ ] An unknown flag, or a wrong flag type, raises a clear error (tests).
- [ ] The gate is green, and the code limits hold with no allowlist growth.
- [ ] architecture.md §1 and §7 reflect what was built.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
