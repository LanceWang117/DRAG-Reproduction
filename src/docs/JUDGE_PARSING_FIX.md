# Retrieval Judge Parsing Defect and Local Fix

## Evidence and location

Baseline upstream commit: c035ee1b7e6bc9fa67d5a104df1c2175a8300666 (local v1.0.0).
Location: model/drag.py, DebateAugmentedRAG.query_stage_debate.

The prompt asks for one winning agent name, but the original code uses:

```python
if "Proponent" in moderator_output:
    return query_pool
```

Observed output in the initial one-question test:

```text
Proponent Agent 1:
Opponent Agent 1:
```

The program stops despite no unambiguous verdict. In the subsequent 50-question
NQ pilot, 46 of 64 initial retrieval Judge outputs contained both labels.
The baseline preserves this behavior; the initial bad formatting is an LLM
instruction-following failure, while accepting it is a code validation defect.
This does NOT establish the authors' experiments had the same failure frequency
or that this defect explains any score difference from the paper.

## v1.0.1 repair policy

1. Accept only a full-string single-role name: Proponent or Opponent, optionally
   followed by Agent and a numeric identifier, case-insensitively. A final period
   or exclamation mark is tolerated. Role determines the decision, not agent ID.
2. Keep the original first Judge prompt. Valid initial outputs need no extra call.
3. On invalid initial output, ask the same Judge once more using the same evidence
   and arguments plus the bad output and an explicit single-label instruction.
4. Validate retry identically. Do not pick the first/last mentioned name, map
   ambiguity to Opponent, or infer a winner from free-form reasoning.
5. If still invalid, record an unresolved verdict, keep the query pool unchanged,
   and move to the next existing retrieval round. After the configured limit,
   return the current pool. Unresolved never means convergence or a winning side.

Initial prompt/output fields remain unchanged. New per-round fields:
Decision (Proponent, Opponent or null), DecisionStatus (valid_initial,
valid_retry or unresolved), and, if retried, RetryInputPrompt/RetryOutput.

An extra LLM call occurs only for invalid initial verdicts, at most once per
retrieval round. Up to 3 configured rounds imply at most 3 retry calls per
question. Runtime or retrieval frequency can increase. Accuracy need not improve;
stricter protocol validity is the immediate goal, not score tuning.

## Verification

Run `python -m unittest discover -s tests -v`. Parser/policy tests need only the
standard library. Integration tests need the existing DRAG runtime but use fake
model/retriever calls and do not load GPU weights or the large index.
Set DRAG_REQUIRE_INTEGRATION=1 to fail rather than skip missing-runtime tests.

## Paired rerun protocol

Retain all baseline outputs. Extract its exact 50 ordered IDs and original
question/gold-answer rows into a fixed subset. Rerun v1.0.1 on that subset with
the same Llama revision, full corpus/index, seed, generation parameters and 3/3
round limits. Record code commit/tag, data checksum, environment, timing/resources,
Judge retries/unresolved rates, EM/F1 and paired question-level changes.
This is a bug-fix variant, NOT an unchanged official-paper reproduction.

## Paper comparison before the fix

ACL 2025 Table 1, NQ: DRAG EM 36.80, F1 50.38; Naive RAG EM 38.20, F1 50.08.
Paper section 4.1: 500 randomly sampled questions per dataset. Our 50-question
v1.0.0 pilot: EM 34.00, F1 50.3812. Difference: -2.80 percentage points EM,
essentially zero at two-decimal F1 precision. One question changes our EM by
2 points. Different sample sizes/selections prevent a like-for-like equivalence
claim; no evidence here establishes the bug caused that gap.

Source: https://aclanthology.org/2025.acl-long.770.pdf (Table 1; section 4.1).
