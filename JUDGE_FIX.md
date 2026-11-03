# Retrieval Judge parsing fix

The original code stopped retrieval whenever the Judge output contained
`Proponent`, even if it also mentioned `Opponent` and gave no clear winner.

Our change accepts one role name only, retries once for invalid output, and
keeps the current query pool if the retry is still unclear. It records the
decision and retry for review. This is a validation fix, not a claimed accuracy
improvement; the reported DRAG results use it.

Changed code: `src/reproduction_changes/model/drag.py` and `judge_decision.py`.
Tests: `src/reproduction_changes/tests/`. Original code: `src/original_drag/`.
