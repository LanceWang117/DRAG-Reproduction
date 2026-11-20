# Improvements

`drag_plus/` is the team's pipeline package. With every flag at its default it behaves
exactly like the recorded v1.0.1 baseline (`src/reproduction_changes/model/drag.py`), and
`tests/test_golden_transcript.py` (T0) proves that call for call on scripted fakes.

| Module | Role |
| --- | --- |
| `flags.py` | Validated flag registry; an unknown flag or a wrong type is an error |
| `state.py` | Per-question state record; nothing persists on the pipeline between questions |
| `adapters.py` | Protocols for the injected generator, retriever and prompt template |
| `prompts.py` | Every baseline prompt string, verbatim |
| `query_stage.py`, `answer_stage.py` | The two debate stages |
| `pipeline.py` | Orchestration: `DragPlusPipeline(flags, generator, retriever, template, dataset).run(items)` |
| `retrieval/cache.py` | Caching retriever proxy: `search` is `batch_search([q])[0]`, one JSONL line per query |
| `analysis/` | Scoring, the unified answer extractor and paired statistics |

The package imports without FlashRAG, vLLM, torch or FAISS; the runner injects those
objects. Its only third-party imports are the light `Levenshtein` (query-pool edit
distance) and, for the analysis modules, `numpy`. Keep new variants behind flags whose defaults equal the baseline, and compare
on fixed questions.
