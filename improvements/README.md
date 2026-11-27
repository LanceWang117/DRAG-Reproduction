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
| `pipeline.py` | Orchestration: `DragPlusPipeline(flags, generator, retriever, template, dataset).run(items)`; `run_batch` feeds B questions to the barrier or the refill window |
| `scheduling/barrier.py` | J15 barrier: groups requests at one `(stage, round, role)` into one `generate`/`batch_search`, isolates a failed question and leaves finished ones; J16 gathers all ready searches per step |
| `scheduling/refill.py` | J16 request/result types and `RefillWindow`: a finished question's slot refills immediately; `gather_search` caps and stitches the step's search queries |
| `retrieval/cache.py` | Caching retriever proxy: `search` is `batch_search([q])[0]`, one JSONL line per query |
| `analysis/` | Scoring, the unified answer extractor and paired statistics |
| `grounding/` | J19 soft faithfulness scorers (heuristic, lazy MiniCheck) and the weighted aggregation rule |
| `gates/router_features.py`, `gates/router.py` | J20 four pre-debate features (retrieval margin, `s_bar_O`, answer agreement, Naive-answer entropy) and the fitted cost-aware router: `argmax_r P(correct \| r) - lambda*cost(r)`; `router` defaults to null |
| `gates/conformal.py`, `gates/conformal_metrics.py`, `scripts/eval_conformal.py` | J21 split-conformal selective answering: score `1 - p_hat` over E(.) answer clusters, per-dataset `ceil((n+1)(1-alpha))/n` threshold with a below-`min_n` pooled fallback, abstain (or escalate) when the set is not a singleton; the script cross-fits five ways and reports Clopper-Pearson coverage, set size, abstention and selective accuracy with a dependency-free SVG curve; `conformal_alpha` defaults to null |

The package imports without FlashRAG, vLLM, torch or FAISS; the runner injects those
objects. Its only third-party imports are the light `Levenshtein` (query-pool edit
distance) and, for the analysis modules, `numpy`. Keep new variants behind flags whose defaults equal the baseline, and compare
on fixed questions.
