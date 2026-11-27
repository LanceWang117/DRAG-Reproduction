# Improvements: the DRAG+ pipeline

This directory is the team's improvement work. `drag_plus/` reimplements the recorded
baseline behind flags, then adds changes one at a time. With every flag at its default
it is call-for-call identical to `src/reproduction_changes/model/drag.py`;
`improvements/tests/test_golden_transcript.py` (T0) proves that on scripted fakes.

**Status, in one sentence.** The code and offline tests for J00–J22, J24–J26, J30 and
J31 are merged; J23 and J27–J29 are not. The A100 measurement jobs (E00–E06) have **not**
run yet, so this README explains what each change is designed to do and what evidence
exists so far — not a measured win. The recorded 200-question baseline is still the only
accuracy and latency evidence: DRAG 28.23 s/question versus Naive 4.86 s/question, with a
small and uneven EM lead.

## How to read the evidence

| Tag | Meaning |
| --- | --- |
| **[C]** | Read from code or proved by an offline test |
| **[D]** | Recomputed from committed data (`results/`, `retrieval-200.jsonl`) |
| **[M]** | Measured on the A100 by an E-job; none yet in this directory |
| **[E]** | Estimate or hypothesis, waiting for measurement |
| **[IC]** | Not enough context to verify |

Each section below says what problem the change targets (the B-numbers are the team's
finding register), what it changes, what it should achieve, how to turn it on, and what
evidence exists. "Expected" is a hypothesis until an E-job reports **[M]**.

## The improvements at a glance

| ID | Change | Targets | What it does | What it should achieve | Flag (default) |
| --- | --- | --- | --- | --- | --- |
| J30/J31 | Full-query search | B2 | `search(q)` becomes `batch_search([q])[0]` | Removes the 64-character truncation; DRAG and Naive see the same passages | `search_via_batch` (false; on after E01) |
| J04 | Retrieval cache | B2 | One proxy, one JSONL line per query, shared by both arms | Reproducible retrieval; removes the arm-to-arm query-path difference | `retrieval_cache` (null) |
| J17 | Hybrid retrieval | B17 | E5 + BM25 weighted RRF, score floors, optional reranker, title cap | More gold passages in context; E04 measures Containment@k and nDCG | `retriever` (e5) |
| J09 | Retrieval acceleration | B5 | GPU FAISS, thread count, memory preflight | Faster retrieval without an OOM; E01 measures the gain | `faiss_gpu`, `omp_threads`, `gpu_memory_utilization` |
| J10 | Anchored evidence store | B7, B8, B13 | Append-only passage store, global `E{qid}-{rank}` labels, data delimiters, rewrite audit | Stops Optimization from deleting the original evidence; fewer duplicate slots; an injection boundary | `evidence_store` (dict) |
| J11 | Token budget and layout | B8, B9 | MMR-style selection under a token budget; evidence-first prefix; optional vLLM prefix caching | Fits more useful evidence in the context; shares a prefix across roles; E02b measures EM and latency | `context_token_budget` (null), `prompt_layout` (role_first), `enable_prefix_caching` (false) |
| J13 | Structured opponent actions | B7, B14 | One validated JSON action per round; stable query IDs; optional refined query in the Judge prompt | Fewer invalid actions and less silent pool corruption; E02b measures the invalid-action rate | `opponent_action_schema` (false), `judge_sees_refined_query` (false) |
| J12 | Log-prob Judge | B10 | Score the explicit labels at the first position; average PO/OP order; no retry | A cheaper, more consistent verdict; E02b measures position consistency | `judge_scoring` (generative), `judge_order_balance` (false) |
| J06 | Final-only moderator and intra-round batching | B3, B4 | Run the answer moderator only at the last round; send both agents in one 2-prompt call | 9 → 7 answer-stage calls at R=3 and half the sequential depth; E01 measures latency | `moderator_final_only`, `batch_agents_in_round` (false) |
| J07 | Duplicate guard and zero-novelty exit | B15 | Normalise a proposed query; skip duplicates; exit when retrieval adds no new passage ID | Avoids paid rounds that cannot change the pool; E02a measures the exit rate and EM | `skip_duplicate_query`, `zero_novelty_exit` (false) |
| J08 | Round and output budgets | B6 | `response_rounds` 0–3, per-role `max_tokens`, concise agent answers | Fewer rounds and tokens; E02a tests EM non-inferiority and latency | `response_rounds`, `agent_max_tokens`, `moderator_max_tokens`, `concise_agents` (null/false) |
| J15/J16 | Barrier and refill throughput | B4, B5 | Run B questions in lockstep; group same-stage calls; refill a finished slot; batch all ready searches | Throughput ≥ 1.2× with no state leakage; E03 measures questions/hour and per-question equivalence | `question_batch_size` (1), `refill_window` (false), `retrieval_batch_size` (64) |
| J14 | Escalation gate and lazy Opponent | B6, B16 | Stop after round 1 on a fitted risk score; skip the query Opponent when retrieval is sufficient | Fewer rounds on confident questions; E02b/E05 measure the stop rate and wrong-stop rate | `escalation_gate` (none), `lazy_query_opponent` (false) |
| J20 | Cost-aware router | B16 | Four pre-debate features choose Naive, refine-only or full DRAG under a cost penalty | Beat equal-share random routing on the EM-vs-seconds Pareto curve; E05 fits and evaluates | `router` (null) |
| J18 | Grounded answer stage and vote | B11 | Evidence-seeing Conflict Auditor and Evidence Synthesizer; verdict sequestration; vote and self-consistency baselines | Fewer wrong closed-book answers on identical evidence; E-track measures EM | `opponent_mode` (closed_book), `aggregator` (moderator) |
| J19 | Faithfulness and weighted aggregation | B11 | Deterministic containment or MiniCheck support; weighted confidence/faithfulness/agreement | Better answer selection when agents disagree; E-track measures EM | `faithfulness_scorer` (null), `aggregator` (moderator) |
| J21 | Conformal selective answering | — | Per-dataset split-conformal threshold; answer only a singleton set, otherwise abstain or escalate | Coverage close to the target with a useful abstention rate; E-track measures coverage and selective accuracy | `conformal_alpha` (null) |
| J22 | Multi-hop decomposition | B7, B17 | Plan at most three grounded hops; run a width-2 beam; compose only from grounded evidence | Better multi-hop answers without inventing hops; E-track measures EM on 2Wiki/HotpotQA | `decompose` (false) |

The job IDs are the implementation units. The E-jobs are the measurement units; a flag
without an E-job result is code and test evidence only.

## 1. Retrieval: find the right passages

The recorded baseline has two retrieval problems. **B2**: DRAG's single-query `search`
passes a bare string to FlashRAG's encoder, which embeds only the first 64 characters;
Naive's `batch_search` embeds the whole query. In 32 of 124 unchanged pools the two arms
therefore saw different passages. **B17**: 69% of wrong answers had no gold passage in
context, so better retrieval is a real lever, not only a speed problem.

**Full-query search (J30/J31).** `search_via_batch` makes DRAG's `search(q)` exactly
`batch_search([q])[0]`. It removes the truncation at the source instead of relying on a
cache. Every generated arm after E01 carries it; E01 keeps the recorded path so the
speed comparison is against the published baseline, and `E02b/opt-no-b2fix` is the
control that measures its EM effect. The expected result is fewer arm-to-arm retrieval
mismatches; E02b's A-versus-control comparison is the test.

**Retrieval cache (J04).** `retrieval_cache` writes one JSONL line per query and shares
one proxy between the arms. It makes a resumed run reproducible and gives both arms the
same query path while the B2 fix is being rolled out. Its key includes the index,
encoder, top-k and a canonical config fingerprint, so a changed retriever never hits an
old line. This is a measurement tool, not a speed claim.

**Hybrid retrieval (J17).** `retriever=hybrid` fuses E5 and BM25 with weighted reciprocal
rank fusion, applies per-source score floors before fusion, optionally reranks the head
with a cross-encoder, and caps passages per title before top-k. It is designed for the
B17 failure: a lexical match can rescue a passage that dense retrieval misses. E04
builds the BM25 index and measures Containment@3/@5, MRR and nDCG@5 against E5; the
end-to-end 2 × 2 run happens only if the retrieval gain clears a margin stated in
advance.

**Retrieval acceleration (J09).** `faiss_gpu` clones the index in fp16 before vLLM
reserves its pool; `gpu_memory_utilization` and `omp_threads` come from the variant;
`scripts/preflight_gpu.py` refuses a plan that cannot fit. The expected effect is a
faster exact scan; E01 measures it and the overlap against CPU fp32. If the GPU path
OOMs, E01's CPU-thread fallback is the supported route.

## 2. Evidence: keep it, label it, fit it

**Anchored store (J10).** The baseline's Optimization step can `pop` the original
query and its passages; 53 of 200 pools lost the original query, and 27.5% of Naive's
top-3 passages were absent from DRAG's final pool (B7). `evidence_store=anchored` keeps
an append-only passage store: Optimization may supersede a later query but never the
original, and passage labels are global `E{qid}-{rank}` rather than a per-query `Doc 1`
(B8). `evidence_delimiters` wraps passages in `<document>` data blocks with a
not-instructions notice, closing the most obvious indirect prompt-injection channel
(B13). `rewrite_novelty_audit` logs invented entities and numbers; it never blocks.
Expected effect: the final pool still contains the original evidence, with fewer
duplicate slots; E02b measures gold-in-context recall and EM.

**Token budget and layout (J11).** `context_token_budget` selects evidence MMR-style
under a token cap, with a two-chunks-per-article limit; anchored passages are never
dropped, only truncated with a warning. `prompt_layout=evidence_first` puts one shared
header + question + evidence prefix before every role instruction, and under the
anchored store each round's prefix extends the previous one. `enable_prefix_caching`
plumbs vLLM APC through the backend. Expected effect: more useful evidence per token and
a shared prefix across roles and rounds; E02b measures EM and latency, and arm F compares
the caching A/A floor with arm C.

**Structured opponent actions (J13).** The baseline Opponent emits free text, and the
next step guesses which query to replace with Levenshtein matching; an unparseable
action can silently leave the pool unchanged or replace the wrong query (B7). With
`opponent_action_schema`, the Opponent ends with one JSON action. The parser takes the
last balanced object and rejects multiple actions, unknown or missing fields, wrong
types, an empty rewrite, a duplicate query, an inactive source ID, a 3–30 token rewrite
and a dropped entity without a reason. A failure leaves the pool unchanged and logs the
raw and parsed values. Stable `Qid` labels replace the edit-distance guess. Expected
effect: fewer invalid actions and a traceable query trail; E02b reports the invalid-action
rate.

## 3. Judge: score the labels, don't parse prose

The baseline Judge can return an unresolved verdict, which burns a round; it always
lists the Proponent first; and a retry adds a call (B10). **J12** addresses all three.
`judge_scoring=logprob` reads the explicit `Yes`/`No` labels at the first generated
position, so there is no free-text parse and no retry. `judge_order_balance` builds the
PO and OP prompts (arguments and numbered lines swapped) and averages both margins in one
2-prompt call; `judge_tau` sets the decision threshold. A missing or non-finite label
falls back to text parsing and logs `judge_fallback`. Expected effect: a cheaper and
more position-consistent verdict; E02b measures the share of rounds where PO and OP
agree. The real vLLM scored path is server-unverified until then.

## 4. Cost and throughput: spend fewer calls

**Final-only moderator and intra-round batching (J06).** The baseline runs the answer
moderator every round even though only the last output is used: 9 answer-stage calls at
R=3 instead of the paper's 7 (B3). `moderator_final_only` runs it only after round R−1
with the byte-identical last-round prompt, so the early moderator keys disappear and
`pred` is written once. `batch_agents_in_round` renders both agents' prompts and sends
one 2-prompt `generate` per round in both stages; prompts are unchanged, only the call
grouping. Expected effect: two fewer calls and half the per-round sequential depth; E01
measures the latency change against the A/A floor.

**Novelty exit (J07).** Eight of 76 changed pools retrieved no new passage at all (B15).
`skip_duplicate_query` normalises a proposed query and compares token sets with the
pool before searching; an effective duplicate skips the retrieval call. `zero_novelty_exit`
snapshots the passage IDs before the action and, when the result adds nothing, restores
or refuses the pool and ends the query stage. Both log a `query_exit` decision. Expected
effect: no paid round that cannot change the evidence; E02a measures the exit rate and
the EM on the 76 changed pools.

**Round and output budgets (J08).** The baseline fixes R=3 and 256 output tokens for
every role, while the paper reports R=1 beating R=3 on five of six datasets (B6).
`response_rounds` overrides R (0–3; null inherits), `agent_max_tokens` and
`moderator_max_tokens` pass per-call caps, and `concise_agents` asks for at most two
sentences while keeping the answer marker. Truncation is logged as `truncated` when a
budgeted answer loses "The answer is:". Expected effect: fewer rounds and tokens; E02a
tests 1 pp non-inferiority and reports latency, with explicit attention to 2Wiki (the
paper's exception).

**Barrier and refill throughput (J15/J16).** The baseline runs one question at a time
and one agent at a time (B4). `question_batch_size=B` runs B questions in lockstep and
groups requests at one `(stage, round, role)` into one `generate` or `batch_search`;
B=1 keeps T0, and B ∈ {2, 4, 7} reproduces each per-question transcript. A failed
question is isolated and checkpoints flush when it finishes. `refill_window` admits the
next question as soon as a slot frees instead of waiting for a fixed chunk, and every
ready search at a step goes into one `batch_search` (capped by `retrieval_batch_size`).
Expected effect: throughput ≥ 1.2× with no state leakage; E03 measures questions/hour,
p50/p95 latency and per-question equivalence, and reports the best B.

**Escalation gate and lazy Opponent (J14).** Sixty-two percent of pools never changed,
yet each paid for a query round (B16). `escalation_gate=risk` computes four label-free
features after response round 1 and stops when the calibrated risk is at or below
`escalation_tau`; otherwise it runs one more round, capped at 3. `lazy_query_opponent`
scores retrieval sufficiency in one label call and ends the query stage when it is
sufficient; an unavailable score falls back to the baseline Opponent. `scripts/fit_gate.py`
fits and Platt-calibrates the model from JSONL transcripts and writes a reliability SVG.
Expected effect: fewer rounds and calls on confident questions; E02b/E05 measure the
stop rate and the wrong-stop rate.

**Cost-aware router (J20).** The router scores four pre-debate features — retrieval
margin, the Opponent sufficiency score, model-versus-retrieval answer agreement, and
Naive-answer token entropy — and chooses Naive, refine-only or full DRAG by
`argmax_r P(correct | r) − λ·cost(r)`. The model JSON records per-route coefficients,
costs, `feature_cost`, `lambda` and the training split; the loader refuses the test split,
feature drift, unknown routes and bad costs. `scripts/fit_router.py` fits per-route
logistic models with leave-one-dataset-out; `scripts/eval_router.py` plots the Pareto
curve against equal-share random routing. Expected effect: beat random routing at the
same DRAG share; E05 fits on train and evaluates on dev.

## 5. Answer quality and abstention

**Grounded answer stage and vote (J18).** The closed-book Opponent can switch to a
wrong parametric answer on identical evidence (B11). `opponent_mode=grounded` replaces it
with an evidence-seeing Conflict Auditor and the Proponent with an Evidence Synthesizer
that cites `[E..]` labels; `verdict_sequestration` removes only the peer-relayed
"The answer is:" segments. `aggregator=vote` takes one answer per agent;
`self_consistency` samples one grounded prompt several times at temperature. Both use
plurality over the unified extractor with an evidence-supported, then Proponent/first
tie-break. Expected effect: fewer wrong answers on identical evidence; the E-track
measures EM.

**Faithfulness and weighted aggregation (J19).** `faithfulness_scorer=heuristic` is the
deterministic whole-word containment of the normalised answer in the cited passages;
`minicheck` lazily loads the optional model and takes the maximum support probability
over `(passage, answer)` pairs. `aggregator=weighted` maximises
`w_conf·confidence + w_faith·faithfulness + w_agr·agreement`, with weights from a JSON
file that records the training split and manifest SHA-256; the loader refuses the test
split. `scripts/fit_aggregator.py` fits non-negative weights. Expected effect: better
selection when agents disagree; the E-track measures EM. The real MiniCheck path is
server-unverified.

**Conformal selective answering (J21).** `conformal_alpha` adds split-conformal
selective answering as the last step. The score is `1 − p_hat` over answer clusters
built with the unified extractor; `p_hat` combines smoothed agent-vote share, the
evidence-support feature and the query-stage Judge margin. Calibration is Mondrian per
dataset at the `ceil((n+1)(1−alpha))/n` empirical quantile; a dataset below `min_n` uses
the pooled threshold. A singleton answers; otherwise the run abstains (or escalates)
while `pred` stays untouched for full-coverage EM. `scripts/eval_conformal.py`
cross-fits five ways and reports Clopper-Pearson coverage, set size, abstention and
selective accuracy. Expected effect: coverage within 2 pp of the target with a useful
abstention rate; the E-track measures it.

**Multi-hop decomposition (J22).** B17's failures are concentrated on multi-hop
questions. `decompose` asks a planner for at most three acyclic hops with backward
`#E<n>` placeholders; a width-2 beam runs only within a 0.15 confidence margin; every
binding must be grounded in its hop's passages; one rewrite is allowed for an ungrounded
hop; the question is capped at 8 retrieval + generation calls; original and hop passages
are anchored in the J10 store. An invalid plan, an ungrounded binding or an exhausted
cap falls back to the router/normal pipeline instead of composing from memory. Expected
effect: better 2Wiki/HotpotQA EM without invented hops; the E-track measures it.

## How we test and run it

- **Baseline fidelity.** `improvements/tests/test_golden_transcript.py` (T0) runs
  `DragPlusPipeline` and the frozen baseline on scripted fakes and compares every call
  and output byte for byte. Every new flag defaults to the baseline path.
- **Offline gate.** `bash scripts/gate.sh` (or `scripts/gate-docker.sh` on the laptop)
  runs the evidence checks, the frozen Judge tests, `improvements/tests` with no skips,
  the code limits and the repository guards. The package imports without FlashRAG,
  vLLM, torch or FAISS; only `Levenshtein` and `numpy` are used by the light modules.
- **Variant runner.** `scripts/run_variant.py --variant <file> --config config.local.json`
  runs the arms in a variant file and writes `checkpoints.jsonl`, `transcripts.jsonl`,
  `stage-timings.jsonl` (with `--profile`), `run-manifest.json` and `summary.json` under
  `new-runs/`. `--resume` continues from the checkpoints; `--interleave` alternates the
  variant's arms per question.
- **Arms.** `improvements/configs/arms/<JOB>/<arm>.json` are generated by
  `python scripts/make_arms.py` from `spec.json` and `measured.json`; `--check` fails if
  they drift. Do not hand-edit them. A description that says `PROVISIONAL` is not ready
  to book on the A100.
- **Rehearsal.** `scripts/rehearse_local.py --questions 12 --variant <arms>` runs the
  same flags on the local Ollama model and stored passages. It reports structural
  behaviour only (calls, rounds, parse failures, empty outputs), never accuracy or
  latency, and is required before booking a GPU session.
- **A100 evidence.** E-jobs write raw runs to `new-runs/` and commit compact evidence to
  `experiments/<ID>/`: README, `summary.json`, figures, tables and `SHA256SUMS`. A claim
  becomes [M] only when it links that record.

## Stage gates

The E-jobs promote only what clears these gates. The conditions are fixed in advance;
the A100 run is the test, not a search for a favourable subset.

| Stage | Jobs | Pass condition |
| --- | --- | --- |
| 0 | J00–J05, E00 | CI enforces limits and frozen paths; the baseline rerun matches the recorded baseline within the A/A floor; the test set is pre-registered |
| 1 | J06, J07, J09, E01 | Answer-preserving changes stay within the A/A floor; the paired latency CI excludes 0 |
| 2 | J08, J10–J14, E02a, E02b | Each method change is EM non-inferior (TOST 1 pp) or significantly better; calls and latency reported |
| 3 | J15–J19, E03, E04 | Throughput ≥ 1.2× with no state leakage; Q1/Q4 paired gains on pre-specified subsets |
| 4 | J20–J23, E05 | The router beats random routing at equal DRAG share on the EM-vs-seconds Pareto curve; conformal coverage within 2 pp |
| 5 | E06 | Frozen configurations evaluated once on the test set |

## What is not merged

Do not cite these as improvements yet. J23 (LoRA-DPO Judge) depends on E05 and is not on
`main`. J27–J29 (Jev decision backends) are not on `main`; their E-track follow-ups are
not booked. The current package contains only the merged work above.

## Package map

| Module | Role |
| --- | --- |
| `flags.py` | Validated flag registry; unknown flags and wrong types are errors |
| `state.py` | Per-question state; nothing persists between questions |
| `adapters.py` | Protocols for the injected generator, retriever and prompt template |
| `prompts.py` | Every baseline prompt string, verbatim |
| `query_stage.py`, `answer_stage.py` | The two debate stages |
| `pipeline.py` | Orchestration; `run` for one question, `run_batch` for B questions |
| `scheduling/barrier.py`, `scheduling/refill.py` | J15/J16 request grouping, failure isolation and the refill window |
| `retrieval/cache.py`, `retrieval/hybrid.py`, `retrieval/bm25.py` | Caching proxy, hybrid fusion, BM25 adapter |
| `evidence/` | Anchored store, delimiters, token budget, layout and rewrite audit |
| `judge/` | Log-prob scoring, order balancing, JSON action schema |
| `gates/` | Escalation gate, router, conformal calibration and metrics |
| `grounding/` | Heuristic and MiniCheck faithfulness scorers |
| `aggregators.py` | Moderator, vote, self-consistency and weighted aggregation |
| `decompose/` | Multi-hop planner, beam, grounding and composition |
| `analysis/` | Unified answer extractor, scoring, paired statistics |
| `rehearsal/`, `runner/`, `server/` | Local rehearsal, variant runner and server checks |

The package imports without FlashRAG, vLLM, torch or FAISS; the runner injects those
objects. Keep new variants behind flags whose defaults equal the baseline, and compare
on fixed questions.

## Limitations and next steps

- **No A100 result yet.** Every accuracy, latency and throughput number in this README
  is a target or an estimate. E00 establishes the A/A floor; E01–E06 test the changes.
- **Server-unverified paths.** Log-prob Judge, GPU FAISS, prefix caching, MiniCheck,
  hybrid retrieval and the decomposition planner have offline tests but have not run on
  the real vLLM/FlashRAG stack. E00–E04 are the checks.
- **The test set is sealed.** `experiments/J05-testset/` is used once, in E06. Do not
  tune on it, prompt with it or inspect it before the frozen run.
- **The baseline is small.** The recorded 200 questions are uneven and underpowered;
  the 600-question test set is the confirmatory sample. Treat 200-question differences
  as exploratory.
- **One change at a time.** The E-job arm files are paired against the promoted
  configuration, so a gain is attributable. Do not bundle untested flags into a new arm.
