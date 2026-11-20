# Findings register

Each finding says where it is, what the evidence is, and which job addresses it. Tags
are defined in [README.md](README.md). When a job fixes or refutes a finding, add a
`Resolved by <JOB> (#PR)` note; do not delete the row.

| ID | Finding | Location | Evidence | Status | Addressed by |
| --- | --- | --- | --- | --- | --- |
| **B1** | EM gain not established; underpowered, uneven, post-pilot sample; StrategyQA on train; no CIs or retrieval metrics; Naive (`c035ee1`) and DRAG (`3870c0e`) ran from different commits; the portable runner is untested on GPU | `results/`, `REPRODUCTION.md` | baseline.md; `run-records.json` | [D] | J01, J05, E00, E06 |
| **B2** | **Same query, different passages:** 32 of 124 unchanged pools differ from Naive's retrieval (`NQ/dev_5420`: zero overlap) | `drag.py:79,86,298,302` (`search`) vs Naive's `batch_search` | `retrieval-200.jsonl` | [D]; root cause [IC] | J04, E00 |
| **B3** | **2 dead moderator calls per question.** The answer moderator runs every round, and `pred` is overwritten each time (`drag.py:183`); agents never read its output (relay reads peers only, `:259`). That makes 9 response calls vs the paper's 7 | `answer_stage_debate`, `drag.py:144-183` | Code + paper §5.6 | [C][P]; cost [E] | J06, E01 |
| **B4** | **Strictly serial execution.** Proponent and Opponent prompts within a round are independent but run one after another; questions run one at a time, although upstream `mad.py:34-90` batches across questions | `drag.py:51-69, 89-100, 146-168`; `scripts/run_evaluation.py:118-124` | Code | [C] | J06, J15, J16 |
| **B5** | **CPU exact scan** of ≈ 64.5 GB per query at 12 threads (≈ 27 allowed); single-query calls | `config.example.json` (`faiss_gpu: false`); `run_evaluation.py:66` | baseline.md | [C]; ~3 s [E] | J09, E01 |
| **B6** | **Fixed R = 3 response rounds and 256 output tokens for every role** | `drag.py:144`; `run_evaluation.py:113`; `base_config.yaml:77` | Paper Table 6: R = 1 beats R = 3 on 5/6 datasets (avg 45.23 vs 44.17); the exception is 2Wiki (27.8 vs 28.8) | [C][P] | J08, J14 |
| **B7** | **Query Optimization deletes the original question's evidence.** 53/200 pools lack the original query (46 fully replaced); 27.5% of Naive's top-3 passages are absent from DRAG's final pool; rewrites add invented constraints ("from 1964 to 1985") | `maintain_query_pool`, `drag.py:282-314` (`pop` at `:296`); Levenshtein `:324` | `NQ/dev_4016` (Pelé → Romário); `2wiki/dev_1245` (hop 1 resolved, answer "Unknown") | [D][C] | J10, J13 |
| **B8** | **No passage dedup** (30/705 slots, 4.3%); `Doc 1` numbering restarts per query; no token budget; FlashRAG's vLLM generator never truncates (`max_model_len = 4096`) | `drag.py:272-322` | Pool ≈ 516 / 1,008 / 1,613 tokens (median / p90 / max, chars/4) before debate history | [D][C]; overflow [IC] | J10, J11 |
| **B9** | **Role instructions come before the shared pool,** so roles share no prompt prefix; FlashRAG doesn't enable prefix caching | `drag.py:185-240` | Code | [C] | J11 |
| **B10** | **Judge:** an unresolved verdict burns a round via `continue`; Proponent is always listed first; the retry adds a call | `drag.py:109-130` (`:130`), `:205-215` | Verdict and retry rates are not logged | [C]; rates [IC] | J12 |
| **B11** | **Closed-book Opponent.** DRAG switched to wrong parametric answers on *identical* evidence (`dev_3381`, `dev_5657`, `test_12352`). Greedy decoding and a peer relay that includes "The answer is: X" invite anchoring | `drag.py:217-240, 251-261` | Data + code | [D][C] | J18 |
| **B12** | **Extraction asymmetry.** Naive is parsed with `single_agent_pred_parse` and DRAG is not; 0/200 DRAG outputs contain the marker. Format-only outcomes: 5 of DRAG's 24 wins and 3 of Naive's 14 wins (`dev_8509`, `dev_7633`, `dev_4830`) | `run_evaluation.py:121-124`; `utils.py:3` | `paired-200.json` | [D][C] | J01 |
| **B13** | **Raw Wikipedia text enters system prompts** (including the Judge's) with no delimiters: an indirect prompt-injection channel | `drag.py:272-280` and every prompt | Code | [C] | J10 |
| **B14** | **The query Judge always sees the original question,** never the refined query. A design question, not necessarily a bug | `drag.py:78,85,102` | Code; paper silent | [C] | J13 |
| **B15** | **Zero-novelty refinements keep the debate going.** 8 of 76 changed pools retrieved no new passage IDs (5 single-replaced, 3 multi-query: `HotpotQA/dev_840`, `HotpotQA/dev_987`, `PopQA/test_8583`); 17–20 more shared 2 of 3 IDs. There is no duplicate-query guard and no fixed-point exit. Single-replaced pools can only be compared with Naive's passages, so those counts are approximate (B2) | `drag.py:83-141, 282-304` | `retrieval-200.jsonl` | [D] | J07 |
| **B16** | **62% of pools (124/200) were never changed,** yet each paid for at least one query round. All of DRAG's net EM gain is on the 76 changed pools (28 vs 18); the 124 tie at 60 vs 60. "Proponent won at round 0" is an **upper bound** for the 124: unresolved rounds or an unparseable Opponent action (`maintain_query_pool` returning `None`) also leave a pool unchanged | `query_stage_debate`, `drag.py:71-142` | `retrieval-200.jsonl` | [D]; round-0 rate [IC] | J14, J20 |
| **B17** | **69% of wrong answers had no gold in context** (excluding StrategyQA: DRAG 59/86, Naive 65/95). Gold containment is 114–115/200 for both | Retriever config | `retrieval-200.jsonl` | [D] | J17, E04 |

## Minor defects [C]

- `drag.py:30`: `a != b + c & a != d + e` is a chained comparison (`&` binds tighter than
  `!=`), masked by the `agents_num != 2` check.
- `prompt.py:8,54,106`: "Given **four** answer candidates, Yes and No" vs "two" in
  `drag.py`.
- `drag.py:39-49`: agent state lives on `self`, which blocks cross-question batching.
- `baseline.py:107`: `sure()` doesn't return its result. `baseline.py:146-147`: hardcoded
  `/data/share` paths.
- `drag.py:307-312`: dumps the pool on error.
- `run_evaluation.py:45-53`: `expanduser()` on config paths; trusted configs only.

The frozen baseline keeps these as they are. The new pipeline (`improvements/`) must not
copy them. Each job that re-implements the affected code fixes the defect and notes it
in its status log.
