# Local Reproduction Versions

These are local project versions, not official upstream releases.

## v1.0.1 - Retrieval Judge output validation

- Replace the substring-based early-stop condition with a validated single-role verdict.
- Retry once only when the initial verdict is malformed; retain initial prompt and valid-output behavior.
- If retry is still invalid, leave the query pool unchanged and proceed within the existing round limit.
- Save initial/retry text, retry prompt, normalized decision and decision status for auditing.
- Add dependency-free parser/policy tests and real query-stage tests with fake model/retriever calls.
- No changes to response debate, corpus, retriever, base model, metrics or round limits.

See docs/JUDGE_PARSING_FIX.md for evidence, policy and experiment protocol.

## v1.0.0 - Unmodified baseline

Exact official upstream commit c035ee1b7e6bc9fa67d5a104df1c2175a8300666.
The initial 50-question NQ run used this version. External setup/monitor scripts
are not part of the upstream repository. This tag does not include large data or credentials.
