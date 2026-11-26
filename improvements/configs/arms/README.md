# E-job arm files

Every variant file under `<job>/<arm>.json` is generated from two inputs, so arms
cannot drift apart:

- `spec.json`: named flag layers, and per job and arm the layers to merge in order. A
  later layer wins, key by key.
- `measured.json`: the layers only the server can decide, each with `provisional` and
  `source`:
  - `P`: E01's promoted configuration;
  - `threads`: E00's best CPU thread count;
  - `best_hybrid`: E04's best hybrid settings;
  - `promoted_drag`: flags promoted by E02a/E02b, used by E05;
  - `best_batch`: E03's promoted question batch size, used by E05.

```bash
python scripts/make_arms.py            # regenerate after editing either file
python scripts/make_arms.py --check    # the team tests run this; a stale file fails CI
```

A generated file's `description` names its job, arm and layers, and marks any
provisional layer, so `run-manifest.json` records what ran. **Do not book the A100 for an
arm whose description says PROVISIONAL.** After E01, set `P` to the promoted flags with
`"provisional": false` and a source that cites `experiments/E01/summary.json`, then
regenerate and commit.

Run an arm with `python scripts/run_variant.py --variant improvements/configs/arms/E02b/E.json ...`
or rehearse a job's arms on the PC with
`python scripts/rehearse_local.py --questions 12 --variant improvements/configs/arms/E02b/*.json`.

Not generated, because no flag exists for them:
- E03's INT4 and speculative-decoding variants: model-level server options;
- E05's refine-only and decompose routes: they need J20 and J22;
- E06's frozen arms: chosen from E-job evidence.

The `PC/` arms combine many features for rehearsals and integration tests. They are
never results.
