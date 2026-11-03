# Source ownership

- `original_drag/`: unchanged authors' repository at commit
  `c035ee1b7e6bc9fa67d5a104df1c2175a8300666`, including its license and README.
- `reproduction_changes/model/drag.py`: our edited copy of the original file;
  changes are limited to retrieval Judge validation and retry recording.
- `reproduction_changes/model/judge_decision.py`: our added parsing/retry helper.
- `reproduction_changes/tests/`: our added tests, with comments explaining the
  failure cases. They are not part of the authors' original project.

`scripts/prepare_source.py` combines these into ignored `.runtime/drag/` for a
run. It does not edit `original_drag/`. The runtime directory is generated;
make source changes in the appropriate tracked directory instead.
