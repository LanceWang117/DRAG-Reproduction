"""Import the frozen v1.0.1 modules for comparison tests (test-only path setup).

The runnable copy of the DRAG sources is built by ``scripts/prepare_source.py`` and the
fake ``flashrag`` package from ``stubs/`` satisfies the module-level imports. Neither
path is ever added by runtime code.
"""
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _entry in (PROJECT / "scripts", HERE / "stubs"):
    sys.path.insert(0, str(_entry))
from prepare_source import build_runtime  # noqa: E402

sys.path.insert(0, str(build_runtime()))
sys.path.insert(0, str(PROJECT / "improvements"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("TQDM_DISABLE", "1")

from model import judge_decision as frozen_judge  # noqa: E402,F401
from model.drag import DebateAugmentedRAG  # noqa: E402,F401
