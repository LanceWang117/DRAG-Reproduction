#!/usr/bin/env bash
# The one gate: CI runs exactly this script, so local and CI results match.
#
# Usage: bash scripts/gate.sh
#
# Requires Python 3.11 (CI, or the container from scripts/gate-docker.sh).
# Budget on the team laptop: <= 5 minutes and <= 2 GB peak RSS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STARTED=$SECONDS

fail() {
  echo "GATE FAILED: $*" >&2
  exit 1
}

step() {
  echo "==> $*"
}

require() {
  "$@" || fail "$*"
}

# The pinned interpreter is 3.11; another version may install different wheels
# or report different results, so refuse rather than produce a misleading pass.
VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
if [ -z "$VERSION" ]; then
  fail "no python on PATH; install Python 3.11 or run scripts/gate-docker.sh"
fi
if [ "$VERSION" != "3.11" ]; then
  fail "the gate requires Python 3.11, found $VERSION. Use scripts/gate-docker.sh (Docker),
or scripts/setup-venv.sh on a machine that can build 3.11. Pyenv 3.11 cannot build
against this laptop's OpenSSL 4.1, which is why Docker is the local route."
fi
step "Python $VERSION"

export PYTHONDONTWRITEBYTECODE=1
SCRIPTS="$ROOT/scripts"
STUBS="$ROOT/improvements/tests/stubs"

# Runtime and evidence scripts run with no stub on the path, so the test-only
# `flashrag` package can never satisfy an import in real code.
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"

step "Evidence integrity"
require python scripts/verify_evidence.py

step "Submission tests"
require python scripts/run_tests.py scripts --pattern test_submission.py

step "Evaluation input validation"
require python scripts/run_evaluation.py --validate-only

# DRAG_REQUIRE_INTEGRATION turns the frozen group's skip into a failure, so a
# missing import cannot pass as green. The stub is on the path for this step and
# the team tests only.
step "Frozen Judge tests (integration required)"
PYTHONPATH="$SCRIPTS:$STUBS${PYTHONPATH:+:$PYTHONPATH}" \
  DRAG_REQUIRE_INTEGRATION=1 \
  python scripts/run_tests.py src/reproduction_changes/tests --pattern 'test_*.py' \
  || fail "frozen Judge tests"

step "Team tests (no skips)"
PYTHONPATH="$SCRIPTS:$STUBS${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/run_tests.py improvements/tests || fail "improvements/tests"

step "Code limits"
require python scripts/check_code_limits.py

step "Repository guards"
require python scripts/check_guards.py

echo "GATE PASSED in $((SECONDS - STARTED))s"
