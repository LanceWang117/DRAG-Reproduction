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

export PYTHONPATH="$ROOT/scripts:$ROOT/improvements/tests/stubs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

step "Evidence integrity"
python scripts/verify_evidence.py || fail "verify_evidence.py"

step "Submission tests"
python scripts/run_tests.py scripts --pattern test_submission.py || fail "scripts tests"

step "Evaluation input validation"
python scripts/run_evaluation.py --validate-only || fail "run_evaluation.py --validate-only"

# DRAG_REQUIRE_INTEGRATION turns the frozen group's skip into a failure, so a
# missing import cannot pass as green. The stub satisfies model.drag's imports.
step "Frozen Judge tests (integration required)"
DRAG_REQUIRE_INTEGRATION=1 \
  python scripts/run_tests.py src/reproduction_changes/tests --pattern 'test_*.py' \
  || fail "frozen Judge tests"

step "Team tests (no skips)"
python scripts/run_tests.py improvements/tests || fail "improvements/tests"

step "Code limits"
python scripts/check_code_limits.py || fail "code limits"

step "Repository guards"
python scripts/check_guards.py || fail "repository guards"

echo "GATE PASSED in $((SECONDS - STARTED))s"
