#!/usr/bin/env bash
# Optional local virtualenv for the gate, for machines where Python 3.11 can be
# built or is already installed.
#
# Usage: bash scripts/setup-venv.sh
#
# On the team laptop this is expected to FAIL: pyenv builds of 3.11 do not
# compile `_ssl.c` against the system OpenSSL 4.1 and roll back. That is not a
# bug to retry; the script says so once and points at scripts/gate-docker.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
WANTED=3.11

die() {
  echo "setup-venv: $*" >&2
  exit 1
}

find_python() {
  for candidate in "python$WANTED" python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
      if [ "$version" = "$WANTED" ]; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON="$(find_python)" || die "no Python $WANTED interpreter found.
This machine cannot build one with pyenv either: 3.11's _ssl.c does not compile
against OpenSSL 4.1, and pyenv rolls the build back. Use the container route
instead: bash scripts/gate-docker.sh"

echo "==> creating $VENV with $PYTHON"
"$PYTHON" -m venv "$VENV" || die "could not create a virtualenv; retry nothing, use bash scripts/gate-docker.sh"

"$VENV/bin/python" -m pip install --quiet --upgrade pip \
  || die "pip upgrade failed; use bash scripts/gate-docker.sh"

echo "==> installing requirements-ci.txt"
"$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements-ci.txt" \
  || die "dependency install failed; use bash scripts/gate-docker.sh"

echo "==> verifying the interpreter"
VERSION="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$VERSION" = "$WANTED" ] || die "venv reports $VERSION, expected $WANTED"

echo "setup-venv: ready. Run the gate with: PATH=\"$VENV/bin:\$PATH\" bash scripts/gate.sh"
