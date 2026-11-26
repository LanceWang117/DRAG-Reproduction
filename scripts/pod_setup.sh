#!/usr/bin/env bash
# Check an A100 pod before an E-job, and optionally create config.local.json (J26).
#
#   bash scripts/pod_setup.sh                 # report what is ready and what is missing
#   bash scripts/pod_setup.sh --write-config  # also copy config.example.json if absent
#   bash scripts/pod_setup.sh --teardown      # fail if any of your GPU processes remain
#   --dry-run                                 # never write anything
#
# Idempotent: it never overwrites config.local.json, never downloads, never prints
# tokens. Exit 0 when everything is present, 1 when something is missing (or, with
# --teardown, a GPU process remains), 3 if config.local.json is tracked by git.
set -uo pipefail

ROOT="${POD_SETUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT" || exit 2
DRY_RUN=0 WRITE_CONFIG=0 TEARDOWN=0 MISSING=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --write-config) WRITE_CONFIG=1 ;;
    --teardown) TEARDOWN=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

ok() { echo "  ok       $*"; }
missing() { echo "  MISSING  $*"; MISSING=$((MISSING + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

teardown() {
  local pids owner own="" query
  have nvidia-smi || { echo "  ok       no nvidia-smi: no GPU processes"; return 0; }
  if ! query="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)"; then
    echo "  FAILED   nvidia-smi could not list GPU processes: $query"
    return 1
  fi
  pids="$(printf '%s\n' "$query" | tr -d ' ')"
  for pid in $pids; do
    owner="$(stat -c %u "/proc/$pid" 2>/dev/null || ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ -z "$owner" ]; then  # e.g. a host PID seen from a container: never assume it is gone
      own="$own $pid(owner unknown)"
    elif [ "$owner" = "$(id -u)" ]; then
      own="$own $pid($(cat "/proc/$pid/comm" 2>/dev/null || echo '?'))"
    fi
  done
  if [ -n "$own" ]; then
    echo "  LEFT     your GPU processes:$own (stop them before ending the session)"
    return 1
  fi
  echo "  ok       none of your processes hold the GPU"
}

if [ "$TEARDOWN" = 1 ]; then
  echo "==> teardown"
  teardown
  exit $?
fi

echo "==> tools"
for tool in nvidia-smi conda git gh huggingface-cli rsync; do
  if have "$tool"; then ok "$tool"; else missing "$tool (see REPRODUCTION.md and DATA_DOWNLOADS.md)"; fi
done
if have python && python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
  ok "python 3.11 in the active environment"
else
  missing "python 3.11 in the active environment (conda activate the DRAG env)"
fi
if have nvidia-smi; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
fi

echo "==> accounts (no tokens printed)"
if have gh && gh auth status >/dev/null 2>&1; then ok "gh logged in"; else missing "gh auth login"; fi
if [ -n "$(git config user.name 2>/dev/null)" ] && [ -n "$(git config user.email 2>/dev/null)" ]; then
  ok "git identity set"
else
  missing "git config user.name / user.email"
fi
if have huggingface-cli && huggingface-cli whoami >/dev/null 2>&1; then
  ok "huggingface-cli logged in (Llama licence)"
else
  missing "huggingface-cli login (Llama-3.1 licence)"
fi

echo "==> config.local.json"
if git ls-files --error-unmatch config.local.json >/dev/null 2>&1; then
  echo "  REFUSED  config.local.json is tracked by git; untrack it before anything else" >&2
  exit 3
fi
if [ -f config.local.json ]; then
  ok "config.local.json exists (kept as is)"
elif [ "$WRITE_CONFIG" = 1 ] && [ "$DRY_RUN" = 0 ]; then
  cp config.example.json config.local.json && ok "created config.local.json from config.example.json: edit its paths"
else
  missing "config.local.json (rerun with --write-config, then edit the paths)"
fi
if [ -f config.local.json ] && ! git check-ignore -q config.local.json 2>/dev/null; then
  missing "config.local.json is not ignored by git"
fi

echo "==> data paths in config.local.json"
if [ -f config.local.json ] && have python; then
  python - <<'PY' || MISSING=$((MISSING + 1))
import json, pathlib, sys
config = pathlib.Path("config.local.json")
data = json.loads(config.read_text())
paths = list(data.get("model2path", {}).values()) + [data.get("index_path"), data.get("corpus_path")]
bad = 0
for value in paths:
    path = pathlib.Path(str(value)).expanduser()
    path = path if path.is_absolute() else config.parent / path
    print(("  ok       " if path.exists() else "  MISSING  ") + str(path))
    bad += not path.exists()
sys.exit(1 if bad else 0)
PY
fi

echo "==> pack (local-only files)"
if [ -f AGENTS.md ] && [ -d docs/jobs ]; then ok "pack present"; else missing "pack (run scripts/pack_sync.sh push from the PC)"; fi
if git check-ignore -q docs/jobs/E00.md 2>/dev/null && git check-ignore -q AGENTS.md 2>/dev/null; then
  ok "pack is git-excluded"
else
  missing "pack exclude block in .git/info/exclude (scripts/pack_sync.sh push adds it)"
fi

echo "==> $MISSING missing"
[ "$MISSING" = 0 ]
