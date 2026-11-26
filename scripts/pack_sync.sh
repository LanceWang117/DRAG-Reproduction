#!/usr/bin/env bash
# Copy the local-only pack between the PC and a server clone (J26; README §2b).
#
#   bash scripts/pack_sync.sh push <host:/path/to/repo | /local/repo>
#   bash scripts/pack_sync.sh pull <host:/path/to/repo | /local/repo>
#
# push: adds this checkout's local-only exclude lines to the destination clone's
#       .git/info/exclude (once), confirms git ignores the pack there and nothing from it
#       is staged, then copies AGENTS.md, CLAUDE.md and docs/.
# pull: copies back only docs/jobs, docs/research and docs/ledger/inbox, and never over a
#       newer local file.
# Code always travels through git, never through this script. The pack is read from the
# main checkout even when the script runs in a worktree; its .git/info/exclude is only
# read, never changed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PACK_SYNC_ROOT:-$(cd "$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)}"
cd "$ROOT"
[ $# -eq 2 ] || { echo "usage: pack_sync.sh push|pull <host:/path | /path>" >&2; exit 2; }
MODE="$1" DEST="$2"
PACK=(AGENTS.md CLAUDE.md docs/prompts docs/jobs docs/ledger docs/research)
PULLED=(docs/jobs docs/research docs/ledger/inbox)

remote() { [[ "$DEST" == *:* && "$DEST" != /* ]]; }
in_dest() {  # run a shell snippet inside the destination clone; stdin passes through
  if remote; then
    ssh "${DEST%%:*}" "cd $(printf '%q' "${DEST#*:}") && $1"
  else
    (cd "$DEST" && bash -c "$1")
  fi
}
copy() {  # copy <update-only:0|1> <source...> <target>; push keeps relative paths
  local update="$1"; shift
  if command -v rsync >/dev/null 2>&1; then
    if [ "$update" = 1 ]; then rsync -a --update "$@"; else rsync -aR "$@"; fi
  elif remote; then
    echo "pack_sync: rsync is needed for a remote destination" >&2; exit 1
  elif [ "$update" = 1 ]; then
    cp -a -u "$@"
  else
    cp -a --parents "$@"
  fi
}
exclude_lines() {
  grep -v '^[[:space:]]*#' "$(git rev-parse --git-path info/exclude)" | grep -v '^[[:space:]]*$' || true
}

push() {
  in_dest 'git rev-parse --is-inside-work-tree >/dev/null' \
    || { echo "pack_sync: $DEST is not a git clone" >&2; exit 1; }
  exclude_lines | in_dest 'f="$(git rev-parse --git-path info/exclude)"; mkdir -p "$(dirname "$f")"; touch "$f"
    while IFS= read -r line; do grep -qxF -- "$line" "$f" || printf "%s\n" "$line" >> "$f"; done'
  local paths="${PACK[*]}"
  if [ -n "$(in_dest "git ls-files -- $paths; git diff --cached --name-only -- $paths")" ]; then
    echo "pack_sync: the destination tracks or stages pack files; remove them from git first" >&2
    exit 1
  fi
  in_dest 'for p in AGENTS.md CLAUDE.md docs/jobs/x.md docs/prompts/x.md docs/ledger/x.md docs/research/x.md; do
      git check-ignore -q --no-index "$p" || exit 1; done' \
    || { echo "pack_sync: the destination does not ignore the pack; stopping" >&2; exit 1; }
  local items=()
  for item in "${PACK[@]}"; do [ -e "$item" ] && items+=("$item"); done
  [ "${#items[@]}" -gt 0 ] || { echo "pack_sync: no pack files in $ROOT" >&2; exit 1; }
  copy 0 "${items[@]}" "$DEST/"
  local visible
  visible="$(in_dest "git status --porcelain --untracked-files=all -- ${items[*]}")"
  if [ -n "$visible" ]; then
    printf 'pack_sync: copied files are visible to git in the destination:\n%s\n' "$visible" >&2
    exit 1
  fi
  echo "pack_sync: pushed ${items[*]} to $DEST (all ignored by git there)"
}

pull() {
  for folder in "${PULLED[@]}"; do
    mkdir -p "$folder"
    if in_dest "test -d $(printf '%q' "$folder")"; then
      copy 1 "$DEST/$folder/." "$folder/"
      echo "pack_sync: pulled $folder (newer local files kept)"
    fi
  done
}

case "$MODE" in
  push) push ;;
  pull) pull ;;
  *) echo "usage: pack_sync.sh push|pull <dest>" >&2; exit 2 ;;
esac
