"""Repository guards that git history can enforce (architecture ledger section 6).

Three independent checks, each reporting every offending path so one run explains
the whole problem:

* **Frozen paths**: nothing may change under ``src/original_drag/``,
  ``src/reproduction_changes/`` or ``results/``.
* **Large files**: a changed file may not exceed 1 MB. Files that already exceed
  it are untouched and grandfathered, so only *changed* paths are measured.
* **Hygiene**: no tracked or staged pack/agent path, and no commit in the
  compared range may carry an attribution trailer.

Comparison base: the merge base with ``origin/main`` on a branch, ``HEAD~1`` on a
push to main, overridable with ``--base`` for tests and unusual checkouts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FROZEN_PREFIXES = ('src/original_drag/', 'src/reproduction_changes/', 'results/')
PACK_PATHS = ('AGENTS.md', 'CLAUDE.md', 'CLAUDE.local.md')
PACK_PREFIXES = ('docs/prompts/', 'docs/jobs/', 'docs/ledger/', 'docs/research/',
                 '.agent/', '.claude/', '.codex/', '.opencode/', '.cursor/')
ATTRIBUTION_MARKERS = ('co-authored-by', 'assisted-by', 'generated with')
MAX_BYTES = 1024 * 1024


class Git:
    def __init__(self, root):
        self.root = root

    def run(self, *args):
        completed = subprocess.run(['git', *args], cwd=self.root, capture_output=True,
                                   text=True, check=True)
        return completed.stdout

    def existing(self, *args):
        try:
            return self.run(*args).strip()
        except subprocess.CalledProcessError:
            return ''


def resolve_base(git: Git, requested):
    """Pick a comparison base that exists in this checkout.

    Order: an explicit ``--base``, then ``GUARD_BASE`` from the environment (CI),
    then the merge base with ``origin/main``; a push straight to main has no
    divergence to measure, so it compares the pushed commit with its parent.
    """
    for candidate in (requested, os.environ.get('GUARD_BASE')):
        if candidate and git.existing('rev-parse', '--verify', '--quiet', candidate):
            return git.existing('merge-base', 'HEAD', candidate) or candidate
    branch = git.existing('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'main' and git.existing('rev-parse', '--verify', '--quiet', 'HEAD~1'):
        return 'HEAD~1'
    for candidate in ('origin/main', 'main'):
        if not git.existing('rev-parse', '--verify', '--quiet', candidate):
            continue
        merge_base = git.existing('merge-base', 'HEAD', candidate)
        if merge_base and merge_base != git.existing('rev-parse', 'HEAD'):
            return merge_base
    if git.existing('rev-parse', '--verify', '--quiet', 'HEAD~1'):
        return 'HEAD~1'
    return ''


def changed_files(git: Git, base):
    """Paths added, copied, modified or renamed between ``base`` and ``HEAD``."""
    if not base:
        return []
    output = git.run('diff', '--name-only', '--diff-filter=ACMR', '-z', base, 'HEAD')
    return [path for path in output.split('\0') if path]


def staged_files(git: Git):
    output = git.run('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z')
    return [path for path in output.split('\0') if path]


def untracked_files(git: Git):
    output = git.run('ls-files', '--others', '--exclude-standard', '-z')
    return [path for path in output.split('\0') if path]


def check_frozen(paths):
    return [f'{path}: frozen path changed' for path in paths
            if path.startswith(FROZEN_PREFIXES)]


def check_large(root: Path, paths):
    problems = []
    for path in paths:
        target = root / path
        if target.is_file() and target.stat().st_size > MAX_BYTES:
            problems.append(f'{path}: {target.stat().st_size} bytes > 1 MB')
    return problems


def is_pack_path(path):
    return path in PACK_PATHS or path.startswith(PACK_PREFIXES) or \
        Path(path).name.startswith('.aider')


def check_hygiene(paths):
    return [f'{path}: pack or agent file' for path in paths if is_pack_path(path)]


def check_messages(git: Git, base):
    if not base:
        return []
    revision = f'{base}..HEAD'
    problems = []
    for line in git.run('log', '--format=%H%x1f%B%x1e', revision).split('\x1e'):
        if '\x1f' not in line:
            continue
        sha, message = line.split('\x1f', 1)
        lowered = message.lower()
        for marker in ATTRIBUTION_MARKERS:
            if marker in lowered:
                problems.append(f'{sha[:12]}: commit message contains "{marker}"')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', help='commit or ref to compare against')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    git = Git(root)

    base = resolve_base(git, args.base)
    committed = changed_files(git, base)
    problems = []
    problems += check_frozen(committed)
    problems += check_large(root, committed + staged_files(git))
    problems += check_hygiene(committed + staged_files(git) + untracked_files(git))
    problems += check_messages(git, base)

    if problems:
        print(f'{len(problems)} guard violation(s) against {base or "the working tree"}:')
        for problem in problems:
            print(f'  {problem}')
        return 1
    print(f'Guards OK against {base or "the working tree"}: '
          f'frozen paths, {MAX_BYTES // 1024} KB files, pack hygiene, commit messages.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
