"""Repository guards that git history can enforce (architecture ledger section 6).

Three independent checks, each reporting every offending path so one run explains
the whole problem:

* **Frozen paths**: nothing may change under ``src/original_drag/``,
  ``src/reproduction_changes/`` or ``results/`` — added, modified, renamed away
  or deleted.
* **Large files**: a file may not grow past 1 MB. The size comes from the git
  blob at the compared revision (or the working tree for staged/untracked files),
  and a file that is already over the limit in the base is grandfathered as long
  as it is not changed.
* **Hygiene**: no tracked, staged or untracked pack/agent path, and no commit in
  the compared range may carry an attribution trailer.

Comparison base: ``--base``, else ``GUARD_BASE`` from the environment (CI), else
the merge base with ``origin/main``. On a push to main there is nothing to
compare, so the pushed commit's parent is used. The base is never allowed to
equal ``HEAD``: that would compare the tree with itself and pass vacuously.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FROZEN_PREFIXES = ('src/original_drag/', 'src/reproduction_changes/', 'results/')
PACK_PATHS = ('AGENTS.md', 'CLAUDE.md', 'CLAUDE.local.md', 'GEMINI.md',
              '.cursorrules', '.windsurfrules')
PACK_PREFIXES = ('docs/prompts/', 'docs/jobs/', 'docs/ledger/', 'docs/research/',
                 '.agent/', '.claude/', '.codex/', '.opencode/', '.cursor/',
                 '.github/copilot-instructions.md')
TRAILERS = ('co-authored-by:', 'assisted-by:')
GENERATED_MARKER = 'generated with'
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

    def head(self):
        return self.existing('rev-parse', 'HEAD')


def _resolve(git: Git, candidate):
    if not candidate or not git.existing('rev-parse', '--verify', '--quiet', candidate):
        return ''
    return git.existing('merge-base', 'HEAD', candidate) or ''


def resolve_base(git: Git, requested):
    """Pick a comparison base that exists and differs from ``HEAD``."""
    head = git.head()
    for candidate in (requested, os.environ.get('GUARD_BASE')):
        if not candidate:
            continue
        base = _resolve(git, candidate)
        if not base:
            raise ValueError(f'cannot resolve a comparison base for {candidate!r}')
        if base != head:
            return base
        break
    branch = git.existing('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'main' and git.existing('rev-parse', '--verify', '--quiet', 'HEAD~1'):
        return 'HEAD~1'
    for candidate in ('origin/main', 'main'):
        base = _resolve(git, candidate)
        if base and base != head:
            return base
    if git.existing('rev-parse', '--verify', '--quiet', 'HEAD~1'):
        return 'HEAD~1'
    raise ValueError('no comparison base available; pass --base')


def changed_files(git: Git, base):
    """Every path touched between ``base`` and ``HEAD``, both sides of a rename."""
    output = git.run('diff', '--name-only', '--no-renames', '--diff-filter=ACMRD',
                     '-z', base, 'HEAD')
    return [path for path in output.split('\0') if path]


def staged_files(git: Git):
    output = git.run('diff', '--cached', '--name-only', '--no-renames',
                     '--diff-filter=ACMRD', '-z')
    return [path for path in output.split('\0') if path]


def untracked_files(git: Git):
    output = git.run('ls-files', '--others', '--exclude-standard', '-z')
    return [path for path in output.split('\0') if path]


def check_frozen(paths):
    return [f'{path}: frozen path changed' for path in paths
            if path.startswith(FROZEN_PREFIXES)]


def _baseline_size(git: Git, base, path):
    """Size of ``path`` at ``base``, or ``None`` if it did not exist there."""
    if not base:
        return None
    value = git.existing('cat-file', '-s', f'{base}:{path}')
    return int(value) if value.isdigit() else None


def _matches_any(path, prefixes):
    return any(path.startswith(prefix) for prefix in prefixes)


def check_large(git: Git, base, paths):
    problems = []
    for path in paths:
        baseline = _baseline_size(git, base, path)
        if baseline is not None and baseline > MAX_BYTES:
            continue  # already over the limit before this change; grandfathered
        size = git.existing('cat-file', '-s', f'HEAD:{path}')
        if not size.isdigit():
            target = git.root / path
            size = str(target.stat().st_size) if target.is_file() else '0'
        if int(size) > MAX_BYTES:
            problems.append(f'{path}: {size} bytes > 1 MB')
    return problems


def is_pack_path(path):
    return path in PACK_PATHS or _matches_any(path, PACK_PREFIXES) or \
        Path(path).name.startswith('.aider')


def check_hygiene(paths):
    return [f'{path}: pack or agent file' for path in paths if is_pack_path(path)]


def check_messages(git: Git, base):
    """Reject attribution trailers and "generated with" footers.

    Trailer markers are matched at the start of a line, so a commit message that
    *discusses* the policy is not itself a violation; the phrase "generated with"
    is matched anywhere, because it is a footer rather than a trailer.
    """
    if not base:
        return []
    problems = []
    for record in git.run('log', '--format=%H%x1f%B%x1e', f'{base}..HEAD').split('\x1e'):
        record = record.lstrip('\n')
        if '\x1f' not in record:
            continue
        sha, message = record.split('\x1f', 1)
        for line in message.splitlines():
            stripped = line.strip().lower()
            marker = next((item for item in TRAILERS if stripped.startswith(item)), None)
            if marker:
                problems.append(f'{sha[:12]}: commit message contains "{marker}"')
                break
        else:
            if GENERATED_MARKER in message.lower():
                problems.append(f'{sha[:12]}: commit message contains "{GENERATED_MARKER}"')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', help='commit or ref to compare against')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    git = Git(args.root.resolve())
    try:
        base = resolve_base(git, args.base)
        committed = changed_files(git, base)
        every_path = committed + staged_files(git)
        problems = []
        problems += check_frozen(every_path)
        problems += check_large(git, base, every_path)
        problems += check_hygiene(every_path + untracked_files(git))
        problems += check_messages(git, base)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f'FAILED guard check: {error}', file=sys.stderr)
        return 1
    if problems:
        print(f'{len(problems)} guard violation(s) against {base[:12]}:')
        for problem in problems:
            print(f'  {problem}')
        return 1
    print(f'Guards OK against {base[:12]}: frozen paths, {MAX_BYTES // 1024} KB files, '
          f'pack hygiene, commit messages.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
