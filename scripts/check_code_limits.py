"""Enforce the code-size limits from the architecture ledger (section 6).

Three rules, one AST pass, no third-party dependency:

* a function is at most 40 lines, counted from its first decorator (or its
  ``def``) to its last body line inclusive;
* a file is at most 250 lines;
* a function's cyclomatic complexity is at most 10, counted as one plus each
  ``if``, loop, ``try``, ``except`` and nested ``def`` in its own body. That is
  ``mccabe``'s counting, verified against ``mccabe==0.7.0``.

``allowlist`` grandfathers code that was already over a limit before the checker
existed. An entry records the measured size and may only shrink: a function that
grows past its recorded size fails.

Every entry point fails closed. A root that is not a repository, an unparsable
file or a malformed allowlist is an error, never a silent pass.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_FUNCTION_LINES = 40
MAX_FILE_LINES = 250
MAX_COMPLEXITY = 10
DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / 'code_limits_allowlist.json'
EXCLUDED_PREFIXES = ('src/original_drag/', 'src/reproduction_changes/', 'results/')
DECISIONS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
SKIP_DIRS = ('__pycache__', '.venv', '.runtime', '.git')


@dataclass(frozen=True)
class Violation:
    path: str
    name: str
    rule: str
    measured: int
    limit: int

    def __str__(self) -> str:
        return f'{self.path}::{self.name}: {self.rule} {self.measured} > {self.limit}'


def _own_nodes(function):
    """Nodes in a function's own body; a nested function is its own scope."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, FUNCTIONS):
            continue
        stack.extend(ast.iter_child_nodes(node))


def complexity(function) -> int:
    """McCabe complexity of one function, matching ``mccabe``'s counting."""
    score = 1
    for node in _own_nodes(function):
        if isinstance(node, DECISIONS):
            score += 1
        elif isinstance(node, FUNCTIONS):
            score += 1
    return score


def function_lines(function) -> int:
    start = function.lineno
    if function.decorator_list:
        start = min([start] + [item.lineno for item in function.decorator_list])
    return function.end_lineno - start + 1


def collect(tree):
    """Map every function to ``Class::method`` (nested: ``outer.inner``).

    Descends through classes and through statements such as ``if`` and ``try``,
    so a definition guarded by ``try/except ImportError`` is still measured.
    """
    found = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{prefix}{child.name}::')
            elif isinstance(child, FUNCTIONS):
                found.setdefault(f'{prefix}{child.name}', []).append(child)
                visit(child, f'{prefix}{child.name}.')
            else:
                visit(child, prefix)

    visit(tree, '')
    return found


def measure(path: Path, display: str):
    """Return ``(file_lines, {qualname: [(lines, complexity), ...]})``."""
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=display)
    functions = collect(tree)
    measured = {name: [(function_lines(f), complexity(f)) for f in nodes]
                for name, nodes in functions.items()}
    return len(source.splitlines()), measured


def _allowed(entry, lines, cx):
    """An allowlisted function may shrink but never grow past its record."""
    return lines <= entry['lines'] and cx <= entry['complexity']


def _allowlist_violation(display, name, entry, sizes):
    lines = max(size[0] for size in sizes)
    if lines > entry['lines']:
        return Violation(display, name, 'allowed lines', lines, entry['lines'])
    cx = max(size[1] for size in sizes)
    if cx > entry['complexity']:
        return Violation(display, name, 'allowed complexity', cx, entry['complexity'])
    return None


def check_file(path: Path, display: str, allowlist) -> list[Violation]:
    file_lines, functions = measure(path, display)
    found = []
    if file_lines > MAX_FILE_LINES:
        found.append(Violation(display, '<file>', 'file lines', file_lines, MAX_FILE_LINES))
    entries = allowlist.get(display, {})
    for name, sizes in sorted(functions.items()):
        entry = entries.get(name)
        if entry is not None:
            violation = _allowlist_violation(display, name, entry, sizes)
            if violation is not None:
                found.append(violation)
            continue
        for lines, cx in sizes:
            if lines > MAX_FUNCTION_LINES:
                found.append(Violation(display, name, 'function lines', lines,
                                       MAX_FUNCTION_LINES))
            if cx > MAX_COMPLEXITY:
                found.append(Violation(display, name, 'complexity', cx, MAX_COMPLEXITY))
    return found


def candidate_files(root: Path):
    """Watched Python files: tracked plus untracked-but-not-ignored.

    Asking git keeps the list identical on every machine: a Conda tree or a
    downloaded package under an ignored directory is never scanned.
    """
    command = ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
               '--', '*.py']
    output = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    seen = set()
    for relative in output.stdout.split('\0'):
        if not relative or relative in seen:
            continue
        seen.add(relative)
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        if any(f'/{part}/' in f'/{relative}' for part in SKIP_DIRS):
            continue
        path = root / relative
        if path.is_file():
            yield path, relative


def load_allowlist(path: Path = DEFAULT_ALLOWLIST) -> dict:
    if not path.is_file():
        return {}
    entries = json.loads(path.read_text(encoding='utf-8'))
    for display, functions in entries.items():
        if not isinstance(functions, dict) or display == '__file__':
            raise ValueError(f'allowlist entry for {display} has no functions')
        for name, entry in functions.items():
            if not {'lines', 'complexity'} <= set(entry):
                raise ValueError(f'allowlist entry {display}::{name} needs lines and complexity')
    return entries


def run(root: Path, allowlist_path: Path = DEFAULT_ALLOWLIST) -> list[Violation]:
    if not (root / '.git').exists():
        raise ValueError(f'{root} is not a git repository; the checker needs git ls-files')
    allowlist = load_allowlist(allowlist_path)
    violations = []
    for path, display in candidate_files(root):
        violations.extend(check_file(path, display, allowlist))
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--allowlist', type=Path, default=DEFAULT_ALLOWLIST)
    args = parser.parse_args(argv)
    try:
        violations = run(args.root.resolve(), args.allowlist)
    except (ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f'FAILED code-limit check: {error}', file=sys.stderr)
        return 1
    if violations:
        print(f'{len(violations)} code-limit violation(s):')
        for violation in violations:
            print(f'  {violation}')
        return 1
    print(f'Code limits OK: functions <= {MAX_FUNCTION_LINES} lines, files <= '
          f'{MAX_FILE_LINES} lines, complexity <= {MAX_COMPLEXITY}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
