"""Enforce the code-size limits from the architecture ledger (section 6).

Three rules, one AST pass, no third-party dependency:

* a function is at most 40 lines, counted from its ``def`` line to its last body
  line inclusive;
* a file is at most 250 lines;
* a function's cyclomatic complexity is at most 10, counted as one plus the
  decision statements in its own body (nested ``def`` blocks are separate
  functions and add one each), which reproduces ``mccabe``'s numbers.

``allowlist`` grandfathers code that was already over a limit before the checker
existed. An entry records the measured size and may only shrink: a function that
grows past its recorded size fails.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_FUNCTION_LINES = 40
MAX_FILE_LINES = 250
MAX_COMPLEXITY = 10
DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / 'code_limits_allowlist.json'
EXCLUDED_DIRS = ('src/original_drag/', 'src/reproduction_changes/', 'results/')
DECISIONS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Violation:
    path: str
    name: str
    rule: str
    measured: int
    limit: int

    def __str__(self) -> str:
        return f'{self.path}::{self.name}: {self.rule} {self.measured} > {self.limit}'


def _own_body(function):
    """Yield the nodes in a function's own body, not a nested function's."""
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
    for node in _own_body(function):
        if isinstance(node, DECISIONS):
            score += 1
        elif isinstance(node, FUNCTIONS):
            score += 1
    return score


def function_lines(function) -> int:
    return function.end_lineno - function.lineno + 1


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, FUNCTIONS):
            yield node


def qualnames(tree):
    """Map each function to ``Class::method``, descending only through classes."""
    found = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f'{prefix}{child.name}::')
            elif isinstance(child, FUNCTIONS):
                found[child] = f'{prefix}{child.name}'
                walk(child, f'{prefix}{child.name}.')

    walk(tree, '')
    return found


def measure(path: Path, display: str):
    """Return ``(file_lines, {qualname: (lines, complexity)})`` for one file."""
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=display)
    names = qualnames(tree)
    return source.count('\n'), {names[f]: (function_lines(f), complexity(f)) for f in names}


def _allowed(entry, lines, cx):
    """An allowlisted function may shrink but never grow past its record."""
    return lines <= entry['lines'] and cx <= entry['complexity']


def check_file(path: Path, display: str, allowlist) -> list[Violation]:
    file_lines, functions = measure(path, display)
    found = []
    if file_lines > MAX_FILE_LINES and allowlist.get(display, {}).get('__file__') is None:
        found.append(Violation(display, '<file>', 'file lines', file_lines, MAX_FILE_LINES))
    entries = allowlist.get(display, {})
    for name, (lines, cx) in sorted(functions.items()):
        entry = entries.get(name)
        if entry is not None:
            if not _allowed(entry, lines, cx):
                found.append(Violation(display, name, 'allowed size', max(lines, cx),
                                       max(entry['lines'], entry['complexity'])))
            continue
        if lines > MAX_FUNCTION_LINES:
            found.append(Violation(display, name, 'function lines', lines, MAX_FUNCTION_LINES))
        if cx > MAX_COMPLEXITY:
            found.append(Violation(display, name, 'complexity', cx, MAX_COMPLEXITY))
    return found


def candidate_files(root: Path):
    """Watched Python files: everything tracked, minus frozen paths and caches."""
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        if any(part in relative for part in ('__pycache__/', '.venv/', '.runtime/')):
            continue
        if relative.startswith(EXCLUDED_DIRS):
            continue
        yield path, relative


def load_allowlist(path: Path = DEFAULT_ALLOWLIST) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def run(root: Path, allowlist_path: Path = DEFAULT_ALLOWLIST) -> list[Violation]:
    allowlist = load_allowlist(allowlist_path)
    violations = []
    for path, display in candidate_files(root):
        violations.extend(check_file(path, display, allowlist))
    return violations


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    allowlist = Path(argv[1]) if len(argv) > 1 else DEFAULT_ALLOWLIST
    violations = run(root, allowlist)
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
