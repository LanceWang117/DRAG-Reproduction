"""Prove the code-limit checker rejects each rule it enforces."""
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from check_code_limits import (  # noqa: E402
    MAX_COMPLEXITY, MAX_FILE_LINES, MAX_FUNCTION_LINES, check_file, complexity, measure,
    run,
)


def body(lines):
    return '\n'.join(f'    x{i} = {i}' for i in range(lines))


def function_of(lines):
    """A function whose total length, comments included, is exactly ``lines``."""
    return f'def target():\n{body(lines - 1)}\n'


class FunctionLengthTests(unittest.TestCase):
    def test_the_limit_itself_passes(self):
        found = self.check(function_of(MAX_FUNCTION_LINES), allowlist={})
        self.assertEqual(found, [])

    def test_one_line_over_the_limit_fails(self):
        found = self.check(function_of(MAX_FUNCTION_LINES + 1), allowlist={})
        self.assertEqual(len(found), 1)
        self.assertIn('function lines', found[0].rule)
        self.assertEqual(found[0].measured, MAX_FUNCTION_LINES + 1)

    def check(self, source, allowlist):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text(textwrap.dedent(source))
            return check_file(path, 'sample.py', allowlist)


class FileLengthTests(unittest.TestCase):
    def test_a_file_at_the_limit_passes(self):
        source = '\n'.join('x = 1' for _ in range(MAX_FILE_LINES)) + '\n'
        self.assertEqual(self.check(source), [])

    def test_a_file_one_line_over_the_limit_fails(self):
        source = '\n'.join('x = 1' for _ in range(MAX_FILE_LINES + 1)) + '\n'
        found = self.check(source)
        self.assertEqual([violation.rule for violation in found], ['file lines'])

    def check(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text(source)
            return check_file(path, 'sample.py', {})


class ComplexityTests(unittest.TestCase):
    def test_a_single_branch_passes(self):
        source = 'def target(flag):\n    if flag:\n        return 1\n    return 0\n'
        self.assertEqual(self.complexity_of(source), 2)

    def test_ten_branches_pass_and_eleven_fail(self):
        self.assertEqual(self.complexity_of(self.branches(MAX_COMPLEXITY - 1)), MAX_COMPLEXITY)
        self.assertEqual(self.complexity_of(self.branches(MAX_COMPLEXITY)), MAX_COMPLEXITY + 1)

    def test_a_nested_function_is_counted_separately(self):
        source = (
            'def outer(flag):\n'
            '    def inner():\n'
            '        if flag:\n'
            '            return 1\n'
            '        return 0\n'
            '    return inner\n'
        )
        measured = self.measure(source)[1]
        self.assertEqual(measured['outer.inner'], (4, 2))
        self.assertEqual(measured['outer'], (6, 2))

    def branches(self, count):
        lines = ['def target(value):']
        lines.extend(f'    if value == {index}:\n        return {index}' for index in range(count))
        lines.append('    return -1')
        return '\n'.join(lines) + '\n'

    def complexity_of(self, source):
        return self.measure(source)[1]['target'][1]

    def measure(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text(source)
            return measure(path, 'sample.py')


class AllowlistTests(unittest.TestCase):
    def test_a_grandfathered_function_may_keep_its_size(self):
        found = self.check(grown(0), {'sample.py': {'target': record(MAX_FUNCTION_LINES + 1)}})
        self.assertEqual(found, [])

    def test_a_grandfathered_function_may_shrink(self):
        found = self.check(grown(-1), {'sample.py': {'target': record(MAX_FUNCTION_LINES + 2)}})
        self.assertEqual(found, [])

    def test_a_grandfathered_function_may_not_grow(self):
        found = self.check(grown(0), {'sample.py': {'target': record(MAX_FUNCTION_LINES)}})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].rule, 'allowed size')

    def check(self, source, allowlist):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text(source)
            return check_file(path, 'sample.py', allowlist)


def record(lines):
    return {'lines': lines, 'complexity': 1}


def grown(extra):
    return function_of(MAX_FUNCTION_LINES + 1 + extra)


class RepositoryTests(unittest.TestCase):
    """The checker must accept this repository as committed."""

    def test_the_repository_passes_its_own_limits(self):
        root = Path(__file__).resolve().parents[2]
        allowlist = root / 'scripts/code_limits_allowlist.json'
        self.assertEqual(run(root, allowlist), [])

    def test_the_allowlist_is_valid_json_with_sizes(self):
        root = Path(__file__).resolve().parents[2]
        entries = json.loads((root / 'scripts/code_limits_allowlist.json').read_text())
        for path, functions in entries.items():
            self.assertTrue(path.endswith('.py'), path)
            for name, entry in functions.items():
                self.assertIn('lines', entry, name)
                self.assertIn('complexity', entry, name)

    def test_frozen_paths_are_excluded(self):
        root = Path(__file__).resolve().parents[2]
        watched = {display for _, display in
                   __import__('check_code_limits').candidate_files(root)}
        self.assertFalse([path for path in watched if path.startswith('src/original_drag/')])
        self.assertFalse([path for path in watched if path.startswith('src/reproduction_changes/')])


if __name__ == '__main__':
    unittest.main()
