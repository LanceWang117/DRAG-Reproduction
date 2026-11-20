"""Prove the code-limit checker rejects each rule it enforces."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from check_code_limits import (  # noqa: E402
    MAX_COMPLEXITY, MAX_FILE_LINES, MAX_FUNCTION_LINES, check_file, measure, run,
)


def body(lines):
    return '\n'.join(f'    x{i} = {i}' for i in range(lines))


def function_of(lines):
    """A function whose total length, decorators included, is exactly ``lines``."""
    return f'def target():\n{body(lines - 1)}\n'


def measure_source(source):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'sample.py'
        path.write_text(textwrap.dedent(source))
        return measure(path, 'sample.py')


def check_source(source, allowlist=None):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'sample.py'
        path.write_text(textwrap.dedent(source))
        return check_file(path, 'sample.py', allowlist or {})


def record(lines, cx=1):
    return {'lines': lines, 'complexity': cx}


class FunctionLengthTests(unittest.TestCase):
    def test_the_limit_itself_passes(self):
        self.assertEqual(check_source(function_of(MAX_FUNCTION_LINES)), [])

    def test_one_line_over_the_limit_fails(self):
        found = check_source(function_of(MAX_FUNCTION_LINES + 1))
        self.assertEqual([violation.rule for violation in found], ['function lines'])
        self.assertEqual(found[0].measured, MAX_FUNCTION_LINES + 1)

    def test_decorators_count_towards_the_length(self):
        source = '@decorator\n' * (MAX_FUNCTION_LINES - 1) + 'def target():\n    return 1\n'
        self.assertEqual([violation.rule for violation in check_source(source)],
                         ['function lines'])

    def test_a_definition_inside_try_is_still_measured(self):
        source = ('try:\n    import x\nexcept ImportError:\n    x = None\n\n'
                  + function_of(MAX_FUNCTION_LINES + 1))
        self.assertEqual([violation.rule for violation in check_source(source)],
                         ['function lines'])

    def test_an_earlier_duplicate_definition_is_still_measured(self):
        duplicate = function_of(MAX_FUNCTION_LINES + 1) + '\ndef target():\n    return 1\n'
        self.assertEqual([violation.rule for violation in check_source(duplicate)],
                         ['function lines'])


class FileLengthTests(unittest.TestCase):
    def test_a_file_at_the_limit_passes(self):
        source = '\n'.join('x = 1' for _ in range(MAX_FILE_LINES)) + '\n'
        self.assertEqual(check_source(source), [])

    def test_a_file_one_line_over_the_limit_fails(self):
        source = '\n'.join('x = 1' for _ in range(MAX_FILE_LINES + 1)) + '\n'
        self.assertEqual([violation.rule for violation in check_source(source)],
                         ['file lines'])

    def test_a_last_line_without_a_newline_still_counts(self):
        source = '\n'.join('x = 1' for _ in range(MAX_FILE_LINES + 1))
        self.assertEqual([violation.rule for violation in check_source(source)],
                         ['file lines'])


class ComplexityTests(unittest.TestCase):
    def test_a_single_branch_passes(self):
        source = 'def target(flag):\n    if flag:\n        return 1\n    return 0\n'
        self.assertEqual(self.complexity_of(source), 2)

    def test_ten_branches_pass_and_eleven_fail(self):
        passing = self.complexity_of(self.branches(MAX_COMPLEXITY - 1))
        failing = self.complexity_of(self.branches(MAX_COMPLEXITY))
        self.assertEqual((passing, failing), (MAX_COMPLEXITY, MAX_COMPLEXITY + 1))
        found = check_source(self.branches(MAX_COMPLEXITY))
        self.assertEqual([violation.rule for violation in found], ['complexity'])

    def test_a_try_block_counts_like_mccabe(self):
        # 1 base + 2 try + 2 except + 9 if = 14, the number mccabe reports.
        source = ('def target(value):\n'
                  '    try:\n        value += 1\n    except ValueError:\n        pass\n'
                  '    try:\n        value += 1\n    except KeyError:\n        pass\n'
                  + ''.join(f'    if value == {i}:\n        return {i}\n' for i in range(9))
                  + '    return -1\n')
        self.assertEqual(self.complexity_of(source), 14)

    def test_a_nested_function_is_counted_separately(self):
        source = (
            'def outer(flag):\n'
            '    def inner():\n'
            '        if flag:\n'
            '            return 1\n'
            '        return 0\n'
            '    return inner\n'
        )
        measured = measure_source(source)[1]
        self.assertEqual(measured['outer.inner'][0], (4, 2))
        self.assertEqual(measured['outer'][0], (6, 2))

    def branches(self, count):
        lines = ['def target(value):']
        lines.extend(f'    if value == {index}:\n        return {index}' for index in range(count))
        lines.append('    return -1')
        return '\n'.join(lines) + '\n'

    def complexity_of(self, source):
        return measure_source(source)[1]['target'][0][1]


class AllowlistTests(unittest.TestCase):
    def test_a_grandfathered_function_may_keep_its_size(self):
        allowlist = {'sample.py': {'target': record(MAX_FUNCTION_LINES + 1)}}
        self.assertEqual(check_source(function_of(MAX_FUNCTION_LINES + 1), allowlist), [])

    def test_a_grandfathered_function_may_shrink(self):
        allowlist = {'sample.py': {'target': record(MAX_FUNCTION_LINES + 2)}}
        self.assertEqual(check_source(function_of(MAX_FUNCTION_LINES + 1), allowlist), [])

    def test_a_grandfathered_function_may_not_grow(self):
        allowlist = {'sample.py': {'target': record(MAX_FUNCTION_LINES)}}
        found = check_source(function_of(MAX_FUNCTION_LINES + 1), allowlist)
        self.assertEqual([violation.rule for violation in found], ['allowed lines'])


class FailClosedTests(unittest.TestCase):
    def test_a_directory_that_is_not_a_repository_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run(Path(directory), Path(directory) / 'missing.json')

    def test_an_unknown_flag_is_an_error(self):
        script = Path(__file__).resolve().parents[2] / 'scripts/check_code_limits.py'
        completed = subprocess.run([sys.executable, str(script), '--nonsense'],
                                   capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)


class RepositoryTests(unittest.TestCase):
    """The checker must accept this repository as committed."""

    def test_the_repository_passes_its_own_limits(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(run(root, root / 'scripts/code_limits_allowlist.json'), [])

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
