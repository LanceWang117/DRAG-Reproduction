"""Run a unittest suite strictly: no skips, and a per-test time budget.

``unittest`` counts a skipped test as a success, so a suite that silently
skipped everything still reports OK. This harness fails instead, and it also
fails any single test that takes longer than ``--max-seconds``.

Usage:
    python scripts/run_tests.py improvements/tests --max-seconds 10
    python scripts/run_tests.py src/reproduction_changes/tests --pattern test_*.py
"""
from __future__ import annotations

import argparse
import sys
import time
import unittest
from pathlib import Path


class TimedResult(unittest.TextTestResult):
    """Record per-test durations and reject anything over the budget."""

    def __init__(self, *args, max_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seconds = max_seconds
        self.slow = []

    def startTest(self, test):
        self._started = time.monotonic()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.monotonic() - self._started
        if elapsed > self.max_seconds:
            self.slow.append((test.id(), elapsed))
        super().stopTest(test)


def build_suite(paths, pattern):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in paths:
        target = str(path)
        suite.addTests(loader.discover(target, pattern=pattern))
    return suite


def run(paths, pattern, max_seconds, stream):
    suite = build_suite(paths, pattern)
    runner = unittest.TextTestRunner(stream=stream, verbosity=2,
                                     resultclass=_result_factory(max_seconds))
    result = runner.run(suite)
    return result


def _result_factory(max_seconds):
    class BoundResult(TimedResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, max_seconds=max_seconds, **kwargs)

    return BoundResult


def report(result, stream):
    """Return the process exit code, explaining every failure mode."""
    failures = []
    if result.failures or result.errors:
        failures.append(f'{len(result.failures)} failure(s), {len(result.errors)} error(s)')
    if result.skipped:
        names = ', '.join(test.id() for test, _ in result.skipped)
        failures.append(f'{len(result.skipped)} skipped test(s): {names}')
    if result.unexpectedSuccesses:
        failures.append(f'{len(result.unexpectedSuccesses)} unexpected success(es)')
    if result.slow:
        detail = ', '.join(f'{name} {seconds:.1f}s' for name, seconds in result.slow)
        failures.append(f'{len(result.slow)} test(s) over {result.max_seconds}s: {detail}')
    for line in failures:
        print(f'FAIL: {line}', file=stream)
    if failures:
        return 1
    print(f'OK: {result.testsRun} tests, 0 skipped, none over {result.max_seconds}s.',
          file=stream)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--max-seconds', type=float, default=10.0)
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.paths if not path.is_dir()]
    if missing:
        print(f'FAIL: not a directory: {", ".join(missing)}', file=sys.stderr)
        return 1
    result = run(args.paths, args.pattern, args.max_seconds, sys.stderr)
    return report(result, sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
