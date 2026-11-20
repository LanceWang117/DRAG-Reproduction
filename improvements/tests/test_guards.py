"""Prove the repository guards reject each thing they exist to stop.

Each test builds a throwaway git repository, commits a base state, then makes one
forbidden change and asserts the guard reports it. They also pin the cases that
must stay quiet, so a guard cannot pass by flagging everything.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from check_guards import (  # noqa: E402
    Git, MAX_BYTES, check_frozen, check_hygiene, check_large, check_main_history,
    check_messages, changed_files, resolve_base,
)


class Repository:
    """A minimal git repository for one guard scenario."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.git = Git(self.root)
        self.run('init', '-q', '-b', 'main')
        self.run('config', 'user.name', 'Guard Test')
        self.run('config', 'user.email', 'guard@example.com')

    def run(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout

    def write(self, path, content='x\n'):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def commit(self, message):
        self.run('add', '-A')
        self.run('commit', '-q', '-m', message)
        return self.git.head()


class GuardTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.repo = Repository(self._directory.name)
        self.repo.write('keep.txt')
        self.base = self.repo.commit('base')

    def tearDown(self):
        self._directory.cleanup()

    def problems(self):
        changed = changed_files(self.repo.git, self.base)
        return (check_frozen(changed)
                + check_large(self.repo.git, self.base, changed)
                + check_hygiene(changed)
                + check_messages(self.repo.git, self.base))


class FrozenPathTests(GuardTestCase):
    def test_a_modified_frozen_file_is_reported(self):
        self.repo.write('src/original_drag/model/drag.py', 'changed\n')
        self.repo.commit('edit frozen')
        self.assertEqual(len(self.problems()), 1)
        self.assertIn('frozen path changed', self.problems()[0])

    def test_a_deleted_frozen_file_is_reported(self):
        self.repo.write('results/summary.csv')
        self.base = self.repo.commit('add evidence')
        (self.repo.root / 'results/summary.csv').unlink()
        self.repo.commit('delete frozen')
        changed = changed_files(self.repo.git, self.base)
        self.assertIn('results/summary.csv', changed)
        self.assertEqual(check_frozen(changed), ['results/summary.csv: frozen path changed'])

    def test_a_frozen_file_moved_out_is_reported(self):
        self.repo.write('src/original_drag/keep.py')
        self.base = self.repo.commit('add frozen')
        self.repo.run('mv', 'src/original_drag/keep.py', 'moved.py')
        self.repo.commit('move out of frozen')
        changed = changed_files(self.repo.git, self.base)
        self.assertIn('src/original_drag/keep.py', changed)
        self.assertTrue(check_frozen(changed))

    def test_an_unrelated_fix_is_not_reported(self):
        self.repo.write('scripts/fix.py', 'x = 1\n')
        self.repo.commit('normal change')
        self.assertEqual(check_frozen(changed_files(self.repo.git, self.base)), [])


class LargeFileTests(GuardTestCase):
    def test_a_new_file_over_one_megabyte_is_reported(self):
        self.repo.write('big.bin', 'x' * (MAX_BYTES + 1))
        self.repo.commit('add big file')
        problems = check_large(self.repo.git, self.base,
                              changed_files(self.repo.git, self.base))
        self.assertEqual(len(problems), 1)
        self.assertIn('> 1 MB', problems[0])

    def test_a_file_at_the_limit_is_allowed(self):
        self.repo.write('exact.bin', 'x' * MAX_BYTES)
        self.repo.commit('add exact file')
        self.assertEqual(check_large(self.repo.git, self.base,
                                     changed_files(self.repo.git, self.base)), [])

    def test_a_pre_existing_large_file_stays_grandfathered_if_untouched(self):
        self.repo.write('legacy.bin', 'x' * (MAX_BYTES + 10))
        self.base = self.repo.commit('legacy large file')
        self.repo.write('small.txt')
        self.repo.commit('touch something else')
        changed = changed_files(self.repo.git, self.base)
        self.assertEqual(check_large(self.repo.git, self.base, changed), [])


class HygieneTests(GuardTestCase):
    def test_each_pack_and_agent_path_is_reported(self):
        for path in ('AGENTS.md', 'CLAUDE.md', 'docs/prompts/x.md', 'docs/ledger/y.md',
                     '.agent/tool', '.cursor/rules', '.aider.conf.yml', 'GEMINI.md'):
            with self.subTest(path=path):
                self.assertTrue(check_hygiene([path]), path)

    def test_an_ordinary_path_is_not_reported(self):
        self.assertEqual(check_hygiene(['scripts/gate.sh', 'docs/README.md']), [])

    def test_an_attribution_trailer_is_reported(self):
        self.repo.write('note.txt')
        self.repo.commit('change\n\nCo-Authored-By: Someone <a@example.com>')
        problems = check_messages(self.repo.git, self.base)
        self.assertEqual(len(problems), 1)
        self.assertIn('co-authored-by:', problems[0])

    def test_a_clean_message_prefixes_the_short_sha(self):
        self.repo.write('note.txt')
        self.repo.commit('a plain message')
        self.assertEqual(check_messages(self.repo.git, self.base), [])

    def test_prose_about_trailers_is_not_an_attribution(self):
        self.repo.write('note.txt')
        self.repo.commit('docs: explain that we never add a co-authored-by trailer')
        self.assertEqual(check_messages(self.repo.git, self.base), [])


class BaseResolutionTests(GuardTestCase):
    def test_the_base_is_never_the_head_itself(self):
        self.repo.run('update-ref', 'refs/remotes/origin/main', self.repo.git.head())
        self.repo.write('after.txt')
        self.repo.commit('after base')
        base = resolve_base(self.repo.git, None)
        self.assertNotEqual(base, self.repo.git.head())
        self.assertTrue(changed_files(self.repo.git, base))

    def test_an_unresolvable_explicit_base_is_an_error(self):
        with self.assertRaises(ValueError):
            resolve_base(self.repo.git, 'no-such-ref')

    def test_an_all_zero_before_commit_falls_back(self):
        # A first push reports an all-zero "before" commit for GUARD_BASE. The
        # repository here has one commit and no parent, so there is nothing to
        # compare and the guards must stay quiet rather than fail CI.
        before = '0' * 40
        with unittest.mock.patch.dict(os.environ, {'GUARD_BASE': before}):
            base = resolve_base(self.repo.git, None)
        self.assertEqual(base, '')
        self.assertEqual(changed_files(self.repo.git, base), [])
        self.assertEqual(check_main_history(self.repo.git, base, enabled=True), [])


class MainMergePolicyTests(GuardTestCase):
    """Nothing may reach main except through a pull request.

    Branch protection is unavailable to this team (no admin rights, and the repo
    owner cannot enable it on their plan), so the gate carries this rule instead.
    """

    def test_the_policy_is_off_unless_requested(self):
        self.repo.write('direct.txt')
        self.repo.commit('direct push to main')
        self.assertEqual(
            check_main_history(self.repo.git, self.base, enabled=False), [])

    def test_a_direct_commit_is_reported(self):
        self.repo.write('direct.txt')
        self.repo.commit('direct push to main')
        problems = check_main_history(self.repo.git, self.base, enabled=True)
        self.assertEqual(len(problems), 1)
        self.assertIn('not from a pull request', problems[0])

    def test_a_squash_merge_is_accepted(self):
        self.repo.write('feature.txt')
        self.repo.commit('J01: add analysis toolkit (#7)')
        self.assertEqual(
            check_main_history(self.repo.git, self.base, enabled=True), [])

    def test_a_merge_commit_is_accepted(self):
        self.repo.write('feature.txt')
        self.repo.commit('Merge pull request #8 from team/job-J01')
        self.assertEqual(
            check_main_history(self.repo.git, self.base, enabled=True), [])

    def test_every_commit_in_the_push_is_checked(self):
        self.repo.write('one.txt')
        self.repo.commit('J01: good step (#9)')
        self.repo.write('two.txt')
        self.repo.commit('sneaky direct commit')
        problems = check_main_history(self.repo.git, self.base, enabled=True)
        self.assertEqual(len(problems), 1)
        self.assertIn('sneaky direct commit', problems[0])


if __name__ == '__main__':
    unittest.main()
