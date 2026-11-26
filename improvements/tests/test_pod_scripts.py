"""J26: pod_setup.sh and pack_sync.sh in temporary git repos (no pod, no network)."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import frozen  # noqa: F401

PROJECT = Path(__file__).resolve().parents[2]
POD_SETUP = PROJECT / "scripts/pod_setup.sh"
PACK_SYNC = PROJECT / "scripts/pack_sync.sh"
EXCLUDE = ["docs/prompts/", "docs/jobs/", "docs/ledger/", "docs/research/", "AGENTS.md", "CLAUDE.md"]


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                          check=True, env=dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null"))


def make_repo(path, exclude=()):
    path.mkdir(parents=True)
    git(path, "init", "-q")
    info = path / ".git/info/exclude"
    info.write_text("# git default comment\n" + "".join(f"{line}\n" for line in exclude))
    return path


def run(script, *args, env=None):
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          timeout=60, env=dict(os.environ, **(env or {})))


class PodSetupTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = make_repo(Path(folder.name) / "pod")
        shutil.copy(PROJECT / "config.example.json", self.root / "config.example.json")
        (self.root / ".gitignore").write_text("config.local.json\n")
        self.env = {"POD_SETUP_ROOT": str(self.root)}

    def test_dry_run_reports_missing_items_and_writes_nothing(self):
        result = run(POD_SETUP, "--dry-run", "--write-config", env=self.env)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("MISSING  config.local.json", result.stdout)
        self.assertIn("MISSING  pack", result.stdout)
        self.assertFalse((self.root / "config.local.json").exists())

    def test_write_config_creates_once_and_never_overwrites(self):
        run(POD_SETUP, "--write-config", env=self.env)
        config = self.root / "config.local.json"
        self.assertEqual(config.read_text(), (self.root / "config.example.json").read_text())
        config.write_text('{"edited": true}')
        second = run(POD_SETUP, "--write-config", env=self.env)
        self.assertIn("config.local.json exists (kept as is)", second.stdout)
        self.assertEqual(config.read_text(), '{"edited": true}')

    def test_a_tracked_config_is_refused(self):
        (self.root / "config.local.json").write_text("{}")
        git(self.root, "add", "-f", "config.local.json")
        result = run(POD_SETUP, env=self.env)
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSED", result.stderr)

    def test_teardown_finds_your_gpu_processes(self):
        stub = self.root / "bin"
        stub.mkdir()
        smi = stub / "nvidia-smi"
        smi.write_text(f"#!/bin/sh\necho {os.getpid()}\n")
        smi.chmod(0o755)
        path = {"PATH": f"{stub}:{os.environ['PATH']}"}
        result = run(POD_SETUP, "--teardown", env={**self.env, **path})
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"LEFT     your GPU processes: {os.getpid()}(", result.stdout)
        smi.write_text("#!/bin/sh\n")
        self.assertEqual(run(POD_SETUP, "--teardown", env={**self.env, **path}).returncode, 0)

    def test_an_unknown_option_exits_2(self):
        self.assertEqual(run(POD_SETUP, "--force", env=self.env).returncode, 2)


class PackSyncTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        base = Path(folder.name)
        self.pc = make_repo(base / "pc", EXCLUDE)
        for name in ("AGENTS.md", "CLAUDE.md", "docs/jobs/E00.md", "docs/prompts/E00.md",
                     "docs/ledger/jobs.md", "docs/research/E00.md"):
            (self.pc / name).parent.mkdir(parents=True, exist_ok=True)
            (self.pc / name).write_text(f"pc {name}\n")
        self.server = make_repo(base / "server")
        self.env = {"PACK_SYNC_ROOT": str(self.pc)}

    def push(self):
        return run(PACK_SYNC, "push", str(self.server), env=self.env)

    def test_push_adds_the_exclude_block_once_and_copies_the_pack(self):
        self.assertEqual(self.push().returncode, 0)
        self.assertEqual(self.push().returncode, 0)
        lines = (self.server / ".git/info/exclude").read_text().splitlines()
        self.assertEqual([line for line in lines if line in EXCLUDE], EXCLUDE)
        self.assertEqual((self.server / "docs/jobs/E00.md").read_text(), "pc docs/jobs/E00.md\n")
        self.assertEqual(git(self.server, "status", "--porcelain").stdout, "")

    def test_push_stops_when_pack_files_are_staged_on_the_server(self):
        (self.server / "AGENTS.md").write_text("server\n")
        git(self.server, "add", "AGENTS.md")
        result = self.push()
        self.assertEqual(result.returncode, 1)
        self.assertIn("staged", result.stderr)
        self.assertEqual((self.server / "AGENTS.md").read_text(), "server\n")

    def test_push_stops_when_the_pc_has_no_exclude_block(self):
        (self.pc / ".git/info/exclude").write_text("# nothing\n")
        result = self.push()
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not ignore the pack", result.stderr)

    def test_pull_copies_only_status_folders_and_keeps_newer_local_files(self):
        self.assertEqual(self.push().returncode, 0)
        (self.server / "docs/jobs/E01.md").write_text("server E01\n")
        (self.server / "docs/ledger/inbox").mkdir(parents=True)
        (self.server / "docs/ledger/inbox/E00.md").write_text("inbox\n")
        (self.server / "docs/ledger/jobs.md").write_text("server ledger\n")
        newer = self.pc / "docs/jobs/E00.md"
        (self.server / "docs/jobs/E00.md").write_text("older server copy\n")
        os.utime(self.server / "docs/jobs/E00.md", (1_000_000, 1_000_000))
        newer.write_text("fresh PC status\n")
        result = run(PACK_SYNC, "pull", str(self.server), env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.pc / "docs/jobs/E01.md").read_text(), "server E01\n")
        self.assertEqual((self.pc / "docs/ledger/inbox/E00.md").read_text(), "inbox\n")
        self.assertEqual(newer.read_text(), "fresh PC status\n")
        self.assertEqual((self.pc / "docs/ledger/jobs.md").read_text(), "pc docs/ledger/jobs.md\n")

    def test_bad_usage_exits_2(self):
        self.assertEqual(run(PACK_SYNC, "push", env=self.env).returncode, 2)
        self.assertEqual(run(PACK_SYNC, "sideways", str(self.server), env=self.env).returncode, 2)


if __name__ == "__main__":
    unittest.main()
