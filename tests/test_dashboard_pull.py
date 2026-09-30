import os
import sys
import time

import pytest

from psu import config
from psu.dashboard import pull


def _settings(tmp_path, api_key="k"):
    return config.Settings(
        api_key=api_key, raw_dir=tmp_path / "raw", db_path=tmp_path / "psu.duckdb", current_season=2026
    )


class FakeProc:
    def __init__(self, lines, code):
        self.stdout = iter(lines)
        self.code = code
        self.running = True
        self.terminated = False

    def poll(self):
        return None if self.running else self.code

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.running = False

    def wait(self, timeout=None):
        self.running = False
        return self.code


def _popen(lines, code, calls=None, procs=None):
    def factory(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        proc = FakeProc(lines, code)
        if procs is not None:
            procs.append(proc)
        return proc

    return factory


def test_refresh_command():
    cmd = pull.refresh_command()
    assert cmd[0] == sys.executable
    assert cmd[-2:] == ["--max-calls", "60"]


def test_streams_lines_and_returns_code(tmp_path):
    s, got, calls = _settings(tmp_path), [], []
    assert pull.run_refresh(s, got.append, popen=_popen(["a\n", "b  \n"], 3, calls)) == 3
    assert got == ["a", "b"]
    kw = calls[0][1]
    assert kw["text"] is True and kw["cwd"] == config.PROJECT_ROOT
    assert kw["encoding"] == "utf-8" and kw["errors"] == "replace"
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8" and kw["env"]["PYTHONUTF8"] == "1"
    assert not (tmp_path / pull.LOCK_NAME).exists()


def test_lock_removed_on_success_and_when_on_line_raises(tmp_path):
    s = _settings(tmp_path)
    pull.run_refresh(s, lambda _: None, popen=_popen(["x\n"], 0))
    assert not (tmp_path / pull.LOCK_NAME).exists()

    def boom(_):
        raise ValueError

    with pytest.raises(ValueError):
        pull.run_refresh(s, boom, popen=_popen(["x\n"], 0))
    assert not (tmp_path / pull.LOCK_NAME).exists()


def test_busy_when_lock_exists(tmp_path):
    lock = tmp_path / pull.LOCK_NAME
    lock.write_text("1")
    with pytest.raises(pull.PullBusy):
        pull.run_refresh(_settings(tmp_path), lambda _: None, popen=_popen([], 0))
    assert lock.exists()


def test_stale_lock_is_replaced(tmp_path):
    lock = tmp_path / pull.LOCK_NAME
    lock.write_text("1")
    old = time.time() - 31 * 60
    os.utime(lock, (old, old))
    assert pull.run_refresh(_settings(tmp_path), lambda _: None, popen=_popen([], 0)) == 0
    assert not lock.exists()


def test_pull_available(tmp_path):
    assert not pull.pull_available(_settings(tmp_path, api_key=None))
    assert pull.pull_available(_settings(tmp_path))
