"""Run `psu refresh` from the dashboard: command line, single-run lock, and streamed output (no Streamlit here)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from psu import config

LOCK_NAME = "refresh.lock"
STALE_SECONDS = 30 * 60


class PullBusy(RuntimeError):
    """Another pull is already running."""


def refresh_command(max_calls: int = 60) -> list[str]:
    return [sys.executable, "-m", "psu.cli", "refresh", "--max-calls", str(max_calls)]


def pull_available(settings: config.Settings) -> bool:
    return bool(settings.api_key)


def _take_lock(path: Path) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    for attempt in range(2):
        try:
            fd = os.open(path, flags)
        except FileExistsError:
            try:
                stale = time.time() - path.stat().st_mtime > STALE_SECONDS
            except FileNotFoundError:
                continue  # released between the two calls; retry
            if attempt == 1 or not stale:
                raise PullBusy("A pull is already running.") from None
            # Claim the stale lock atomically: only one racer's replace can succeed.
            grave = path.with_name(f"{path.name}.{os.getpid()}.stale")
            try:
                os.replace(path, grave)
            except FileNotFoundError:
                continue
            grave.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return


def run_refresh(
    settings: config.Settings,
    on_line: Callable[[str], None],
    *,
    max_calls: int = 60,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    lock = settings.db_path.parent / LOCK_NAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    _take_lock(lock)
    proc = None
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        proc = popen(
            refresh_command(max_calls),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=config.PROJECT_ROOT,
            env=env,
        )
        for line in proc.stdout:
            on_line(line.rstrip())
        return proc.wait()
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            lock.unlink(missing_ok=True)
