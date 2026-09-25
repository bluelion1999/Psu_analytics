# Sub-project 1: Engineering Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **In this repo:** implementers and reviewers follow `.claude/agents/psu-implementer.md` and `.claude/agents/psu-reviewer.md`. Give each agent only its own task, plus **Global Constraints** and **Review Focus**.

**Goal:** `psu refresh` runs ingest → build → train → simulate in one command, and every push and PR into `main` is linted (ruff) and tested (pytest) in GitHub Actions.

**Architecture:** `src/psu/cli.py` keeps its argparse setup, but each command body becomes a handler `cmd_<name>(args, settings) -> int`, and `main()` dispatches through a dict. Ingest and simulate get inner functions (`_ingest`, `_simulate`) that return `(code, result)`, so `cmd_refresh` can chain the steps and print a summary. Tuned defaults move to module constants in `src/psu/config.py`. Ruff config lives in `pyproject.toml`, and CI is one workflow file.

**Tech Stack:** Python 3.11+ (local 3.13), argparse, DuckDB, pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-23-foundation-design.md`

## Global Constraints

- Branch `feat/foundation` (it already exists and has the spec commit). Make one Conventional Commit per task (the message is given in the task), staging only that task's files by explicit path. End each message with a second `-m` whose text is exactly `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No CFBD API calls**, and never run `psu` commands against the real `data/psu.duckdb`. Tests use `tmp_path` settings and fakes.
- Run tests with `.venv/Scripts/python -m pytest` and ruff with `.venv/Scripts/ruff` (Windows). Never import `sklearn.impute` or `sklearn.neighbors`, because Smart App Control blocks them.
- `requires-python = ">=3.11"`: no syntax newer than 3.11 (for example, no PEP 695 `type` statements and no nested same-quote f-strings).
- The existing command behaviour, exit codes (0 ok, 2 usage/data error, 3 budget stop) and messages must not change. Every existing test must pass without edits, except Task 1's mechanical lint/format edits.
- Console output is ASCII only, because the Windows console is cp1252.
- Ruff settings, verbatim: `line-length = 120`, `target-version = "py311"`, lint `select = ["E", "F", "W", "I", "UP", "B"]`, `ignore = ["B008"]`.

## Review Focus

1. **The database is held by another process** (a second `psu` command is writing while you run `psu refresh`): expect a one-line `error: ... in use by another process ...` and exit 2, not a DuckDB traceback. Tested in Task 2 (`test_database_in_use_is_a_clean_error`).
2. **`psu refresh` with no API key and nothing cached for the current season:** exit 2, with the existing `CFBD_API_KEY` message and `refresh stopped at ingest`; build, train and simulate never run. Tested in Task 3 (`test_refresh_without_key_stops_at_ingest`).
3. **`psu refresh --skip-ingest` on an empty database:** exit 2 with the existing "run `psu ingest` first" message, not a crash in train or simulate. Tested in Task 3 (`test_refresh_skip_ingest_on_empty_db_stops_at_build`).
4. **The one-time format pass changing behaviour** (for example, a `zip(strict=True)` that raises on real data where lengths differ): the full suite must pass unchanged after Task 1, and `strict=True` only goes where the lengths are equal by construction. Checked by the reviewer in Task 1.
5. **Linux-only CI failures** (case-sensitive paths, default UTF-8 vs cp1252, `\\` path separators hard-coded in tests): the first CI run on the PR shows them. The orchestrator checks the PR's CI after pushing and fixes any failures in a `fix(ci):` commit.

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | `[tool.ruff]` config; `ruff` in the `dev` extras |
| `.git-blame-ignore-revs` | new: the format-pass commit SHA |
| `src/`, `tests/`, `app/` (many files) | Task 1: mechanical `ruff format` and lint fixes only |
| `src/psu/config.py` | shared default constants |
| `src/psu/cli.py` | handler split, dispatch dict, `_ingest`/`_simulate`, DB-in-use handling, `refresh` |
| `tests/test_cli.py` | new tests for defaults, DB-in-use and `refresh` |
| `.github/workflows/ci.yml` | new: lint and test on 3.11 and 3.13 |
| `README.md` | CI badge; "Daily use" and "Development" sections |

---

### Task 1: Ruff config and one-time lint/format pass

**Files:**
- Modify: `pyproject.toml`
- Modify: any `.py` under `src/`, `tests/`, `app/` that ruff changes (mechanical only)
- Create: `.git-blame-ignore-revs`

**Interfaces:**
- Consumes: nothing.
- Produces: a ruff-clean tree. Later tasks must keep `ruff check .` and `ruff format --check .` clean.

- [ ] **Step 1: Add the ruff config and dev dependency**

In `pyproject.toml`, change the dev extras line to:

```toml
dev = ["pytest>=8", "ruff>=0.6"]
```

and append at the end of the file:

```toml

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["B008"]
```

- [ ] **Step 2: Install ruff into the venv**

Run: `.venv/Scripts/python -m pip install -e ".[dev]"`
Expected: `Successfully installed ruff-...` (other packages are already satisfied).

- [ ] **Step 3: Confirm the baseline fails**

Run: `.venv/Scripts/ruff check .`
Expected: FAIL with about 48 findings (I001, UP017, E501, B905, UP035, F401, E741).

- [ ] **Step 4: Apply the automatic fixes and formatting**

Run:
```bash
.venv/Scripts/ruff check . --fix
.venv/Scripts/ruff format .
.venv/Scripts/ruff check .
```
Expected: the last command still reports a handful of findings that can't be fixed automatically. The baseline had these:
- **B905** `zip()` without `strict=`: `app/views/games.py:17`, `app/views/predictions.py:32`, `src/psu/dashboard/games.py:67`, `src/psu/sim/ratings.py:36`, `src/psu/simulate.py:281`
- **E501** line too long: `src/psu/cli.py`, `src/psu/features.py` (2), `src/psu/transform.py`, `tests/test_build.py` (3), `tests/test_flatten.py`, `tests/test_transform.py`
- **E741** ambiguous name `l`: `tests/test_sim_standings.py:11`

Line numbers may have moved after formatting.

- [ ] **Step 5: Fix the remaining findings by hand**

Rules:
- **B905:** read each call. If both iterables come from the same DataFrame/Series, or are otherwise equal in length by construction, add `strict=True`. If lengths can legitimately differ, add `strict=False`. Never change which items are zipped.
- **E501:** wrap inside the existing brackets, or split a long string literal into adjacent literals (`"abc" "def"`). Never change a string's value or a test's data.
- **E741:** rename `l` to `loser` (or a similarly descriptive name) throughout that one function.

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` and `N files already formatted`.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python -m pytest`
Expected: all tests pass (about 260). If any fail, the fix that caused it changed behaviour; revert that specific edit and use `strict=False` or a different wrap.

- [ ] **Step 7: Commit the pass**

```bash
git status --short
git add pyproject.toml src tests app
git commit -m "style: adopt ruff and apply one-time format and lint pass" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(`git add` on these directories is allowed here only because every change under them is this task's mechanical pass. Check `git status --short` first, and confirm nothing under `data/`, `graphify-out/` or `docs/` is staged.)

- [ ] **Step 8: Record the commit for `git blame`**

```bash
git rev-parse HEAD
```
Create `.git-blame-ignore-revs` with this content (the SHA is the full 40-character output above):

```
# One-time ruff format and lint pass (sub-project 1)
<full sha from git rev-parse HEAD>
```

```bash
git add .git-blame-ignore-revs
git commit -m "chore: ignore the ruff format pass in git blame" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Shared defaults and CLI handler split

**Files:**
- Modify: `src/psu/config.py` (add constants after `FIRST_SEASON`)
- Modify: `src/psu/cli.py` (full rewrite below)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: the existing `psu.build.build`, `psu.ingest.ingest` / `IngestResult`, `psu.train.load_features` / `train_and_save` / `report_markdown`, and `psu.simulate.simulate_season` / `write_results` / `load_sigma` / `report_markdown` / `MissingModel` / `SimResult`.
- Produces, for Task 3:
  - `config.GARBAGE: str = "38,28,22"`, `config.BUILD_ALPHA: float = 50.0`, `config.TRAIN_ALPHA: float = 20.0`, `config.SHRINK_PLAYS: int = 75`, `config.SIM_N: int = 10_000`, `config.SIM_SEED: int = 0`, `config.SIM_TAU: float = 5.0`
  - `cli._ingest(settings, seasons: list[int], max_calls: int | None) -> tuple[int, IngestResult | None]`: prints the same output as today's `psu ingest`.
  - `cli._simulate(settings, *, team: str, n_sims: int, seed: int, tau: float) -> tuple[int, SimResult | None]`: prints the same output as today's `psu simulate`.
  - `cli.cmd_build(args, settings) -> int` (uses `args.garbage: str`, `args.alpha: float`) and `cli.cmd_train(args, settings) -> int` (uses `args.alpha: float`, `args.shrink_plays: int`).
  - `cli.HANDLERS: dict[str, Callable[[argparse.Namespace, config.Settings], int]]` and `cli.build_parser() -> argparse.ArgumentParser`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_parser_defaults_come_from_config():
    parser = cli.build_parser()
    b = parser.parse_args(["build"])
    assert (b.garbage, b.alpha) == (config.GARBAGE, config.BUILD_ALPHA)
    t = parser.parse_args(["train"])
    assert (t.alpha, t.shrink_plays) == (config.TRAIN_ALPHA, config.SHRINK_PLAYS)
    s = parser.parse_args(["simulate"])
    assert (s.sims, s.seed, s.tau, s.team) == (config.SIM_N, config.SIM_SEED, config.SIM_TAU, config.TEAM)


def test_database_in_use_is_a_clean_error(settings, capsys, monkeypatch):
    import duckdb

    from psu import db

    def locked(*args, **kwargs):
        raise duckdb.IOException("Could not set lock on file: held by PID 1234")

    monkeypatch.setattr(db, "connect", locked)
    assert cli.main(["build"]) == 2
    err = capsys.readouterr().err
    assert "in use by another process" in err and "PID 1234" in err
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -k "defaults_come_from_config or database_in_use" -v`
Expected: FAIL. `config` has no `GARBAGE` / `cli` has no `build_parser`, and the lock test errors with an uncaught `IOException`.

- [ ] **Step 3: Add the constants to `src/psu/config.py`**

Insert directly after the `FIRST_SEASON = ...` line:

```python

# Tuned pipeline defaults, shared by the individual commands and `psu refresh`.
GARBAGE = "38,28,22"  # Q2,Q3,Q4 garbage-time margins
BUILD_ALPHA = 50.0  # ridge shrinkage for opponent adjustment
TRAIN_ALPHA = 20.0  # ridge shrinkage for rolling team ratings
SHRINK_PLAYS = 75  # plays before a season's own data outweighs last season
SIM_N = 10_000  # simulated seasons
SIM_SEED = 0
SIM_TAU = 5.0  # spread (points) of each team's season-long strength draw
```

- [ ] **Step 4: Rewrite `src/psu/cli.py`**

Replace the whole file with:

```python
"""Command-line entry point: `psu ingest`, `psu status`, `psu build`, `psu train` and `psu simulate`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

import duckdb

from psu import config, db
from psu.build import build
from psu.client import BudgetExceeded, CachedClient, Fetch, MissingApiKey
from psu.ingest import IngestResult, ingest
from psu.simulate import MissingModel, SimResult, load_sigma, simulate_season, write_results
from psu.simulate import report_markdown as sim_report
from psu.train import load_features, report_markdown, train_and_save
from psu.transform import GarbageTime


def make_fetch(settings: config.Settings) -> Fetch:
    if not settings.api_key:

        def missing_key(endpoint, params):
            raise MissingApiKey(
                f"CFBD_API_KEY is not set and {endpoint} {params} is not cached. "
                "Copy .env.example to .env and add your free key from https://collegefootballdata.com/key"
            )

        return missing_key
    from psu.cfbd_api import make_cfbd_fetch  # imported lazily: cfbd is slow to import

    return make_cfbd_fetch(settings.api_key)


def _print_counts(counts: dict[str, int]) -> None:
    for table, n in counts.items():
        print(f"{table:<18} {n:>10,}")


def _no_plays(con: duckdb.DuckDBPyConnection) -> bool:
    if db.row_counts(con)["plays"] == 0:
        print("error: no plays loaded yet; run `psu ingest` first", file=sys.stderr)
        return True
    return False


def cmd_status(args: argparse.Namespace, settings: config.Settings) -> int:
    if settings.db_path.exists():
        con = db.connect(settings.db_path, read_only=True)
        try:
            _print_counts(db.row_counts(con))
        finally:
            con.close()
    else:
        _print_counts({name: 0 for name in db.SPECS})
    return 0


def _ingest(
    settings: config.Settings, seasons: list[int], max_calls: int | None
) -> tuple[int, IngestResult | None]:
    client = CachedClient(
        settings.raw_dir,
        make_fetch(settings),
        max_calls=settings.max_calls if max_calls is None else max_calls,
        min_interval_s=settings.min_interval_s,
    )
    con = db.connect(settings.db_path)
    try:
        result = ingest(
            client,
            con,
            seasons,
            current=settings.current_season,
            final_after=settings.final_after,
            refresh_after=settings.refresh_after,
        )
    except MissingApiKey as e:
        print(f"error: {e}", file=sys.stderr)
        return 2, None
    except BudgetExceeded as e:
        _print_counts(db.row_counts(con))
        print(f"stopped: {e}", file=sys.stderr)
        return 3, None
    finally:
        con.close()
    _print_counts(result.row_counts)
    print(f"API calls this run: {result.api_calls}")
    return 0, result


def cmd_ingest(args: argparse.Namespace, settings: config.Settings) -> int:
    try:
        seasons = config.parse_seasons(
            args.seasons or f"{config.FIRST_SEASON}-{settings.current_season}", last=settings.current_season
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return _ingest(settings, seasons, args.max_calls)[0]


def cmd_build(args: argparse.Namespace, settings: config.Settings) -> int:
    try:
        garbage = GarbageTime.parse(args.garbage)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    con = db.connect(settings.db_path)
    try:
        if _no_plays(con):
            return 2
        _print_counts(build(con, garbage=garbage, alpha=args.alpha))
    finally:
        con.close()
    return 0


def cmd_train(args: argparse.Namespace, settings: config.Settings) -> int:
    con = db.connect(settings.db_path)
    try:
        if _no_plays(con):
            return 2
        features = load_features(con, alpha=args.alpha, shrink_plays=args.shrink_plays)
        try:
            report = train_and_save(
                con, features, current_season=settings.current_season, out_dir=settings.db_path.parent
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    finally:
        con.close()
    print(report_markdown(report))
    return 0


def _simulate(
    settings: config.Settings, *, team: str, n_sims: int, seed: int, tau: float
) -> tuple[int, SimResult | None]:
    try:
        sigma = load_sigma(settings.db_path.parent)
        con = db.connect(settings.db_path)
        try:
            result = simulate_season(
                con, season=settings.current_season, sigma=sigma, team=team, n_sims=n_sims, seed=seed, tau=tau
            )
            write_results(con, result, settings.db_path.parent)
        finally:
            con.close()
    except (MissingModel, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2, None
    print(sim_report(result))
    return 0, result


def cmd_simulate(args: argparse.Namespace, settings: config.Settings) -> int:
    return _simulate(settings, team=args.team, n_sims=args.sims, seed=args.seed, tau=args.tau)[0]


HANDLERS: dict[str, Callable[[argparse.Namespace, config.Settings], int]] = {
    "status": cmd_status,
    "ingest": cmd_ingest,
    "build": cmd_build,
    "train": cmd_train,
    "simulate": cmd_simulate,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psu", description="Penn State football analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Pull CFBD data into DuckDB (cached; safe to re-run)")
    ing.add_argument("--seasons", help='e.g. "2024", "2022-2026" or "2022,2024-2025" (default: every season)')
    ing.add_argument("--max-calls", type=int, help="Stop before making more than this many API calls")
    sub.add_parser("status", help="Show row counts per table")
    bld = sub.add_parser("build", help="Compute metric tables from ingested data (no API calls)")
    bld.add_argument("--garbage", default=config.GARBAGE, help='Q2,Q3,Q4 garbage-time margins, or "off"')
    bld.add_argument("--alpha", type=float, default=config.BUILD_ALPHA, help="Ridge shrinkage for opponent adjustment")
    trn = sub.add_parser("train", help="Backtest and train the game model; write predictions (no API calls)")
    trn.add_argument("--alpha", type=float, default=config.TRAIN_ALPHA, help="Ridge shrinkage for rolling team ratings")
    trn.add_argument(
        "--shrink-plays",
        type=int,
        default=config.SHRINK_PLAYS,
        help="Plays before a season's own data outweighs last season",
    )
    sim = sub.add_parser("simulate", help="Simulate the rest of the season (no API calls)")
    sim.add_argument("--sims", type=int, default=config.SIM_N, help="Number of simulated seasons")
    sim.add_argument("--seed", type=int, default=config.SIM_SEED, help="Random seed (same seed, same results)")
    sim.add_argument(
        "--tau", type=float, default=config.SIM_TAU, help="Spread (points) of each team's season-long strength draw"
    )
    sim.add_argument("--team", default=config.TEAM, help="Team to report on")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = config.load_settings()
    try:
        return HANDLERS[args.command](args, settings)
    except duckdb.IOException as e:
        print(
            f"error: {settings.db_path} is in use by another process; close it and retry ({e})",
            file=sys.stderr,
        )
        return 2
```

Note: `duckdb.IOException` is raised when another process holds the file lock. The message above is ASCII only.

- [ ] **Step 5: Run the new tests and the whole CLI file**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: all pass, including every pre-existing test (unchanged).

- [ ] **Step 6: Run the full suite and lint**

Run: `.venv/Scripts/python -m pytest && .venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: all pass. If ruff format wants changes, run `.venv/Scripts/ruff format src/psu/cli.py src/psu/config.py tests/test_cli.py` and re-run.

- [ ] **Step 7: Commit**

```bash
git status --short
git add src/psu/config.py src/psu/cli.py tests/test_cli.py
git commit -m "refactor(cli): split commands into handlers and share tuned defaults" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `psu refresh`

**Files:**
- Modify: `src/psu/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes (from Task 2): `_ingest`, `_simulate`, `cmd_build`, `cmd_train`, `HANDLERS`, `build_parser`, and the `config.*` defaults. `IngestResult(api_calls: int, row_counts: dict[str, int])` from `psu.ingest`. `SimResult.team: str` and `SimResult.mean_wins: float` from `psu.simulate`.
- Produces: `cli.cmd_refresh(args, settings) -> int`, registered as `HANDLERS["refresh"]`. `refresh` flags: `--skip-ingest`, `--max-calls N`, `--sims N` (default `config.SIM_N`), `--seed N` (default `config.SIM_SEED`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def _stub_steps(monkeypatch, codes=None, api_calls=4):
    """Replace refresh's four steps with recorders; codes maps step -> exit code (default 0)."""
    from types import SimpleNamespace

    from psu.ingest import IngestResult

    codes = codes or {}
    calls = []

    def ingest_step(settings, seasons, max_calls):
        calls.append(("ingest", seasons, max_calls))
        code = codes.get("ingest", 0)
        return code, (IngestResult(api_calls=api_calls, row_counts={}) if code == 0 else None)

    def build_step(args, settings):
        calls.append(("build", args.garbage, args.alpha))
        return codes.get("build", 0)

    def train_step(args, settings):
        calls.append(("train", args.alpha, args.shrink_plays))
        return codes.get("train", 0)

    def simulate_step(settings, *, team, n_sims, seed, tau):
        calls.append(("simulate", team, n_sims, seed, tau))
        code = codes.get("simulate", 0)
        return code, (SimpleNamespace(team=team, mean_wins=9.47) if code == 0 else None)

    monkeypatch.setattr(cli, "_ingest", ingest_step)
    monkeypatch.setattr(cli, "cmd_build", build_step)
    monkeypatch.setattr(cli, "cmd_train", train_step)
    monkeypatch.setattr(cli, "_simulate", simulate_step)
    return calls


def test_refresh_runs_every_step_in_order_with_shared_defaults(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch)
    assert cli.main(["refresh"]) == 0
    assert calls == [
        ("ingest", [2026], None),
        ("build", config.GARBAGE, config.BUILD_ALPHA),
        ("train", config.TRAIN_ALPHA, config.SHRINK_PLAYS),
        ("simulate", config.TEAM, config.SIM_N, config.SIM_SEED, config.SIM_TAU),
    ]
    out = capsys.readouterr().out
    assert out.index("== ingest ==") < out.index("== build ==") < out.index("== train ==") < out.index("== simulate ==")
    assert "refresh ok in" in out and "4 API calls" in out and "Penn State mean wins 9.47" in out


def test_refresh_passes_flags_through(settings, monkeypatch):
    calls = _stub_steps(monkeypatch)
    assert cli.main(["refresh", "--max-calls", "7", "--sims", "500", "--seed", "3"]) == 0
    assert calls[0] == ("ingest", [2026], 7)
    assert calls[-1] == ("simulate", config.TEAM, 500, 3, config.SIM_TAU)


def test_refresh_stops_at_the_first_failing_step(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch, codes={"build": 2})
    assert cli.main(["refresh"]) == 2
    assert [c[0] for c in calls] == ["ingest", "build"]
    captured = capsys.readouterr()
    assert "refresh stopped at build (exit 2)" in captured.err
    assert "refresh ok" not in captured.out


def test_refresh_budget_stop_returns_3(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch, codes={"ingest": 3})
    assert cli.main(["refresh"]) == 3
    assert [c[0] for c in calls] == ["ingest"]
    assert "refresh stopped at ingest (exit 3)" in capsys.readouterr().err


def test_refresh_skip_ingest_never_ingests(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch)
    assert cli.main(["refresh", "--skip-ingest"]) == 0
    assert [c[0] for c in calls] == ["build", "train", "simulate"]
    out = capsys.readouterr().out
    assert "== ingest ==" not in out and "0 API calls" in out


def test_refresh_without_key_stops_at_ingest(settings, capsys):
    assert cli.main(["refresh"]) == 2
    err = capsys.readouterr().err
    assert "CFBD_API_KEY" in err and "refresh stopped at ingest (exit 2)" in err


def test_refresh_skip_ingest_on_empty_db_stops_at_build(settings, capsys):
    assert cli.main(["refresh", "--skip-ingest"]) == 2
    err = capsys.readouterr().err
    assert "psu ingest" in err and "refresh stopped at build (exit 2)" in err
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -k refresh -v`
Expected: FAIL with `argument command: invalid choice: 'refresh'` (SystemExit 2 from argparse).

- [ ] **Step 3: Implement `cmd_refresh`**

In `src/psu/cli.py`:

1. Add `import time` to the stdlib imports (after `import sys`).
2. Add this function directly after `cmd_simulate`:

```python
def _step(name: str) -> None:
    print(f"== {name} ==", flush=True)


def _stopped(name: str, code: int) -> int:
    print(f"refresh stopped at {name} (exit {code})", file=sys.stderr)
    return code


def cmd_refresh(args: argparse.Namespace, settings: config.Settings) -> int:
    """ingest (current season) -> build -> train -> simulate, stopping at the first failure."""
    start = time.monotonic()
    api_calls = 0
    if not args.skip_ingest:
        _step("ingest")
        code, ingested = _ingest(settings, [settings.current_season], args.max_calls)
        if code:
            return _stopped("ingest", code)
        api_calls = ingested.api_calls
    _step("build")
    code = cmd_build(argparse.Namespace(garbage=config.GARBAGE, alpha=config.BUILD_ALPHA), settings)
    if code:
        return _stopped("build", code)
    _step("train")
    code = cmd_train(argparse.Namespace(alpha=config.TRAIN_ALPHA, shrink_plays=config.SHRINK_PLAYS), settings)
    if code:
        return _stopped("train", code)
    _step("simulate")
    code, sim = _simulate(settings, team=config.TEAM, n_sims=args.sims, seed=args.seed, tau=config.SIM_TAU)
    if code:
        return _stopped("simulate", code)
    elapsed = time.monotonic() - start
    print(f"refresh ok in {elapsed:.0f}s: {api_calls} API calls, {sim.team} mean wins {sim.mean_wins:.2f}")
    return 0
```

3. Add `"refresh": cmd_refresh,` as the last entry of `HANDLERS`.
4. In `build_parser()`, before `return parser`, add:

```python
    ref = sub.add_parser("refresh", help="Run ingest (current season), build, train and simulate in order")
    ref.add_argument("--skip-ingest", action="store_true", help="Skip ingest (no API calls); rebuild from local data")
    ref.add_argument("--max-calls", type=int, help="Stop ingest before making more than this many API calls")
    ref.add_argument("--sims", type=int, default=config.SIM_N, help="Number of simulated seasons")
    ref.add_argument("--seed", type=int, default=config.SIM_SEED, help="Random seed (same seed, same results)")
```

5. Update the module docstring to:

```python
"""Command-line entry point: `psu ingest`, `status`, `build`, `train`, `simulate` and `refresh`."""
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite and lint**

Run: `.venv/Scripts/python -m pytest && .venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git status --short
git add src/psu/cli.py tests/test_cli.py
git commit -m "feat(cli): add psu refresh to run the whole pipeline" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: GitHub Actions CI and README

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the ruff config (Task 1) and `psu refresh` (Task 3).
- Produces: a CI workflow named `CI` with job `test`, used by the README badge.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Format
        run: ruff format --check .
      - name: Test
        run: pytest
```

- [ ] **Step 2: Validate the YAML parses**

Run: `.venv/Scripts/python -c "import yaml, pathlib; d = yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print(sorted(d[True if True in d else 'on']), d['jobs']['test']['strategy']['matrix'])"`
Expected: prints `['pull_request', 'push'] {'python-version': ['3.11', '3.13']}`. (PyYAML parses the `on:` key as boolean `True`. It ships with Streamlit's dependencies. If `import yaml` fails, skip this step and say so in the report.)

- [ ] **Step 3: Update `README.md`**

1. Directly under the `# PSU Analytics` title line, insert a blank line and then:

```markdown
[![CI](https://github.com/bluelion1999/PSU_analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/bluelion1999/PSU_analytics/actions/workflows/ci.yml)
```

Before writing it, run `git remote get-url origin` and use that owner/repo in both URLs if it differs from `bluelion1999/PSU_analytics`.

2. Directly after the `## Setup` section (before `## Phase 1: Data pipeline`), insert:

````markdown
## Daily use

```powershell
.venv\Scripts\psu refresh                # ingest the current season, then build, train and simulate
.venv\Scripts\psu refresh --skip-ingest  # offline: rebuild everything from local data (no API calls)
```

`psu refresh` runs the same steps as `psu ingest --seasons <current>`, `psu build`, `psu train` and
`psu simulate`, using the tuned defaults in `src/psu/config.py`. It prints a `== step ==` header
for each step and stops at the first one that fails, with that step's exit code: 2 for a usage or
data error, 3 when ingest hits the API budget. A daily in-season run costs about 5 API calls.
Flags: `--max-calls N` (ingest budget), `--sims N` and `--seed N` (simulation).

## Development

```powershell
.venv\Scripts\python -m pytest    # tests (real-data tests skip when data/psu.duckdb is absent)
.venv\Scripts\ruff check .        # lint
.venv\Scripts\ruff format .       # format
```

CI (GitHub Actions) runs lint, a format check and the tests on Python 3.11 and 3.13 for every push
and pull request into `main`. To hide the one-time format commit from `git blame`, run
`git config blame.ignoreRevsFile .git-blame-ignore-revs`.
````

- [ ] **Step 4: Full suite and lint**

Run: `.venv/Scripts/python -m pytest && .venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git status --short
git add .github/workflows/ci.yml README.md
git commit -m "ci: lint and test on Python 3.11 and 3.13; document psu refresh" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the tasks (orchestrator)

1. Run a whole-branch review.
2. Push `feat/foundation` and open a PR into `main` with `C:\Program Files\GitHub CLI\gh.exe`.
3. Check the PR's CI (Review Focus 5). Fix any Linux-only failures in a `fix(ci):` commit.
4. Don't merge without asking the user.
