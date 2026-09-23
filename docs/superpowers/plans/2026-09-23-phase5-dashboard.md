# Phase 5: Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **In this repo:** implementers and reviewers follow `.claude/agents/psu-implementer.md` and `.claude/agents/psu-reviewer.md`. Give each agent only its own task, plus **Global Constraints** and **Review Focus**.

**Goal:** `streamlit run app/streamlit_app.py` opens a five-page Penn State dashboard: Season overview, Efficiency trends, Game explorer, Players and Predictions. It reads everything from `data/psu.duckdb` and makes no API calls.

**Architecture:**
- **`src/psu/dashboard/`** holds pure data functions, each taking `(con, ...)` and returning a DataFrame or dict. There is one module per page, plus `common.py` (database path, table checks, a team's view of its games) and `winprob.py` (in-game win probability). All of these are unit-tested against a small seeded DuckDB.
- **`src/psu/dashboard/ui.py`** is the only Streamlit-aware module in the package. It provides cached reads on short-lived read-only connections, the season picker, and friendly messages when data is missing.
- **`app/`** is thin: a navigation entry point and one script per page under `app/views/`. Pages are smoke-tested with Streamlit's `AppTest`.

**Tech Stack:** Streamlit (>=1.40; `st.navigation`, `st.cache_data`, `streamlit.testing.v1.AppTest`), Altair (ships with Streamlit), pandas, DuckDB, scipy (`norm`).

**Spec:** `docs/superpowers/specs/psu-analytics-build-plan.md`, section "Phase 5 — Dashboard (Streamlit)". The user asked for the plan directly, so the design decisions that section leaves open are recorded below instead of in a separate design doc.

## Design decisions (no separate spec; correct these before execution)

| Topic | Decision |
|---|---|
| Team focus | Penn State (`psu.config.TEAM`) on every page. The Players page also has a team picker. |
| Season | A sidebar picker listing the seasons in `games`, defaulting to the latest one. |
| Pages | 1. **Season overview:** record, conference record, games left; schedule with results, Vegas and model lines, and win probability; key metrics vs. the team's conference average and the FBS average. 2. **Efficiency trends:** offense and defense EPA/play and success rate by season (team, conference average, FBS average), plus game by game for the chosen season (garbage time excluded). 3. **Game explorer:** pick a completed game; see the line score, box score, drive chart, win-probability chart, and top 10 plays by absolute EPA. 4. **Players:** season passing, rushing and receiving tables from box scores, with a team picker and a minimum-volume slider. 5. **Predictions:** Penn State's upcoming games (model margin, win probability, Vegas); the season-simulation tiles (mean wins, P(10+), P(title game), P(Big Ten champ), P(CFP)); the win-total chart; the Big Ten title race; and the next week's FBS slate. |
| In-game win probability | The database has none, so the dashboard derives it (Stern 1994). The rest of the game's home scoring is treated as `N(pregame_margin * r, sigma^2 * r)`, where `r` is the fraction of regulation left. So `P(home wins) = Phi((home_margin + pregame_margin * r) / (sigma * sqrt(r)))`, with `r` clamped to `[1e-4, 1]`. `pregame_margin` is the game's `game_predictions.pred_margin` (0 if missing). `sigma` comes from `data/reports/game_model.json`. Overtime plays count as `r = 0`. |
| Player efficiency | Box-score efficiency: comp %, yards per attempt, per carry and per catch. Plays don't carry player IDs, so per-player EPA is out of scope. CFBD's `" Team"` pseudo-player rows are excluded. |
| Charts | Altair. No new charting dependency. |
| Database access | Every read opens a **read-only** DuckDB connection, runs one data function and closes. Results are cached for 10 minutes (`st.cache_data(ttl=600)`), keyed on the database path. A sidebar "Refresh data" button clears the cache. If the file can't be opened (for example, a `psu` command is writing), the page shows a message instead of a traceback. |
| Missing data | A missing table shows `st.info` naming the command that creates it (`psu ingest`, `psu build`, `psu train` or `psu simulate`). A page stops only when its core table is missing. Optional sections (such as the simulation, or the win probability) show the message and the rest of the page still renders. |
| Database override | The `PSU_DB_PATH` environment variable overrides `Settings.db_path`. Tests use it, and it's handy for pointing at a copy. |

## Global Constraints

- Branch `feat/phase5-dashboard`. Make one Conventional Commit per task (the message is given in the task), staging only that task's files by explicit path. End each message with a second `-m` whose text is exactly `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No CFBD API calls**, and no writes to `data/psu.duckdb`. The dashboard opens DuckDB with `read_only=True` only. Never run `psu ingest`, `psu build`, `psu train` or `psu simulate` against the real database.
- Run tests with `.venv/Scripts/python -m pytest` (Windows). Never import `sklearn.impute` or `sklearn.neighbors`, because Smart App Control blocks them.
- New runtime dependency: `streamlit>=1.40`. It is already installed in `.venv` (1.64). Altair comes with it.
- Data functions live in `src/psu/dashboard/` and take a DuckDB connection first. Only `psu/dashboard/ui.py` and `app/` import `streamlit`.
- Team margins shown to users are from the team's side: `+7` means Penn State is favoured by 7. `game_predictions` stores home-minus-away margins and `home_win_prob`. For away games, flip the sign and use `1 - home_win_prob`.
- Pages must never show a Python traceback for missing data, an empty season or a busy database. They show `st.info` with the command to run.
- Console output must be ASCII only (the Windows console is cp1252). Text rendered in pages may use any characters.

## Review Focus

1. **A fresh or partial database** (no DuckDB file; `psu ingest` run but not `build`, `train` or `simulate`): every page shows an info message naming the command, not a traceback. Tested in Task 1 (`test_read_missing_file_is_missing_data`, `test_require_names_the_command`) and Task 7 (`test_pages_explain_a_missing_database`, `test_pages_work_before_train_and_simulate`).
2. **The database is locked by a running `psu` command:** the page shows "try again" instead of crashing. Tested in Task 1 (`test_read_busy_database_is_missing_data`).
3. **A season that has just started, or hasn't started** (the default season has no completed games): the overview shows 0-0, and the game explorer, trends and metrics show "no games yet" instead of crashing. Tested in Task 2 (`test_metric_comparison_for_a_season_without_stats`), Task 3 (`test_weekly_trends_empty_season`), Task 4 (`test_game_options_empty_before_any_game`) and Task 7 (the default season in the app tests is 2027, which has one unplayed game).
4. **Messy box-score rows:** CFBD's `" Team"` pseudo-player, `"--"` values and a missing `C/ATT` are excluded or counted as 0, not crashes or `NaN` totals. Tested in Task 5 (`test_team_pseudo_player_is_excluded`, `test_malformed_stats_count_as_zero`).
5. **Stale numbers after re-running `psu build`, `train` or `simulate`:** the cache is keyed on the database path, expires after 10 minutes, and the sidebar "Refresh data" button clears it. Tested in Task 7 (`test_refresh_button_clears_cache`).

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | add `streamlit>=1.40` |
| `src/psu/dashboard/__init__.py` | package marker |
| `src/psu/dashboard/common.py` | `MissingData`, `SOURCES`, `db_path`, `read`, `require`, `has_table`, `seasons`, `team_games`, `team_conference`, `conference_members`, `benchmark` |
| `src/psu/dashboard/overview.py` | `record`, `schedule`, `SCHEDULE_COLUMNS`, `METRICS`, `metric_comparison` |
| `src/psu/dashboard/trends.py` | `TREND_METRICS`, `season_trends`, `weekly_trends` |
| `src/psu/dashboard/winprob.py` | `in_game_wp` |
| `src/psu/dashboard/games.py` | `game_options`, `line_scores`, `BOX_STATS`, `box_score`, `drive_chart`, `top_plays`, `win_probability` |
| `src/psu/dashboard/players.py` | `COLUMNS`, `VOLUME`, `teams`, `player_table` |
| `src/psu/dashboard/predictions.py` | `upcoming`, `sim_summary`, `sim_win_totals`, `sim_conference`, `next_slate` |
| `src/psu/dashboard/ui.py` | `load`, `load_or_stop`, `load_or_note`, `season_picker` (Streamlit) |
| `app/streamlit_app.py` | navigation entry point |
| `app/views/{overview,trends,games,players,predictions}.py` | one page each |
| `tests/conftest.py` | adds `DASH_CONF`, `DASH_GAMES`, `seed_dashboard_db(con, *, with_model=True)` |
| `tests/test_dashboard_*.py` | tests |
| `.claude/launch.json`, `README.md` | preview config and a Phase 5 section (Task 8) |

---

### Task 1: Dashboard foundation (`dashboard/common.py`, test world)

**Files:**
- Modify: `pyproject.toml` (add `"streamlit>=1.40",` to `dependencies`, after `"scipy>=1.11",`)
- Create: `src/psu/dashboard/__init__.py`
- Create: `src/psu/dashboard/common.py`
- Modify: `tests/conftest.py` (append the dashboard test world at the end)
- Test: `tests/test_dashboard_common.py`

**Interfaces:**
- Consumes:
  - `psu.config.load_settings().db_path` and `psu.config`
  - `psu.db.SPECS`, `psu.db.upsert`, `psu.db.connect`
  - `psu.build.build(con)`, `psu.build._write(con, name, df)`
- Produces:
  - `class MissingData(RuntimeError)`
  - `SOURCES: dict[str, str]`, mapping a table to the command that creates it
  - `db_path() -> Path`, where `PSU_DB_PATH` overrides the setting
  - `read(fn, *args, path: Path | None = None)`, which calls `fn(con, *args)` on a read-only connection
  - `require(con, *tables)`, which raises `MissingData` naming the commands
  - `has_table(con, table) -> bool`
  - `seasons(con) -> list[int]` (newest first)
  - `team_games(con, season, team) -> DataFrame[game_id, week, season_type, start_date, completed, conference_game, venue, opponent, team_points, opp_points, is_home, result]`, where `completed` is True only when both scores are known, `venue` is `'home'`, `'away'` or `'neutral'`, and `result` is `'W'`, `'L'` or `None`
  - `team_conference(con, season, team) -> str | None`
  - `conference_members(con, season, conference) -> list[str]`
  - `benchmark(con, season, team, table, column, members) -> dict[value, conference_avg, national_avg]`
  - conftest: `DASH_CONF`, `DASH_GAMES`, `seed_dashboard_db(con, *, with_model=True)`

- [ ] **Step 1: Add the test world to `tests/conftest.py`**

Append to the end of `tests/conftest.py`:

```python
DASH_CONF = {"Penn State": "Big Ten", "Ohio State": "Big Ten", "Temple": "American Athletic", "Buffalo": "Mid-American"}
DASH_STRENGTH = {"Penn State": 0.3, "Ohio State": 0.25, "Temple": -0.1, "Buffalo": -0.2}
DASH_GAMES = [
    # id, season, week, home, away, home_points, away_points (None = not played yet)
    (101, 2025, 1, "Penn State", "Temple", 34, 10),
    (102, 2025, 1, "Ohio State", "Buffalo", 40, 7),
    (103, 2025, 2, "Ohio State", "Penn State", 24, 27),
    (104, 2025, 2, "Temple", "Buffalo", 20, 17),
    (201, 2026, 1, "Penn State", "Buffalo", 45, 0),
    (202, 2026, 1, "Temple", "Ohio State", 3, 38),
    (203, 2026, 2, "Penn State", "Ohio State", None, None),
    (204, 2026, 2, "Buffalo", "Temple", None, None),
    (301, 2027, 1, "Penn State", "Temple", None, None),
]


def _dash_quarters(points):
    import json

    base = points // 4
    return json.dumps([base, base, base, points - 3 * base])


def _dash_pred_margin(home, away):
    return 7.0 if home == "Penn State" else -4.0 if away == "Penn State" else 1.0


def seed_dashboard_db(con, *, with_model=True):
    """A small four-team world for dashboard tests, loaded like the real pipeline (upsert, then build).

    Penn State goes 2-0 in 2025 (1-0 in the Big Ten, winning 27-24 at Ohio State) and is 1-0 in 2026 with
    Ohio State still to play; 2027 has one unplayed game. Each played game has one 8-play drive per team
    (home in quarters 1-2, away in quarters 3-4), box stats and QB/RB/WR box lines plus CFBD's " Team" row.
    with_model=False skips game_predictions and the sim_* tables (the state before `psu train`).
    """
    from scipy.stats import norm

    from psu.build import _write, build
    from psu.db import SPECS, upsert

    games, plays, drives, box, players = [], [], [], [], []
    for gid, season, week, home, away, hp, ap in DASH_GAMES:
        done = hp is not None
        games.append({
            "id": gid, "season": season, "week": week, "season_type": "regular",
            "start_date": pd.Timestamp(season, 9, 6) + pd.Timedelta(days=7 * (week - 1)),
            "completed": done, "neutral_site": False, "conference_game": DASH_CONF[home] == DASH_CONF[away],
            "home_team": home, "home_conference": DASH_CONF[home], "home_classification": "fbs",
            "away_team": away, "away_conference": DASH_CONF[away], "away_classification": "fbs",
            "home_points": hp, "away_points": ap,
            "home_line_scores": _dash_quarters(hp) if done else None,
            "away_line_scores": _dash_quarters(ap) if done else None,
            "notes": None,
        })
        if not done:
            continue
        for offense, defense, points in ((home, away, hp), (away, home, ap)):
            first = offense == home
            prefix = offense.split()[0].lower()
            side = "home" if first else "away"
            before = 0 if first else min(hp, 14)  # home scored before the away drive; keep margins out of garbage time
            scored = points >= 20
            drives.append({
                "id": f"{gid}-{prefix}", "game_id": gid, "season": season, "offense": offense,
                "offense_conference": DASH_CONF[offense], "defense": defense, "defense_conference": DASH_CONF[defense],
                "drive_number": 1 if first else 2, "start_period": 1 if first else 3,
                "start_yards_to_goal": 75, "end_yards_to_goal": 5 if scored else 40, "plays": 8,
                "yards": 70 if scored else 35, "drive_result": "TD" if scored else "PUNT",
                "elapsed_minutes": 4, "elapsed_seconds": 30, "start_offense_score": 0,
                "start_defense_score": before, "end_offense_score": 7 if scored else 0, "end_defense_score": before,
            })
            for i in range(8):
                plays.append({
                    "id": f"{gid}-{prefix}-{i}", "game_id": gid, "drive_id": f"{gid}-{prefix}", "season": season,
                    "week": week, "season_type": "regular", "drive_number": 1 if first else 2, "play_number": i + 1,
                    "offense": offense, "offense_conference": DASH_CONF[offense], "defense": defense,
                    "defense_conference": DASH_CONF[defense], "home": home, "away": away,
                    "period": 1 + i // 4 + (0 if first else 2), "clock_minutes": 14 - 3 * (i % 4), "clock_seconds": 0,
                    "offense_score": 0, "defense_score": before, "down": 1 + i % 3, "distance": 10,
                    "yards_to_goal": 75 - 8 * i, "yards_gained": 8,
                    "play_type": "Rush" if i % 2 else "Pass Reception", "play_text": f"{offense} play {i + 1}",
                    "ppa": round(DASH_STRENGTH[offense] + (0.4 if i == 3 else -0.1 if i % 2 else 0.1), 3),
                })
            for category, stat in (
                ("totalYards", str(300 + points)), ("netPassingYards", "200"), ("rushingYards", str(100 + points)),
                ("firstDowns", "18"), ("thirdDownEff", "5-12"), ("turnovers", "0" if scored else "1"),
                ("tacklesForLoss", "4"), ("possessionTime", "30:00"),
            ):
                box.append({"game_id": gid, "season": season, "week": week, "season_type": "regular", "team": offense,
                            "conference": DASH_CONF[offense], "home_away": side, "points": points,
                            "category": category, "stat": stat})
            for category, athlete, stats in (
                ("passing", "QB", {"C/ATT": "20/30", "YDS": str(150 + points), "TD": "2", "INT": "1", "AVG": "7.0"}),
                ("rushing", "RB", {"CAR": "15", "YDS": str(60 + points), "TD": "1", "LONG": "25", "AVG": "5.0"}),
                ("receiving", "WR", {"REC": "6", "YDS": "90", "TD": "1", "LONG": "40", "AVG": "15.0"}),
                ("rushing", " Team", {"CAR": "2", "YDS": "-3", "TD": "0", "LONG": "0"}),
            ):
                name = athlete if athlete == " Team" else f"{offense} {athlete}"
                for stat_type, stat in stats.items():
                    players.append({"game_id": gid, "season": season, "week": week, "season_type": "regular",
                                    "team": offense, "conference": DASH_CONF[offense], "home_away": side,
                                    "category": category, "stat_type": stat_type,
                                    "athlete_id": f"{prefix}-{athlete.strip().lower()}", "athlete_name": name,
                                    "stat": stat})
    games = pd.DataFrame(games)
    games[["home_points", "away_points"]] = games[["home_points", "away_points"]].astype("Int64")
    upsert(con, SPECS["games"], games)
    upsert(con, SPECS["plays"], pd.DataFrame(plays))
    upsert(con, SPECS["drives"], pd.DataFrame(drives))
    upsert(con, SPECS["team_game_stats"], pd.DataFrame(box))
    upsert(con, SPECS["player_game_stats"], pd.DataFrame(players))
    build(con)
    if not with_model:
        return
    preds = pd.DataFrame([
        {
            "game_id": gid, "season": season, "week": week, "season_type": "regular",
            "start_date": pd.Timestamp(season, 9, 6) + pd.Timedelta(days=7 * (week - 1)),
            "neutral_site": False, "home_team": home, "away_team": away, "completed": hp is not None,
            "margin": float(hp - ap) if hp is not None else float("nan"), "vegas_margin": 3.5,
            "pred_margin": _dash_pred_margin(home, away),
            "home_win_prob": float(norm.cdf(_dash_pred_margin(home, away) / 16.0)),
            "split": "upcoming" if hp is None else "no_prior" if season == 2025 else "in_sample",
        }
        for gid, season, week, home, away, hp, ap in DASH_GAMES
    ])
    _write(con, "game_predictions", preds)
    _write(con, "sim_team_summary", pd.DataFrame([{
        "season": 2026, "team": "Penn State", "n_sims": 1000, "seed": 0, "tau": 5.0,
        "as_of": pd.Timestamp(2026, 9, 6), "mean_wins": 1.6, "p_10_plus": 0.0, "p_title_game": 0.55,
        "p_conf_champ": 0.3, "p_cfp": 0.4,
    }]))
    _write(con, "sim_win_totals", pd.DataFrame({"season": 2026, "team": "Penn State", "wins": [0, 1, 2],
                                                "prob": [0.0, 0.4, 0.6]}))
    _write(con, "sim_conference", pd.DataFrame({
        "season": 2026, "team": ["Ohio State", "Penn State"], "mean_conf_wins": [0.6, 0.4],
        "p_title_game": [1.0, 1.0], "p_conf_champ": [0.7, 0.3],
    }))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_dashboard_common.py`:

```python
from pathlib import Path

import duckdb
import pytest

from conftest import seed_dashboard_db
from psu.dashboard.common import (
    MissingData, benchmark, conference_members, db_path, read, require, seasons, team_conference, team_games,
)
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_seasons_newest_first(con):
    assert seasons(con) == [2027, 2026, 2025]


def test_team_games_from_the_teams_side(con):
    g = team_games(con, 2025, "Penn State").set_index("game_id")
    assert list(g.index) == [101, 103]
    assert g.loc[103, "venue"] == "away" and g.loc[103, "opponent"] == "Ohio State"
    assert (g.loc[103, "team_points"], g.loc[103, "opp_points"], g.loc[103, "result"]) == (27, 24, "W")
    assert bool(g.loc[103, "conference_game"]) and not bool(g.loc[101, "conference_game"])
    assert bool(g.loc[101, "is_home"]) and not bool(g.loc[103, "is_home"])


def test_unplayed_games_have_no_result(con):
    g = team_games(con, 2026, "Penn State").set_index("game_id")
    assert not bool(g.loc[203, "completed"]) and g.loc[203, "result"] is None
    assert bool(g.loc[201, "completed"]) and g.loc[201, "result"] == "W"


def test_conference_lookup(con):
    assert team_conference(con, 2025, "Penn State") == "Big Ten"
    assert conference_members(con, 2025, "Big Ten") == ["Ohio State", "Penn State"]
    assert team_conference(con, 2025, "Nobody") is None


def test_benchmark_matches_the_metric_table(con):
    rows = con.execute("SELECT team, epa_per_play FROM team_offense WHERE season = 2025").df().set_index("team")
    b = benchmark(con, 2025, "Penn State", "team_offense", "epa_per_play", {"Penn State", "Ohio State"})
    assert b["value"] == pytest.approx(rows.loc["Penn State", "epa_per_play"])
    assert b["conference_avg"] == pytest.approx(rows.loc[["Penn State", "Ohio State"], "epa_per_play"].mean())
    assert b["national_avg"] == pytest.approx(rows["epa_per_play"].mean())


def test_require_names_the_command():
    c = connect(":memory:")
    with pytest.raises(MissingData, match="psu train"):
        require(c, "game_predictions")
    with pytest.raises(MissingData, match="psu ingest"):
        require(c, "games")


def test_db_path_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PSU_DB_PATH", str(tmp_path / "copy.duckdb"))
    assert db_path() == tmp_path / "copy.duckdb"


def test_read_missing_file_is_missing_data(tmp_path):
    with pytest.raises(MissingData, match="psu ingest"):
        read(lambda c: 1, path=tmp_path / "nope.duckdb")


def test_read_opens_read_only(tmp_path):
    path = tmp_path / "psu.duckdb"
    connect(path).close()
    assert read(lambda c: c.execute("SELECT 41 + 1").fetchone()[0], path=path) == 42
    with pytest.raises(MissingData):  # a write through the dashboard connection is refused
        read(lambda c: c.execute("CREATE TABLE t (x INTEGER)"), path=path)


def test_read_busy_database_is_missing_data(tmp_path):
    path = tmp_path / "psu.duckdb"
    writer = duckdb.connect(str(path))  # another command holding a write connection
    try:
        with pytest.raises(MissingData, match="Try again"):
            read(lambda c: 1, path=path)
    finally:
        writer.close()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.dashboard'`.

- [ ] **Step 4: Write the implementation**

In `pyproject.toml`, add `"streamlit>=1.40",` to `dependencies` after `"scipy>=1.11",`.

`src/psu/dashboard/__init__.py`:

```python
"""Dashboard data functions (pure, take a DuckDB connection) plus the Streamlit helpers in ui.py."""
```

`src/psu/dashboard/common.py`:

```python
"""Shared helpers for dashboard data: the database path, read-only access, table checks and a team's view of games."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from psu import config

SOURCES = {
    "games": "psu ingest", "plays": "psu ingest", "drives": "psu ingest", "team_game_stats": "psu ingest",
    "player_game_stats": "psu ingest",
    "plays_enriched": "psu build", "team_offense": "psu build", "team_defense": "psu build",
    "team_havoc": "psu build", "team_turnovers": "psu build",
    "game_predictions": "psu train",
    "sim_team_summary": "psu simulate", "sim_win_totals": "psu simulate", "sim_conference": "psu simulate",
}


class MissingData(RuntimeError):
    """Data the dashboard needs is missing or unreadable; the message says what to run."""


def db_path() -> Path:
    override = os.environ.get("PSU_DB_PATH")
    return Path(override) if override else config.load_settings().db_path


def read(fn: Callable[..., Any], *args: Any, path: Path | None = None) -> Any:
    """Call fn(con, *args) on a fresh read-only connection, closing it afterwards."""
    path = Path(path) if path is not None else db_path()
    if not path.exists():
        raise MissingData(f"No database at {path}; run `psu ingest` first")
    try:
        con = duckdb.connect(str(path), read_only=True)
    except duckdb.Error as e:
        raise MissingData(
            f"Could not open {path} ({type(e).__name__}); another psu command may be writing to it. "
            "Try again in a moment."
        ) from e
    try:
        return fn(con, *args)
    except duckdb.InvalidInputException as e:
        raise MissingData(f"The dashboard only reads the database: {e}") from e
    finally:
        con.close()


def has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0] > 0


def require(con: duckdb.DuckDBPyConnection, *tables: str) -> None:
    missing = [t for t in tables if not has_table(con, t)]
    if missing:
        commands = " and ".join(f"`{c}`" for c in sorted({SOURCES.get(t, "psu build") for t in missing}))
        raise MissingData(f"Missing {', '.join(missing)}; run {commands} first")


def seasons(con: duckdb.DuckDBPyConnection) -> list[int]:
    require(con, "games")
    return [int(s) for s in con.execute("SELECT DISTINCT season FROM games ORDER BY season DESC").df()["season"]]


TEAM_GAMES_SQL = """
SELECT id AS game_id, week, season_type, start_date, completed, coalesce(conference_game, false) AS conference_game,
       CASE WHEN neutral_site THEN 'neutral' WHEN home_team = $team THEN 'home' ELSE 'away' END AS venue,
       CASE WHEN home_team = $team THEN away_team ELSE home_team END AS opponent,
       CASE WHEN home_team = $team THEN home_points ELSE away_points END AS team_points,
       CASE WHEN home_team = $team THEN away_points ELSE home_points END AS opp_points,
       home_team = $team AS is_home
FROM games
WHERE season = $season AND (home_team = $team OR away_team = $team) AND season_type IN ('regular', 'postseason')
ORDER BY start_date, week
"""


def team_games(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    require(con, "games")
    df = con.execute(TEAM_GAMES_SQL, {"season": season, "team": team}).df()
    # DuckDB hands back nullable integers (pd.NA) for an all-null column; floats keep comparisons simple.
    df[["team_points", "opp_points"]] = df[["team_points", "opp_points"]].astype(float)
    df["completed"] = (
        df["completed"].fillna(False).astype(bool) & df["team_points"].notna() & df["opp_points"].notna()
    )
    df["conference_game"] = df["conference_game"].astype(bool)
    df["is_home"] = df["is_home"].astype(bool)
    df["result"] = pd.Series(  # object dtype keeps None (pandas would otherwise turn it into NaN)
        np.where(df["completed"], np.where(df["team_points"] > df["opp_points"], "W", "L"), None),
        index=df.index, dtype=object,
    )
    return df


def team_conference(con: duckdb.DuckDBPyConnection, season: int, team: str) -> str | None:
    require(con, "games")
    row = con.execute(
        """
        SELECT conf FROM (
            SELECT home_conference AS conf FROM games WHERE season = $season AND home_team = $team
            UNION ALL
            SELECT away_conference FROM games WHERE season = $season AND away_team = $team
        ) WHERE conf IS NOT NULL GROUP BY conf ORDER BY count(*) DESC, conf LIMIT 1
        """,
        {"season": season, "team": team},
    ).fetchone()
    return None if row is None else row[0]


def conference_members(con: duckdb.DuckDBPyConnection, season: int, conference: str) -> list[str]:
    require(con, "games")
    return con.execute(
        """
        SELECT DISTINCT team FROM (
            SELECT home_team AS team FROM games WHERE season = $season AND home_conference = $conf
            UNION
            SELECT away_team FROM games WHERE season = $season AND away_conference = $conf
        ) ORDER BY team
        """,
        {"season": season, "conf": conference},
    ).df()["team"].tolist()


def benchmark(
    con: duckdb.DuckDBPyConnection, season: int, team: str, table: str, column: str, members: set[str]
) -> dict[str, float]:
    """A metric for the team next to its conference average and the FBS average (the tables are FBS-only)."""
    df = con.execute(f'SELECT team, "{column}" AS value FROM "{table}" WHERE season = ?', [season]).df()
    mine = df.loc[df["team"] == team, "value"]
    return {
        "value": float(mine.iloc[0]) if len(mine) else float("nan"),
        "conference_avg": float(df.loc[df["team"].isin(members), "value"].mean()),
        "national_avg": float(df["value"].mean()),
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_common.py -v`
Expected: 10 passed.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python -m pytest`
Expected: everything passes.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/psu/dashboard/__init__.py src/psu/dashboard/common.py tests/conftest.py tests/test_dashboard_common.py
git commit -m "feat(dashboard): add read-only data access and a seeded test world" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Season overview data (`dashboard/overview.py`)

**Files:**
- Create: `src/psu/dashboard/overview.py`
- Test: `tests/test_dashboard_overview.py`

**Interfaces:**
- Consumes: from `psu.dashboard.common`: `team_games`, `has_table`, `require`, `team_conference`, `conference_members` and `benchmark` (Task 1); conftest `seed_dashboard_db`.
- Produces:
  - `record(con, season, team) -> dict[wins, losses, conf_wins, conf_losses, remaining]`
  - `SCHEDULE_COLUMNS = ["game_id", "week", "season_type", "start_date", "opponent", "venue", "completed", "result", "team_points", "opp_points", "vegas_margin", "model_margin", "win_prob", "prediction"]`
  - `schedule(con, season, team) -> DataFrame[SCHEDULE_COLUMNS]`. Margins and win probability are from the team's side. `prediction` is the `game_predictions.split`. With no `game_predictions` table, the prediction columns are NaN.
  - `METRICS: list[tuple[label, table, column, higher_is_better]]`
  - `metric_comparison(con, season, team) -> DataFrame[metric, value, conference_avg, national_avg, higher_is_better]`

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_overview.py`:

```python
import math

import pytest
from scipy.stats import norm

from conftest import seed_dashboard_db
from psu.dashboard.common import MissingData
from psu.dashboard.overview import METRICS, SCHEDULE_COLUMNS, metric_comparison, record, schedule
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_record_counts_conference_games_and_games_left(con):
    assert record(con, 2025, "Penn State") == {
        "wins": 2, "losses": 0, "conf_wins": 1, "conf_losses": 0, "remaining": 0,
    }
    assert record(con, 2026, "Penn State") == {
        "wins": 1, "losses": 0, "conf_wins": 0, "conf_losses": 0, "remaining": 1,
    }
    assert record(con, 2027, "Penn State")["remaining"] == 1


def test_schedule_lines_are_from_the_teams_side(con):
    s = schedule(con, 2025, "Penn State")
    assert list(s.columns) == SCHEDULE_COLUMNS
    away = s.set_index("game_id").loc[103]  # Penn State at Ohio State: stored home margin -4, Vegas 3.5
    assert away["model_margin"] == pytest.approx(4.0)
    assert away["vegas_margin"] == pytest.approx(-3.5)
    assert away["win_prob"] == pytest.approx(1 - norm.cdf(-4.0 / 16.0))
    home = schedule(con, 2026, "Penn State").set_index("game_id").loc[203]
    assert home["model_margin"] == pytest.approx(7.0) and home["prediction"] == "upcoming"
    assert home["win_prob"] == pytest.approx(norm.cdf(7.0 / 16.0))
    assert not bool(home["completed"]) and home["result"] is None


def test_schedule_without_predictions_has_blank_lines():
    c = connect(":memory:")
    seed_dashboard_db(c, with_model=False)
    s = schedule(c, 2026, "Penn State")
    assert len(s) == 2 and s["model_margin"].isna().all() and s["win_prob"].isna().all()


def test_metric_comparison_against_conference_and_fbs(con):
    m = metric_comparison(con, 2025, "Penn State").set_index("metric")
    assert list(m.index) == [label for label, _, _, _ in METRICS]
    offense = con.execute("SELECT team, epa_per_play FROM team_offense WHERE season = 2025").df().set_index("team")
    row = m.loc["Offense EPA/play"]
    assert row["value"] == pytest.approx(offense.loc["Penn State", "epa_per_play"])
    assert row["conference_avg"] == pytest.approx(offense.loc[["Penn State", "Ohio State"], "epa_per_play"].mean())
    assert row["national_avg"] == pytest.approx(offense["epa_per_play"].mean())
    assert not bool(m.loc["Defense EPA/play", "higher_is_better"])


def test_metric_comparison_for_a_season_without_stats(con):
    m = metric_comparison(con, 2027, "Penn State")
    assert len(m) == len(METRICS) and m["value"].isna().all()


def test_metric_comparison_needs_build_tables():
    c = connect(":memory:")
    c.execute("CREATE TABLE games AS SELECT 1 AS id")
    with pytest.raises(MissingData, match="psu build"):
        metric_comparison(c, 2025, "Penn State")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_overview.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.dashboard.overview'`.

- [ ] **Step 3: Write the implementation**

`src/psu/dashboard/overview.py`:

```python
"""Season overview data: record, schedule with results and pregame lines, and metrics against conference and FBS."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from psu.dashboard.common import benchmark, conference_members, has_table, require, team_conference, team_games

SCHEDULE_COLUMNS = [
    "game_id", "week", "season_type", "start_date", "opponent", "venue", "completed", "result",
    "team_points", "opp_points", "vegas_margin", "model_margin", "win_prob", "prediction",
]
METRICS = [  # (label, table, column, higher_is_better)
    ("Offense EPA/play", "team_offense", "epa_per_play", True),
    ("Offense success rate", "team_offense", "success_rate", True),
    ("Offense explosiveness", "team_offense", "explosiveness", True),
    ("Defense EPA/play", "team_defense", "epa_per_play", False),
    ("Defense success rate", "team_defense", "success_rate", False),
    ("Havoc rate", "team_havoc", "havoc_rate", True),
    ("Turnover margin", "team_turnovers", "margin", True),
]


def record(con: duckdb.DuckDBPyConnection, season: int, team: str) -> dict[str, int]:
    games = team_games(con, season, team)
    done = games[games["completed"]]
    conf = done[done["conference_game"]]
    return {
        "wins": int((done["result"] == "W").sum()),
        "losses": int((done["result"] == "L").sum()),
        "conf_wins": int((conf["result"] == "W").sum()),
        "conf_losses": int((conf["result"] == "L").sum()),
        "remaining": int((~games["completed"]).sum()),
    }


def schedule(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    games = team_games(con, season, team)
    if has_table(con, "game_predictions"):
        preds = con.execute(
            "SELECT game_id, pred_margin, vegas_margin, home_win_prob, split FROM game_predictions WHERE season = ?",
            [season],
        ).df()
    else:
        preds = pd.DataFrame({
            "game_id": pd.Series(dtype="int64"), "pred_margin": pd.Series(dtype=float),
            "vegas_margin": pd.Series(dtype=float), "home_win_prob": pd.Series(dtype=float),
            "split": pd.Series(dtype=object),
        })
    df = games.merge(preds, on="game_id", how="left")
    sign = np.where(df["is_home"], 1.0, -1.0)
    home_prob = df["home_win_prob"].astype(float)
    df["model_margin"] = sign * df["pred_margin"].astype(float)
    df["vegas_margin"] = sign * df["vegas_margin"].astype(float)
    df["win_prob"] = np.where(df["is_home"], home_prob, 1.0 - home_prob)
    df["prediction"] = df["split"]
    return df[SCHEDULE_COLUMNS]


def metric_comparison(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    require(con, *sorted({table for _, table, _, _ in METRICS}))
    conference = team_conference(con, season, team)
    members = set(conference_members(con, season, conference)) if conference else set()
    rows = [
        {"metric": label, **benchmark(con, season, team, table, column, members), "higher_is_better": higher}
        for label, table, column, higher in METRICS
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "conference_avg", "national_avg", "higher_is_better"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_overview.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/dashboard/overview.py tests/test_dashboard_overview.py
git commit -m "feat(dashboard): season record, schedule lines and metric comparison" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Efficiency trends data (`dashboard/trends.py`)

**Files:**
- Create: `src/psu/dashboard/trends.py`
- Test: `tests/test_dashboard_trends.py`

**Interfaces:**
- Consumes: from `psu.dashboard.common`: `benchmark`, `conference_members`, `require`, `seasons`, `team_conference` and `team_games` (Task 1).
- Produces:
  - `TREND_METRICS: list[tuple[label, table, column]]`
  - `season_trends(con, team) -> DataFrame[season, metric, group, value]` in long format. `group` is the team name, `f"{conference} avg"` or `"FBS avg"`. Seasons with no metric rows are skipped.
  - `weekly_trends(con, season, team) -> DataFrame[game, start_date, week, opponent, side, epa_per_play, success_rate, plays]`. `side` is `'offense'` or `'defense'`. `game` is `"Wk 3 Buffalo"`, or `"Bowl <opp>"` for the postseason. Rows are ordered by date, and garbage time is excluded.

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_trends.py`:

```python
import pytest

from conftest import seed_dashboard_db
from psu.dashboard.trends import TREND_METRICS, season_trends, weekly_trends
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_season_trends_long_format(con):
    t = season_trends(con, "Penn State")
    assert list(t.columns) == ["season", "metric", "group", "value"]
    assert set(t["season"]) == {2025, 2026}  # 2027 has no metric rows yet
    assert set(t["metric"]) == {label for label, _, _ in TREND_METRICS}
    assert set(t["group"]) == {"Penn State", "Big Ten avg", "FBS avg"}
    mine = t[(t["season"] == 2025) & (t["metric"] == "Offense EPA/play") & (t["group"] == "Penn State")]["value"]
    expected = con.execute(
        "SELECT epa_per_play FROM team_offense WHERE season = 2025 AND team = 'Penn State'"
    ).fetchone()[0]
    assert mine.item() == pytest.approx(expected)


def test_weekly_trends_one_row_per_game_and_side(con):
    w = weekly_trends(con, 2025, "Penn State")
    assert list(w.columns) == [
        "game", "start_date", "week", "opponent", "side", "epa_per_play", "success_rate", "plays",
    ]
    assert list(w["game"]) == ["Wk 1 Temple", "Wk 1 Temple", "Wk 2 Ohio State", "Wk 2 Ohio State"]
    assert set(w["side"]) == {"offense", "defense"}
    offense = w[(w["game"] == "Wk 1 Temple") & (w["side"] == "offense")].iloc[0]
    expected = con.execute(
        "SELECT avg(ppa) FROM plays_enriched WHERE game_id = 101 AND offense = 'Penn State' "
        "AND NOT coalesce(garbage, false) AND ppa IS NOT NULL"
    ).fetchone()[0]
    assert offense["epa_per_play"] == pytest.approx(expected)
    assert w["success_rate"].between(0, 1).all()


def test_weekly_trends_empty_season(con):
    w = weekly_trends(con, 2027, "Penn State")
    assert w.empty and "epa_per_play" in w.columns
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_trends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.dashboard.trends'`.

- [ ] **Step 3: Write the implementation**

`src/psu/dashboard/trends.py`:

```python
"""Efficiency trends data: EPA/play and success rate by season, and game by game within a season."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from psu.dashboard.common import benchmark, conference_members, require, seasons, team_conference, team_games

TREND_METRICS = [  # (label, table, column)
    ("Offense EPA/play", "team_offense", "epa_per_play"),
    ("Defense EPA/play", "team_defense", "epa_per_play"),
    ("Offense success rate", "team_offense", "success_rate"),
    ("Defense success rate", "team_defense", "success_rate"),
]
WEEKLY_COLUMNS = ["game", "start_date", "week", "opponent", "side", "epa_per_play", "success_rate", "plays"]
WEEKLY_SQL = """
SELECT game_id,
       CASE WHEN offense = $team THEN 'offense' ELSE 'defense' END AS side,
       avg(ppa) AS epa_per_play,
       avg(CAST(success AS DOUBLE)) AS success_rate,
       count(*) AS plays
FROM plays_enriched
WHERE season = $season AND (offense = $team OR defense = $team)
  AND NOT coalesce(garbage, false) AND ppa IS NOT NULL
GROUP BY ALL
"""


def season_trends(con: duckdb.DuckDBPyConnection, team: str) -> pd.DataFrame:
    require(con, "games", "team_offense", "team_defense")
    rows = []
    for season in sorted(seasons(con)):
        conference = team_conference(con, season, team)
        members = set(conference_members(con, season, conference)) if conference else set()
        for label, table, column in TREND_METRICS:
            b = benchmark(con, season, team, table, column, members)
            if np.isnan(b["national_avg"]):
                continue  # no metric rows for this season yet
            if not np.isnan(b["value"]):
                rows.append({"season": season, "metric": label, "group": team, "value": b["value"]})
            if members:
                rows.append({"season": season, "metric": label, "group": f"{conference} avg",
                             "value": b["conference_avg"]})
            rows.append({"season": season, "metric": label, "group": "FBS avg", "value": b["national_avg"]})
    return pd.DataFrame(rows, columns=["season", "metric", "group", "value"])


def weekly_trends(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    require(con, "plays_enriched", "games")
    stats = con.execute(WEEKLY_SQL, {"season": season, "team": team}).df()
    games = team_games(con, season, team)[["game_id", "week", "season_type", "start_date", "opponent"]]
    df = stats.merge(games, on="game_id").sort_values(["start_date", "side"], ascending=[True, False])
    if df.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    prefix = pd.Series(
        np.where(df["season_type"] == "postseason", "Bowl", "Wk " + df["week"].astype(str)), index=df.index
    )
    df["game"] = prefix + " " + df["opponent"]
    return df[WEEKLY_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_trends.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/dashboard/trends.py tests/test_dashboard_trends.py
git commit -m "feat(dashboard): efficiency trends by season and by game" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Game explorer data (`dashboard/winprob.py`, `dashboard/games.py`)

**Files:**
- Create: `src/psu/dashboard/winprob.py`
- Create: `src/psu/dashboard/games.py`
- Test: `tests/test_dashboard_games.py`

**Interfaces:**
- Consumes: from `psu.dashboard.common`: `has_table`, `require` and `team_games` (Task 1).
- Produces:
  - `in_game_wp(home_margin, pregame_margin, frac_left, sigma) -> np.ndarray`
  - `game_options(con, season, team) -> DataFrame[game_id, label]` for completed games only. A label looks like `"Wk 2 at Ohio State (W 27-24)"`.
  - `line_scores(con, game_id) -> DataFrame[team, Q1.., Total]`, with the away team's row first
  - `BOX_STATS: list[tuple[category, label]]`
  - `box_score(con, game_id) -> DataFrame[stat, <away>, <home>]` of strings, with `""` when a stat is missing
  - `drive_chart(con, game_id) -> DataFrame[drive_number, offense, quarter, start_pos, end_pos, plays, yards, result, time]`, where `*_pos` is yards from the offense's own goal
  - `top_plays(con, game_id, n=10) -> DataFrame[quarter, clock, offense, down, distance, epa, text]`, sorted by `|epa|` descending
  - `win_probability(con, game_id, sigma) -> DataFrame[play, seconds_left, home_margin, home_wp, minute]`. It includes a kickoff row (3600 s, margin 0) and a final row (0 s, the final margin).

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_games.py`:

```python
import numpy as np
import pytest
from scipy.stats import norm

from conftest import seed_dashboard_db
from psu.dashboard.games import (
    BOX_STATS, box_score, drive_chart, game_options, line_scores, top_plays, win_probability,
)
from psu.dashboard.winprob import in_game_wp
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_in_game_wp_matches_pregame_at_kickoff_and_score_at_the_end():
    assert in_game_wp(0, 7.0, 1.0, 16.0) == pytest.approx(norm.cdf(7.0 / 16.0))
    assert in_game_wp(3, -10.0, 0.0, 16.0) == pytest.approx(1.0)
    assert in_game_wp(-3, 10.0, 0.0, 16.0) == pytest.approx(0.0)
    assert in_game_wp(0, 0.0, 0.5, 16.0) == pytest.approx(0.5)
    out = in_game_wp(np.array([0, 7]), 0.0, np.array([1.0, 0.5]), 16.0)
    assert out.shape == (2,) and out[1] > 0.5


def test_game_options_list_completed_games(con):
    o = game_options(con, 2025, "Penn State")
    assert list(o["game_id"]) == [101, 103]
    assert list(o["label"]) == ["Wk 1 vs Temple (W 34-10)", "Wk 2 at Ohio State (W 27-24)"]


def test_game_options_empty_before_any_game(con):
    o = game_options(con, 2027, "Penn State")
    assert o.empty and list(o.columns) == ["game_id", "label"]


def test_line_scores_away_first_and_add_up(con):
    s = line_scores(con, 101)
    assert list(s["team"]) == ["Temple", "Penn State"]
    assert list(s.columns) == ["team", "Q1", "Q2", "Q3", "Q4", "Total"]
    for _, row in s.iterrows():
        assert row[["Q1", "Q2", "Q3", "Q4"]].sum() == row["Total"]


def test_box_score_away_then_home_with_blanks(con):
    b = box_score(con, 101).set_index("stat")
    assert list(b.columns) == ["Temple", "Penn State"]
    assert list(b.index) == [label for _, label in BOX_STATS]
    assert b.loc["Total yards", "Penn State"] == "334" and b.loc["Total yards", "Temple"] == "310"
    assert b.loc["4th down", "Penn State"] == ""


def test_drive_chart_positions_from_own_goal(con):
    d = drive_chart(con, 101)
    assert list(d.columns) == [
        "drive_number", "offense", "quarter", "start_pos", "end_pos", "plays", "yards", "result", "time",
    ]
    psu = d[d["offense"] == "Penn State"].iloc[0]
    assert (psu["start_pos"], psu["end_pos"], psu["result"], psu["time"]) == (25, 95, "TD", "4:30")


def test_top_plays_by_absolute_epa(con):
    t = top_plays(con, 101, n=3)
    assert len(t) == 3
    assert list(t.columns) == ["quarter", "clock", "offense", "down", "distance", "epa", "text"]
    assert t["epa"].abs().is_monotonic_decreasing
    assert t.iloc[0]["epa"] == pytest.approx(0.7) and t.iloc[0]["clock"] == "5:00"


def test_win_probability_runs_kickoff_to_final(con):
    wp = win_probability(con, 101, 16.0)
    assert list(wp.columns) == ["play", "seconds_left", "home_margin", "home_wp", "minute"]
    assert len(wp) == 16 + 2
    assert wp.iloc[0]["home_wp"] == pytest.approx(norm.cdf(7.0 / 16.0))  # Penn State favoured by 7 at home
    assert wp.iloc[-1]["home_wp"] == pytest.approx(1.0) and wp.iloc[-1]["home_margin"] == 24
    assert wp["minute"].is_monotonic_increasing
    assert wp["home_wp"].between(0, 1).all()


def test_win_probability_without_predictions_starts_even():
    c = connect(":memory:")
    seed_dashboard_db(c, with_model=False)
    assert win_probability(c, 101, 16.0).iloc[0]["home_wp"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_games.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.dashboard.games'`.

- [ ] **Step 3: Write the implementation**

`src/psu/dashboard/winprob.py`:

```python
"""In-game win probability from the score, time left and the pregame expected margin (Stern 1994)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def in_game_wp(home_margin, pregame_margin, frac_left, sigma: float) -> np.ndarray:
    """P(home wins): remaining home scoring ~ N(pregame_margin * r, sigma^2 * r), r = fraction of regulation left."""
    r = np.clip(np.asarray(frac_left, dtype=float), 1e-4, 1.0)
    z = (np.asarray(home_margin, dtype=float) + np.asarray(pregame_margin, dtype=float) * r) / (sigma * np.sqrt(r))
    return norm.cdf(z)
```

`src/psu/dashboard/games.py`:

```python
"""Game explorer data: completed games, line score, box score, drives, biggest plays and win probability."""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from psu.dashboard.common import has_table, require, team_games
from psu.dashboard.winprob import in_game_wp

BOX_STATS = [  # (team_game_stats category, label)
    ("totalYards", "Total yards"),
    ("netPassingYards", "Passing yards"),
    ("rushingYards", "Rushing yards"),
    ("yardsPerPass", "Yards per pass"),
    ("yardsPerRushAttempt", "Yards per rush"),
    ("firstDowns", "First downs"),
    ("thirdDownEff", "3rd down"),
    ("fourthDownEff", "4th down"),
    ("turnovers", "Turnovers"),
    ("totalPenaltiesYards", "Penalties-yards"),
    ("possessionTime", "Possession"),
]
REGULATION_SECONDS = 3600


def _clock(minutes: pd.Series, seconds: pd.Series) -> pd.Series:
    return (
        minutes.fillna(0).astype(int).astype(str) + ":" + seconds.fillna(0).astype(int).astype(str).str.zfill(2)
    )


def game_options(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    games = team_games(con, season, team)
    done = games[games["completed"]].copy()
    if done.empty:
        return pd.DataFrame(columns=["game_id", "label"])
    week = pd.Series(
        np.where(done["season_type"] == "postseason", "Bowl", "Wk " + done["week"].astype(str)), index=done.index
    )
    where = done["venue"].map({"home": "vs", "away": "at", "neutral": "vs"})
    score = done["team_points"].astype(int).astype(str) + "-" + done["opp_points"].astype(int).astype(str)
    done["label"] = week + " " + where + " " + done["opponent"] + " (" + done["result"] + " " + score + ")"
    return done[["game_id", "label"]].reset_index(drop=True)


def _teams(con: duckdb.DuckDBPyConnection, game_id: int) -> tuple:
    row = con.execute(
        "SELECT home_team, away_team, home_points, away_points, home_line_scores, away_line_scores "
        "FROM games WHERE id = ?",
        [game_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"game {game_id} not found")
    return row


def line_scores(con: duckdb.DuckDBPyConnection, game_id: int) -> pd.DataFrame:
    require(con, "games")
    home, away, home_pts, away_pts, home_lines, away_lines = _teams(con, game_id)
    rows = []
    for team, points, raw in ((away, away_pts, away_lines), (home, home_pts, home_lines)):
        periods = json.loads(raw) if raw else []
        labels = [f"Q{i + 1}" if i < 4 else f"OT{i - 3}" for i in range(len(periods))]
        rows.append({"team": team, **dict(zip(labels, periods)), "Total": points})
    return pd.DataFrame(rows)


def box_score(con: duckdb.DuckDBPyConnection, game_id: int) -> pd.DataFrame:
    require(con, "games", "team_game_stats")
    home, away, *_ = _teams(con, game_id)
    stats = con.execute("SELECT team, category, stat FROM team_game_stats WHERE game_id = ?", [game_id]).df()
    lookup = {(r.team, r.category): r.stat for r in stats.itertuples()}

    def cell(team: str, category: str) -> str:
        value = lookup.get((team, category))
        return "" if value is None or pd.isna(value) else str(value)

    rows = [{"stat": label, away: cell(away, category), home: cell(home, category)} for category, label in BOX_STATS]
    return pd.DataFrame(rows, columns=["stat", away, home])


def drive_chart(con: duckdb.DuckDBPyConnection, game_id: int) -> pd.DataFrame:
    require(con, "drives")
    df = con.execute(
        "SELECT drive_number, offense, start_period, start_yards_to_goal, end_yards_to_goal, plays, yards, "
        "drive_result, elapsed_minutes, elapsed_seconds FROM drives WHERE game_id = ? ORDER BY drive_number",
        [game_id],
    ).df()
    df["start_pos"] = 100 - df["start_yards_to_goal"]
    df["end_pos"] = 100 - df["end_yards_to_goal"]
    df["time"] = _clock(df["elapsed_minutes"], df["elapsed_seconds"])
    df = df.rename(columns={"start_period": "quarter", "drive_result": "result"})
    return df[["drive_number", "offense", "quarter", "start_pos", "end_pos", "plays", "yards", "result", "time"]]


def top_plays(con: duckdb.DuckDBPyConnection, game_id: int, n: int = 10) -> pd.DataFrame:
    require(con, "plays")
    df = con.execute(
        "SELECT period AS quarter, clock_minutes, clock_seconds, offense, down, distance, ppa AS epa, "
        "play_text AS text FROM plays WHERE game_id = ? AND ppa IS NOT NULL ORDER BY abs(ppa) DESC, id LIMIT ?",
        [game_id, n],
    ).df()
    df["clock"] = _clock(df["clock_minutes"], df["clock_seconds"])
    return df[["quarter", "clock", "offense", "down", "distance", "epa", "text"]]


def win_probability(con: duckdb.DuckDBPyConnection, game_id: int, sigma: float) -> pd.DataFrame:
    require(con, "games", "plays")
    home, _, home_pts, away_pts, *_ = _teams(con, game_id)
    pregame = 0.0
    if has_table(con, "game_predictions"):
        row = con.execute("SELECT pred_margin FROM game_predictions WHERE game_id = ?", [game_id]).fetchone()
        if row is not None and row[0] is not None and not pd.isna(row[0]):
            pregame = float(row[0])
    plays = con.execute(
        "SELECT period, clock_minutes, clock_seconds, offense, offense_score, defense_score FROM plays "
        "WHERE game_id = ? AND period IS NOT NULL ORDER BY drive_number, play_number, id",
        [game_id],
    ).df()
    home_margin = np.where(
        plays["offense"] == home,
        plays["offense_score"] - plays["defense_score"],
        plays["defense_score"] - plays["offense_score"],
    )
    seconds = np.where(
        plays["period"] <= 4,
        (4 - plays["period"]) * 900 + plays["clock_minutes"].fillna(0) * 60 + plays["clock_seconds"].fillna(0),
        0,
    )
    frames = [pd.DataFrame({"seconds_left": [REGULATION_SECONDS], "home_margin": [0]}),
              pd.DataFrame({"seconds_left": seconds, "home_margin": home_margin})]
    if home_pts is not None and away_pts is not None:
        frames.append(pd.DataFrame({"seconds_left": [0], "home_margin": [home_pts - away_pts]}))
    wp = pd.concat(frames, ignore_index=True).astype(float)
    wp["seconds_left"] = wp["seconds_left"].clip(0, REGULATION_SECONDS)
    wp["home_margin"] = wp["home_margin"].ffill()
    wp["home_wp"] = in_game_wp(wp["home_margin"], pregame, wp["seconds_left"] / REGULATION_SECONDS, sigma)
    wp["minute"] = (REGULATION_SECONDS - wp["seconds_left"]) / 60
    wp.insert(0, "play", range(len(wp)))
    return wp
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_games.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/dashboard/winprob.py src/psu/dashboard/games.py tests/test_dashboard_games.py
git commit -m "feat(dashboard): game explorer data and in-game win probability" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Player tables (`dashboard/players.py`)

**Files:**
- Create: `src/psu/dashboard/players.py`
- Test: `tests/test_dashboard_players.py`

**Interfaces:**
- Consumes: `psu.dashboard.common.require` (Task 1); conftest `seed_dashboard_db`; `psu.db.SPECS` and `psu.db.upsert` (tests only).
- Produces:
  - `COLUMNS: dict[kind, list[str]]`, where kind is `"passing"`, `"rushing"` or `"receiving"`
  - `VOLUME = {"passing": "att", "rushing": "carries", "receiving": "receptions"}`
  - `teams(con, season) -> list[str]`
  - `player_table(con, season, team, kind, minimum=0) -> DataFrame[COLUMNS[kind]]`. It has one row per player, is sorted by yards (descending), and keeps only players whose `VOLUME[kind]` is `>= minimum`. An unknown kind raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_players.py`:

```python
import pandas as pd
import pytest

from conftest import seed_dashboard_db
from psu.dashboard.players import COLUMNS, player_table, teams
from psu.db import SPECS, connect, upsert


@pytest.fixture
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_teams_in_a_season(con):
    assert teams(con, 2025) == ["Buffalo", "Ohio State", "Penn State", "Temple"]


def test_passing_season_totals(con):
    p = player_table(con, 2025, "Penn State", "passing")
    assert list(p.columns) == COLUMNS["passing"]
    row = p.iloc[0]
    assert row["player"] == "Penn State QB" and row["games"] == 2
    assert (row["comp"], row["att"], row["yards"], row["td"], row["int"]) == (40, 60, 361, 4, 2)
    assert row["comp_pct"] == pytest.approx(40 / 60) and row["yds_per_att"] == pytest.approx(361 / 60)


def test_team_pseudo_player_is_excluded(con):
    r = player_table(con, 2025, "Penn State", "rushing")
    assert list(r["player"]) == ["Penn State RB"]
    assert (r.iloc[0]["carries"], r.iloc[0]["yards"], r.iloc[0]["long"]) == (30, 181, 25)
    assert r.iloc[0]["yds_per_carry"] == pytest.approx(181 / 30)


def test_minimum_volume_filter(con):
    assert player_table(con, 2025, "Penn State", "receiving", minimum=12).iloc[0]["receptions"] == 12
    empty = player_table(con, 2025, "Penn State", "receiving", minimum=13)
    assert empty.empty and list(empty.columns) == COLUMNS["receiving"]


def test_malformed_stats_count_as_zero(con):
    rows = [
        {"game_id": 101, "season": 2025, "week": 1, "season_type": "regular", "team": "Penn State",
         "category": "passing", "stat_type": stat_type, "athlete_id": "psu-backup", "athlete_name": "Backup QB",
         "stat": stat}
        for stat_type, stat in (("YDS", "--"), ("TD", "1"))  # no C/ATT at all
    ]
    upsert(con, SPECS["player_game_stats"], pd.DataFrame(rows))
    p = player_table(con, 2025, "Penn State", "passing").set_index("player")
    backup = p.loc["Backup QB"]
    assert (backup["att"], backup["yards"], backup["td"]) == (0, 0, 1)
    assert pd.isna(backup["comp_pct"])
    assert player_table(con, 2025, "Penn State", "passing", minimum=1)["player"].tolist() == ["Penn State QB"]


def test_unknown_kind_is_an_error(con):
    with pytest.raises(ValueError, match="kind"):
        player_table(con, 2025, "Penn State", "kicking")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_players.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.dashboard.players'`.

- [ ] **Step 3: Write the implementation**

`src/psu/dashboard/players.py`:

```python
"""Season player tables (passing, rushing, receiving) from CFBD box scores."""
from __future__ import annotations

import duckdb
import pandas as pd

from psu.dashboard.common import require

COLUMNS = {
    "passing": ["player", "games", "comp", "att", "comp_pct", "yards", "yds_per_att", "td", "int"],
    "rushing": ["player", "games", "carries", "yards", "yds_per_carry", "td", "long"],
    "receiving": ["player", "games", "receptions", "yards", "yds_per_rec", "td", "long"],
}
VOLUME = {"passing": "att", "rushing": "carries", "receiving": "receptions"}
_VOLUME_STAT = {"rushing": "CAR", "receiving": "REC"}
_RATE = {"passing": "yds_per_att", "rushing": "yds_per_carry", "receiving": "yds_per_rec"}


def teams(con: duckdb.DuckDBPyConnection, season: int) -> list[str]:
    require(con, "player_game_stats")
    return con.execute(
        "SELECT DISTINCT team FROM player_game_stats WHERE season = ? ORDER BY team", [season]
    ).df()["team"].tolist()


def _num(wide: pd.DataFrame, stat_type: str) -> pd.Series:
    if stat_type not in wide:
        return pd.Series(0.0, index=wide.index)
    return pd.to_numeric(wide[stat_type], errors="coerce").fillna(0.0)


def _per_game(kind: str, wide: pd.DataFrame) -> pd.DataFrame:
    if kind == "passing":
        if "C/ATT" in wide:
            parts = wide["C/ATT"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)").astype(float).fillna(0.0)
        else:
            parts = pd.DataFrame({0: 0.0, 1: 0.0}, index=wide.index)
        return pd.DataFrame({
            "comp": parts[0], "att": parts[1], "yards": _num(wide, "YDS"), "td": _num(wide, "TD"),
            "int": _num(wide, "INT"),
        })
    return pd.DataFrame({
        VOLUME[kind]: _num(wide, _VOLUME_STAT[kind]), "yards": _num(wide, "YDS"), "td": _num(wide, "TD"),
        "long": _num(wide, "LONG"),
    })


def player_table(
    con: duckdb.DuckDBPyConnection, season: int, team: str, kind: str, minimum: int = 0
) -> pd.DataFrame:
    if kind not in COLUMNS:
        raise ValueError(f"kind must be one of {sorted(COLUMNS)}, not {kind!r}")
    require(con, "player_game_stats")
    raw = con.execute(
        "SELECT game_id, athlete_id, athlete_name, stat_type, stat FROM player_game_stats "
        "WHERE season = ? AND team = ? AND category = ?",
        [season, team, kind],
    ).df()
    raw = raw[(raw["athlete_name"].fillna("").str.strip().str.lower() != "team") & raw["athlete_id"].notna()]
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS[kind])
    wide = raw.pivot_table(
        index=["game_id", "athlete_id"], columns="stat_type", values="stat", aggfunc="first"
    ).reset_index()
    names = raw.groupby("athlete_id")["athlete_name"].first()
    per_game = _per_game(kind, wide)
    named = {c: (c, "max" if c == "long" else "sum") for c in per_game.columns}
    named["games"] = ("yards", "size")
    out = (
        per_game.assign(player=wide["athlete_id"].map(names).to_numpy(), athlete_id=wide["athlete_id"].to_numpy())
        .groupby(["athlete_id", "player"], as_index=False)
        .agg(**named)
    )
    volume = out[VOLUME[kind]]
    out[_RATE[kind]] = out["yards"] / volume.where(volume > 0)
    if kind == "passing":
        out["comp_pct"] = out["comp"] / volume.where(volume > 0)
    out = out[volume >= minimum]
    for column in COLUMNS[kind]:
        if column not in ("player", "comp_pct", _RATE[kind]):
            out[column] = out[column].astype(int)
    return out.sort_values(["yards", "player"], ascending=[False, True])[COLUMNS[kind]].reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_players.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/dashboard/players.py tests/test_dashboard_players.py
git commit -m "feat(dashboard): passing, rushing and receiving tables from box scores" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Predictions data (`dashboard/predictions.py`)

**Files:**
- Create: `src/psu/dashboard/predictions.py`
- Test: `tests/test_dashboard_predictions.py`

**Interfaces:**
- Consumes: `psu.dashboard.common.MissingData` and `require` (Task 1); `psu.dashboard.overview.schedule` (Task 2).
- Produces:
  - `upcoming(con, season, team) -> DataFrame[SCHEDULE_COLUMNS]`: the schedule rows not yet completed
  - `sim_summary(con, team) -> dict`, the `sim_team_summary` row. It raises `MissingData` (mentioning `psu simulate`) if the table or the team's row is missing.
  - `sim_win_totals(con, team) -> DataFrame[wins, prob]`
  - `sim_conference(con) -> DataFrame[team, mean_conf_wins, p_title_game, p_conf_champ]`, sorted by `p_conf_champ` descending
  - `next_slate(con, season) -> DataFrame[week, start_date, away_team, home_team, neutral_site, pred_margin, home_win_prob, vegas_margin]`: the earliest upcoming regular-season week

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_predictions.py`:

```python
import pytest

from conftest import seed_dashboard_db
from psu.dashboard.common import MissingData
from psu.dashboard.predictions import next_slate, sim_conference, sim_summary, sim_win_totals, upcoming
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def untrained():
    c = connect(":memory:")
    seed_dashboard_db(c, with_model=False)
    yield c
    c.close()


def test_upcoming_games_only(con):
    u = upcoming(con, 2026, "Penn State")
    assert list(u["opponent"]) == ["Ohio State"] and u.iloc[0]["model_margin"] == pytest.approx(7.0)
    assert upcoming(con, 2025, "Penn State").empty


def test_sim_tables(con):
    s = sim_summary(con, "Penn State")
    assert s["p_cfp"] == pytest.approx(0.4) and s["season"] == 2026 and s["n_sims"] == 1000
    totals = sim_win_totals(con, "Penn State")
    assert list(totals.columns) == ["wins", "prob"] and totals["prob"].sum() == pytest.approx(1.0)
    conf = sim_conference(con)
    assert list(conf["team"]) == ["Ohio State", "Penn State"]


def test_sim_summary_for_a_team_without_a_run(con):
    with pytest.raises(MissingData, match="psu simulate"):
        sim_summary(con, "Temple")


def test_next_slate_is_the_earliest_upcoming_week(con):
    slate = next_slate(con, 2026)
    assert list(slate["week"].unique()) == [2] and len(slate) == 2
    assert list(slate.columns) == [
        "week", "start_date", "away_team", "home_team", "neutral_site", "pred_margin", "home_win_prob", "vegas_margin",
    ]
    assert next_slate(con, 2025).empty


def test_before_train_and_simulate(untrained):
    with pytest.raises(MissingData, match="psu simulate"):
        sim_summary(untrained, "Penn State")
    with pytest.raises(MissingData, match="psu train"):
        next_slate(untrained, 2026)
    assert upcoming(untrained, 2026, "Penn State")["model_margin"].isna().all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_predictions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.dashboard.predictions'`.

- [ ] **Step 3: Write the implementation**

`src/psu/dashboard/predictions.py`:

```python
"""Predictions data: upcoming games with model and Vegas lines, the season simulation, and the next FBS slate."""
from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from psu.dashboard.common import MissingData, require
from psu.dashboard.overview import schedule


def upcoming(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    games = schedule(con, season, team)
    return games[~games["completed"]].reset_index(drop=True)


def sim_summary(con: duckdb.DuckDBPyConnection, team: str) -> dict[str, Any]:
    require(con, "sim_team_summary")
    df = con.execute("SELECT * FROM sim_team_summary WHERE team = ?", [team]).df()
    if df.empty:
        raise MissingData(f"No season simulation for {team}; run `psu simulate` first")
    return df.iloc[0].to_dict()


def sim_win_totals(con: duckdb.DuckDBPyConnection, team: str) -> pd.DataFrame:
    require(con, "sim_win_totals")
    return con.execute("SELECT wins, prob FROM sim_win_totals WHERE team = ? ORDER BY wins", [team]).df()


def sim_conference(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    require(con, "sim_conference")
    return con.execute(
        "SELECT team, mean_conf_wins, p_title_game, p_conf_champ FROM sim_conference "
        "ORDER BY p_conf_champ DESC, p_title_game DESC, team"
    ).df()


def next_slate(con: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
    require(con, "game_predictions")
    return con.execute(
        """
        SELECT week, start_date, away_team, home_team, neutral_site, pred_margin, home_win_prob, vegas_margin
        FROM game_predictions
        WHERE season = $season AND split = 'upcoming' AND season_type = 'regular'
          AND week = (SELECT min(week) FROM game_predictions
                      WHERE season = $season AND split = 'upcoming' AND season_type = 'regular')
        ORDER BY start_date, home_team
        """,
        {"season": season},
    ).df()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_predictions.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/dashboard/predictions.py tests/test_dashboard_predictions.py
git commit -m "feat(dashboard): upcoming games, simulation results and next slate" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Streamlit app and pages (`dashboard/ui.py`, `app/`)

**Files:**
- Create: `src/psu/dashboard/ui.py`
- Create: `app/streamlit_app.py`
- Create: `app/views/overview.py`, `app/views/trends.py`, `app/views/games.py`, `app/views/players.py`, `app/views/predictions.py`
- Test: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: every `psu.dashboard.*` function from Tasks 1-6 (called by name through `load`); `psu.simulate.load_sigma` and `MissingModel` (Phase 4); conftest `seed_dashboard_db`.
- Produces:
  - `load(name, *args)`, where `name` is `"<module>.<function>"` in `psu.dashboard`
  - `load_or_stop(name, *args)`: shows `st.info` and stops the page on `MissingData`
  - `load_or_note(name, *args)`: shows `st.info` and returns `None` on `MissingData`
  - `season_picker() -> int`: a sidebar selectbox plus a "Refresh data" button
  - `app/streamlit_app.py` (navigation) and the 5 view scripts

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_app.py`:

```python
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import seed_dashboard_db
from psu.db import connect

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["overview", "trends", "games", "players", "predictions"]
REAL_DB = ROOT / "data" / "psu.duckdb"


def _seed(tmp_path, monkeypatch, *, with_model=True):
    path = tmp_path / "psu.duckdb"
    con = connect(path)
    try:
        seed_dashboard_db(con, with_model=with_model)
    finally:
        con.close()
    if with_model:
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "game_model.json").write_text('{"sigma": 16.0}', encoding="utf-8")
    monkeypatch.setenv("PSU_DB_PATH", str(path))
    return path


def _run(view, season=None):
    at = AppTest.from_file(str(ROOT / "app" / "views" / f"{view}.py"), default_timeout=60)
    at.run()
    if season is not None:
        at.sidebar.selectbox[0].set_value(season).run()
    return at


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders_for_the_latest_season(tmp_path, monkeypatch, view):
    _seed(tmp_path, monkeypatch)
    at = _run(view)  # default season 2027: one unplayed game
    assert not at.exception, at.exception
    assert at.title


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders_for_a_season_in_progress(tmp_path, monkeypatch, view):
    _seed(tmp_path, monkeypatch)
    at = _run(view, season=2026)
    assert not at.exception, at.exception


def test_overview_shows_record_and_schedule(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("overview", season=2025)
    assert [m.value for m in at.metric][:2] == ["2-0", "1-0"]
    assert len(at.dataframe) >= 1


def test_game_explorer_shows_win_probability_for_the_first_game(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("games", season=2025)
    assert not at.exception
    assert at.selectbox[0].value == 101
    assert any("Win probability" in s.value for s in at.subheader)


def test_navigation_entry_point_runs(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("view", VIEWS)
def test_pages_explain_a_missing_database(tmp_path, monkeypatch, view):
    monkeypatch.setenv("PSU_DB_PATH", str(tmp_path / "missing.duckdb"))
    at = _run(view)
    assert not at.exception
    assert any("psu ingest" in i.value for i in at.info)


@pytest.mark.parametrize("view", VIEWS)
def test_pages_work_before_train_and_simulate(tmp_path, monkeypatch, view):
    _seed(tmp_path, monkeypatch, with_model=False)
    at = _run(view, season=2026)
    assert not at.exception, at.exception


def test_predictions_page_names_the_missing_commands(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, with_model=False)
    at = _run("predictions", season=2026)
    text = " ".join(i.value for i in at.info)
    assert "psu simulate" in text and "psu train" in text


def test_refresh_button_clears_cache(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("overview")
    at.sidebar.button[0].click().run()
    assert not at.exception


@pytest.mark.skipif(not REAL_DB.exists(), reason="needs data/psu.duckdb")
@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders_on_the_real_database(monkeypatch, view):
    monkeypatch.setenv("PSU_DB_PATH", str(REAL_DB))
    at = _run(view)
    assert not at.exception, at.exception
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_app.py -v -x`
Expected: FAIL, because AppTest can't find `app/views/overview.py` (or the run raises).

- [ ] **Step 3: Write `src/psu/dashboard/ui.py`**

```python
"""Streamlit helpers shared by the pages: cached reads, the season picker, and friendly missing-data messages."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import streamlit as st

from psu.dashboard.common import MissingData, db_path, read


@st.cache_data(ttl=600, show_spinner=False)
def _load(db: str, name: str, *args: Any) -> Any:
    module, func = name.split(".")
    return read(getattr(importlib.import_module(f"psu.dashboard.{module}"), func), *args, path=Path(db))


def load(name: str, *args: Any) -> Any:
    """Run psu.dashboard.<module>.<function>(con, *args) on a fresh read-only connection, cached for 10 minutes."""
    return _load(str(db_path()), name, *args)


def load_or_stop(name: str, *args: Any) -> Any:
    try:
        return load(name, *args)
    except MissingData as e:
        st.info(str(e))
        st.stop()


def load_or_note(name: str, *args: Any) -> Any:
    try:
        return load(name, *args)
    except MissingData as e:
        st.info(str(e))
        return None


def season_picker() -> int:
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
    options = load_or_stop("common.seasons")
    if not options:
        st.info("No games loaded yet; run `psu ingest` first")
        st.stop()
    return st.sidebar.selectbox("Season", options, index=0)
```

- [ ] **Step 4: Write the entry point and the five views**

`app/streamlit_app.py`:

```python
"""Penn State football dashboard. Run: streamlit run app/streamlit_app.py (reads data/psu.duckdb; no API calls)."""
import streamlit as st

st.set_page_config(page_title="Penn State analytics", layout="wide")
st.navigation([
    st.Page("views/overview.py", title="Season overview", default=True),
    st.Page("views/trends.py", title="Efficiency trends"),
    st.Page("views/games.py", title="Game explorer"),
    st.Page("views/players.py", title="Players"),
    st.Page("views/predictions.py", title="Predictions"),
]).run()
```

`app/views/overview.py`:

```python
"""Season overview: record, schedule with results and pregame lines, and key metrics against conference and FBS."""
import pandas as pd
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load, load_or_note, load_or_stop, season_picker

season = season_picker()
st.title(f"{TEAM} {season}")

schedule = load_or_stop("overview.schedule", season, TEAM)
if schedule.empty:
    st.info(f"No {TEAM} games found for {season}.")
    st.stop()

rec = load_or_stop("overview.record", season, TEAM)
record_col, conf_col, left_col = st.columns(3)
record_col.metric("Record", f"{rec['wins']}-{rec['losses']}")
conf_col.metric("Conference", f"{rec['conf_wins']}-{rec['conf_losses']}")
left_col.metric("Games left", rec["remaining"])

st.subheader("Schedule")
score = schedule["team_points"].astype("Int64").astype(str) + "-" + schedule["opp_points"].astype("Int64").astype(str)
table = pd.DataFrame({
    "Week": schedule["week"],
    "Date": pd.to_datetime(schedule["start_date"]).dt.date,
    "Opponent": schedule["opponent"],
    "Venue": schedule["venue"].str.title(),
    "Result": (schedule["result"].fillna("") + " " + score).where(schedule["completed"], ""),
    "Vegas": schedule["vegas_margin"],
    "Model": schedule["model_margin"],
    "Win prob": schedule["win_prob"] * 100,
})
st.dataframe(table, hide_index=True, column_config={
    "Vegas": st.column_config.NumberColumn(format="%+.1f"),
    "Model": st.column_config.NumberColumn(format="%+.1f"),
    "Win prob": st.column_config.NumberColumn(format="%.0f%%"),
})
st.caption(
    f"Lines are from {TEAM}'s side: +7 means {TEAM} is favoured by 7. "
    "For games already played the model line is in-sample, so it flatters the model."
)

st.subheader("Key metrics")
metrics = load_or_note("overview.metric_comparison", season, TEAM)
if metrics is not None:
    conference = load("common.team_conference", season, TEAM) or "Conference"
    shown = metrics.drop(columns="higher_is_better").rename(columns={
        "metric": "Metric", "value": TEAM, "conference_avg": f"{conference} avg", "national_avg": "FBS avg",
    })
    number = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(shown, hide_index=True, column_config={c: number for c in shown.columns if c != "Metric"})
    st.caption("For defense EPA/play and success rate, lower is better. Blank means no plays yet this season.")
```

`app/views/trends.py`:

```python
"""Efficiency trends: EPA/play and success rate by season, and game by game within a season."""
import altair as alt
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_note, season_picker

season = season_picker()
st.title("Efficiency trends")

st.subheader("By season")
trends = load_or_note("trends.season_trends", TEAM)
if trends is not None:
    if trends.empty:
        st.info("No season metrics yet; run `psu build` first.")
    else:
        chart = (
            alt.Chart(trends)
            .mark_line(point=True)
            .encode(
                x=alt.X("season:O", title="Season"),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("group:N", title=None),
                tooltip=["season:O", "group:N", alt.Tooltip("value:Q", format=".3f")],
            )
            .properties(width=320, height=220)
            .facet(facet=alt.Facet("metric:N", title=None), columns=2)
            .resolve_scale(y="independent")
        )
        st.altair_chart(chart)

st.subheader(f"Game by game, {season}")
weekly = load_or_note("trends.weekly_trends", season, TEAM)
if weekly is not None:
    if weekly.empty:
        st.info(f"No {TEAM} plays for {season} yet.")
    else:
        order = list(dict.fromkeys(weekly["game"]))
        for column, title in (("epa_per_play", "EPA/play"), ("success_rate", "Success rate")):
            chart = (
                alt.Chart(weekly)
                .mark_line(point=True)
                .encode(
                    x=alt.X("game:N", sort=order, title=None),
                    y=alt.Y(f"{column}:Q", title=title),
                    color=alt.Color("side:N", title=None),
                    tooltip=["game:N", "side:N", alt.Tooltip(f"{column}:Q", format=".3f"), "plays:Q"],
                )
            )
            st.altair_chart(chart)
        st.caption("Garbage time is excluded. For the defense, lower is better.")
```

`app/views/games.py`:

```python
"""Game explorer: line score, box score, drive chart, win probability and biggest plays for one game."""
import altair as alt
import streamlit as st

from psu.config import TEAM
from psu.dashboard.common import db_path
from psu.dashboard.ui import load, load_or_note, load_or_stop, season_picker
from psu.simulate import MissingModel, load_sigma

season = season_picker()
st.title("Game explorer")

options = load_or_stop("games.game_options", season, TEAM)
if options.empty:
    st.info(f"No completed {TEAM} games in {season} yet.")
    st.stop()
labels = dict(zip(options["game_id"].tolist(), options["label"].tolist()))
game_id = st.selectbox("Game", list(labels), format_func=labels.get)

scores = load("games.line_scores", game_id)
st.dataframe(scores, hide_index=True)
home = scores["team"].iloc[-1]

box_col, drive_col = st.columns(2)
with box_col:
    st.subheader("Box score")
    box = load_or_note("games.box_score", game_id)
    if box is not None:
        st.dataframe(box, hide_index=True)
with drive_col:
    st.subheader("Drives")
    drives = load_or_note("games.drive_chart", game_id)
    if drives is not None and not drives.empty:
        chart = (
            alt.Chart(drives)
            .mark_bar()
            .encode(
                x=alt.X("start_pos:Q", title="Yards from own goal", scale=alt.Scale(domain=[0, 100])),
                x2="end_pos:Q",
                y=alt.Y("drive_number:O", title="Drive"),
                color=alt.Color("offense:N", title=None),
                tooltip=["drive_number:O", "offense:N", "quarter:Q", "result:N", "plays:Q", "yards:Q", "time:N"],
            )
        )
        st.altair_chart(chart)

st.subheader("Win probability")
try:
    sigma = load_sigma(db_path().parent)
except MissingModel as e:
    st.info(str(e))
else:
    wp = load_or_note("games.win_probability", game_id, sigma)
    if wp is not None:
        chart = (
            alt.Chart(wp)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X("minute:Q", title="Minute", scale=alt.Scale(domain=[0, 60])),
                y=alt.Y("home_wp:Q", title=f"{home} win probability", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("minute:Q", format=".1f"), "home_margin:Q",
                         alt.Tooltip("home_wp:Q", format=".0%")],
            )
        )
        st.altair_chart(chart)
        st.caption("Estimated from the score, time left and the model's pregame line (not a play-level model).")

st.subheader("Biggest plays")
plays = load_or_note("games.top_plays", game_id)
if plays is not None:
    st.dataframe(plays, hide_index=True, column_config={"epa": st.column_config.NumberColumn("EPA", format="%+.2f")})
```

`app/views/players.py`:

```python
"""Players: season passing, rushing and receiving tables from box scores."""
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_stop, season_picker

KINDS = {"passing": ("Minimum attempts", 20), "rushing": ("Minimum carries", 10), "receiving": ("Minimum catches", 5)}

season = season_picker()
st.title("Players")

teams = load_or_stop("players.teams", season)
if not teams:
    st.info(f"No player box scores for {season} yet.")
    st.stop()
team = st.sidebar.selectbox("Team", teams, index=teams.index(TEAM) if TEAM in teams else 0)
kind = st.radio("Stat", list(KINDS), horizontal=True, format_func=str.title)
label, default = KINDS[kind]
minimum = st.slider(label, 0, 200, default)

table = load_or_stop("players.player_table", season, team, kind, minimum)
if table.empty:
    st.info("No players meet the minimum.")
else:
    rate = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(table, hide_index=True, column_config={
        "comp_pct": st.column_config.NumberColumn("Comp %", format="percent"),
        "yds_per_att": rate, "yds_per_carry": rate, "yds_per_rec": rate,
    })
st.caption("From CFBD box scores. Plays don't name players, so per-player EPA isn't available.")
```

`app/views/predictions.py`:

```python
"""Predictions: upcoming games, the season simulation and the next FBS slate."""
import altair as alt
import pandas as pd
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_note, season_picker

season = season_picker()
st.title("Predictions")

st.subheader(f"{TEAM} upcoming games")
games = load_or_note("predictions.upcoming", season, TEAM)
if games is not None:
    if games.empty:
        st.info(f"No upcoming {TEAM} games in {season}.")
    else:
        st.dataframe(pd.DataFrame({
            "Week": games["week"], "Date": pd.to_datetime(games["start_date"]).dt.date,
            "Opponent": games["opponent"], "Venue": games["venue"].str.title(),
            "Model": games["model_margin"], "Win prob": games["win_prob"] * 100, "Vegas": games["vegas_margin"],
        }), hide_index=True, column_config={
            "Model": st.column_config.NumberColumn(format="%+.1f"),
            "Vegas": st.column_config.NumberColumn(format="%+.1f"),
            "Win prob": st.column_config.NumberColumn(format="%.0f%%"),
        })

st.subheader("Season simulation")
summary = load_or_note("predictions.sim_summary", TEAM)
if summary is not None:
    tiles = st.columns(5)
    tiles[0].metric("Mean wins", f"{summary['mean_wins']:.1f}")
    for tile, (label, key) in zip(tiles[1:], (
        ("P(10+ wins)", "p_10_plus"), ("P(title game)", "p_title_game"),
        ("P(Big Ten champ)", "p_conf_champ"), ("P(CFP)", "p_cfp"),
    )):
        tile.metric(label, f"{summary[key]:.0%}")
    as_of = pd.Timestamp(summary["as_of"]).date() if pd.notna(summary["as_of"]) else "preseason"
    st.caption(
        f"{int(summary['n_sims']):,} simulated {int(summary['season'])} seasons (tau {summary['tau']:g}), "
        f"results through {as_of}. Re-run `psu simulate` after new games."
    )
    totals = load_or_note("predictions.sim_win_totals", TEAM)
    if totals is not None and not totals.empty:
        st.altair_chart(
            alt.Chart(totals)
            .mark_bar()
            .encode(
                x=alt.X("wins:O", title="Regular-season wins"),
                y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format="%")),
                tooltip=["wins:O", alt.Tooltip("prob:Q", format=".1%")],
            )
        )
    race = load_or_note("predictions.sim_conference")
    if race is not None:
        st.markdown("**Big Ten title race**")
        percent = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(race.assign(p_title_game=race["p_title_game"] * 100, p_conf_champ=race["p_conf_champ"] * 100),
                     hide_index=True, column_config={
                         "mean_conf_wins": st.column_config.NumberColumn("Mean conf wins", format="%.2f"),
                         "p_title_game": percent, "p_conf_champ": percent,
                     })

with st.expander("Next week's FBS games"):
    slate = load_or_note("predictions.next_slate", season)
    if slate is not None:
        if slate.empty:
            st.info(f"No upcoming games in {season}.")
        else:
            st.dataframe(slate, hide_index=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_app.py -v`
Expected: all pass. The real-database tests run when `data/psu.duckdb` exists, and are skipped otherwise.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python -m pytest`
Expected: everything passes.

- [ ] **Step 7: Commit**

```bash
git add src/psu/dashboard/ui.py app/streamlit_app.py app/views/overview.py app/views/trends.py app/views/games.py app/views/players.py app/views/predictions.py tests/test_dashboard_app.py
git commit -m "feat(app): five-page Streamlit dashboard" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Browser check, README, PR (orchestrator)

**Files:**
- Create: `.claude/launch.json`
- Modify: `README.md` (add a "Phase 5: Dashboard" section after Phase 4)

- [ ] **Step 1: Preview configuration**

`.claude/launch.json`:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "dashboard",
      "runtimeExecutable": ".venv/Scripts/streamlit.exe",
      "runtimeArgs": ["run", "app/streamlit_app.py", "--server.headless", "true", "--server.port", "8501"],
      "port": 8501
    }
  ]
}
```

- [ ] **Step 2: Walk every page on the real database**

Start the `dashboard` preview and open each of the five pages in the browser pane, for the current season and for 2025. Check the server logs for errors. Take a screenshot of the overview and predictions pages for the PR. Stop the preview afterwards.

This is a read-only check: the dashboard never writes to the database.

- [ ] **Step 3: README section**

Add after the Phase 4 section:

````markdown
## Phase 5: Dashboard

```
.venv\Scripts\streamlit run app\streamlit_app.py
```

The dashboard opens in your browser. It reads `data/psu.duckdb` read-only and makes no API calls. The pages:
- **Season overview:** record, schedule with Vegas and model lines, and key metrics vs. the Big Ten and FBS.
- **Efficiency trends:** EPA/play and success rate by season, and game by game (garbage time excluded).
- **Game explorer:** line score, box score, drive chart, win probability and the biggest plays.
- **Players:** passing, rushing and receiving tables from box scores, for any team.
- **Predictions:** upcoming games, the season simulation and next week's FBS slate.

Notes:
- Results are cached for 10 minutes. Use **Refresh data** in the sidebar after re-running `psu build`,
  `psu train` or `psu simulate`. If a page says a table is missing, it names the command to run.
- The win-probability chart is estimated from the score, the time left and the model's pregame line.
  It isn't a play-level model.
- Player tables use box-score efficiency. Plays don't name players, so per-player EPA isn't available.
- Set `PSU_DB_PATH` to point the dashboard at a different database file.
````

- [ ] **Step 4: Commit, push, PR**

```bash
git add .claude/launch.json README.md
git commit -m "docs: add Phase 5 dashboard guide and preview config" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/phase5-dashboard
```

Open a PR into `main` with `C:\Program Files\GitHub CLI\gh.exe`. The body should include a summary, the page list, the design decisions table, the test plan, and the screenshots. Don't merge; stop for the user's review.
