# Phase 2 — Metrics Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **In this repo:** implementers and reviewers follow `.claude/agents/psu-implementer.md` and `.claude/agents/psu-reviewer.md`. Give each agent only its own task, plus **Global Constraints** and **Review Focus**.

**Goal:** `psu build` turns the Phase 1 raw tables into analysis tables for every FBS team:
- EPA/play (overall, rush, pass), success rate, explosiveness, explosive rate, and third-down rate
- havoc rate, turnover margin, and red-zone efficiency
- splits by down, quarter, score state, venue and opponent conference
- opponent-adjusted EPA/play and success rate

Penn State's season totals match CFBD's advanced stats.

**Architecture:** four pure-pandas modules, each testable on tiny hand-made frames:
- `transform.py`: filters to scrimmage plays and adds rush/pass, success, explosive, turnover, garbage-time and context columns.
- `metrics.py`: group-by aggregations and box-score metrics.
- `adjust.py`: per-season ridge regression on offense and defense team effects.
- `build.py`: reads the raw DuckDB tables, runs the other three, and writes the derived tables back with `CREATE OR REPLACE`. It's the only one that touches DuckDB, and `psu build` in `cli.py` wraps it.

**Tech Stack:** Python ≥3.11, pandas 3.x, numpy, scipy (sparse), scikit-learn (`Ridge`), DuckDB 1.5.

**Spec:** `docs/superpowers/specs/psu-analytics-build-plan.md` ("Phase 2 — Metrics layer")

## Global Constraints

- Branch `feat/phase2-metrics`. One Conventional Commit per task (message given in the dispatch), staging only that task's files by explicit path, and ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Phase 2 makes no CFBD API calls at all.** Everything reads `data/psu.duckdb`. Never run `psu ingest`.
- Unit tests use hand-made DataFrames or an in-memory DuckDB. Only `tests/test_validation.py` reads the real `data/psu.duckdb`, and it skips itself when the file is absent.
- Definitions (verified against CFBD for Penn State 2024: play counts and EPA/play match exactly; success rate, explosiveness and havoc are within 0.012):
  - **Scrimmage play:** `play_type` in `SCRIMMAGE_TYPES` and `ppa` not null.
  - **EPA:** CFBD's `ppa` column.
  - **Success:** at least 50% of `distance` gained on 1st down, 70% on 2nd, 100% on 3rd and 4th. Offensive TDs always succeed; turnovers never do.
  - **Explosiveness:** mean EPA of successful plays.
  - **Explosive play:** rush ≥ 12 yards or pass ≥ 16 yards (configurable), never a turnover.
  - **Havoc:** (TFL + passes defended + interceptions + opponent fumbles) ÷ defensive scrimmage plays, from `team_game_stats`.
  - **Garbage time:** |margin| > 38 in Q2, > 28 in Q3, > 22 in Q4. Never in Q1 or OT. Thresholds are configurable (`--garbage 38,28,22` or `--garbage off`).
- Season-level efficiency excludes garbage time by default (`exclude_garbage=True`). The CFBD comparison uses `exclude_garbage=False`, because CFBD's advanced stats include garbage time.
- Output tables contain FBS teams only (a team is FBS in a season if `games` lists it with classification `fbs`). Opponent adjustment is fit on all teams, including FCS opponents.
- New runtime dependencies: `scikit-learn>=1.5`, `scipy>=1.11`.
- Known data quirks that Phase 2 must not trip over:
  - `ratings_sp` has a `nationalAverages` row (not used here).
  - `player_game_stats` has negative-`athlete_id` team rows (not used here).
  - About 26% of `plays` have null `ppa`; these are non-scrimmage plays and are filtered out.

## Review Focus

1. **Garbage-time boundaries:** a margin exactly at a threshold is not garbage, the sign of the margin doesn't matter, and OT is never garbage. Tested in Task 1 (`test_garbage_time_default_thresholds`).
2. **A team with no plays of one kind** (e.g. no rush plays, or no third downs in a split) must get NaN for that metric, not an error or a 0. Tested in Task 2 (`test_missing_play_class_gives_nan_not_error`).
3. **Box-score categories missing for a team in a game** (e.g. no `passesDeflected` row) count as 0, and the game isn't dropped. Tested in Task 2 (`test_havoc_rate_counts_box_score_events_and_forced_fumbles`, `test_turnover_margin`).
4. **Neutral-site games** are neither home nor away, in both the venue split and the adjustment's home-field term. Tested in Task 1 (`test_context_columns`) and Task 3 (`test_home_field_term_does_not_bias_neutral_games`).
5. **FCS opponents** stay in the regression but never appear in output tables, and re-running `psu build` replaces the tables rather than appending to them. Tested in Task 4 (`test_build_writes_fbs_only_tables_and_is_repeatable`).

## File Map

| File | Responsibility |
|---|---|
| `src/psu/transform.py` | play-type sets, `GarbageTime`, `Explosive`, `is_success`, `is_garbage`, `score_state`, `play_class`, `enrich_plays` |
| `src/psu/metrics.py` | `efficiency`, `havoc_rate`, `turnover_margin`, `red_zone` |
| `src/psu/adjust.py` | `opponent_adjust` |
| `src/psu/build.py` | `fbs_teams`, `build` (writes the derived tables) |
| `src/psu/cli.py` | adds `psu build` |
| `tests/conftest.py` | adds `seed_raw_tables(con)` (a tiny raw dataset in DuckDB) |
| `tests/test_validation.py` | PSU vs CFBD advanced stats on the real database |

---

### Task 1: Play enrichment (`transform.py`)

**Files:**
- Create: `src/psu/transform.py`
- Test: `tests/test_transform.py`

**Interfaces:**
- Consumes: raw `plays` columns (`id, game_id, drive_id, season, week, season_type, offense, offense_conference, defense, defense_conference, home, away, period, down, distance, yards_to_goal, yards_gained, play_type, play_text, ppa, offense_score, defense_score`) and `games` columns (`id, neutral_site`).
- Produces:
  - `RUSH_TYPES`, `PASS_TYPES`, `FUMBLE_TYPES`, `SCRIMMAGE_TYPES`, `TURNOVER_TYPES`, `OFFENSIVE_TD_TYPES` (frozensets), `SCORE_STATES` (tuple)
  - `@dataclass(frozen=True) GarbageTime(q1=None, q2=38, q3=28, q4=22)` with `GarbageTime.parse(spec: str) -> GarbageTime`
  - `@dataclass(frozen=True) Explosive(rush_yards=12, pass_yards=16)`
  - `success_threshold(down, distance) -> np.ndarray`, `is_success(plays: DataFrame) -> Series[bool]`, `is_garbage(period: Series, margin: Series, garbage: GarbageTime) -> Series[bool]`, `score_state(margin: Series) -> Series[str]`, `play_class(play_type: Series, play_text: Series) -> Series[str]`
  - `ENRICHED_COLUMNS: list[str]`
  - `enrich_plays(plays, games, *, garbage=GarbageTime(), explosive=Explosive()) -> DataFrame`. Returns exactly `ENRICHED_COLUMNS`, with `play_class` ∈ {rush, pass}, bool `success`/`explosive`/`turnover`/`garbage`, int `margin` (offense minus defense score), `score_state`, `quarter` ∈ {"1","2","3","4","OT"}, and `venue` ∈ {home, away, neutral} from the offense's point of view.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, extend `dependencies` to:
```toml
dependencies = [
    "cfbd>=5.29,<6",
    "duckdb>=1.1",
    "pandas>=2.2",
    "python-dotenv>=1.0",
    "scikit-learn>=1.5",
    "scipy>=1.11",
]
```
Run: `.venv\Scripts\python -m pip install -e ".[dev]"`
Expected: it ends with `Successfully installed ...` (scikit-learn and scipy included).

- [ ] **Step 2: Write the failing tests**

`tests/test_transform.py`:
```python
import pandas as pd
import pytest

from psu.transform import Explosive, GarbageTime, enrich_plays, is_garbage, is_success, score_state

BASE = dict(
    game_id=1, drive_id="d1", season=2024, week=1, season_type="regular",
    offense="Penn State", offense_conference="Big Ten", defense="Opp", defense_conference="MAC",
    home="Penn State", away="Opp", period=1, down=1, distance=10, yards_to_goal=75, yards_gained=5,
    play_type="Rush", play_text="RB One run for 5 yds", ppa=0.1, offense_score=0, defense_score=0,
)
GAMES = pd.DataFrame({"id": [1, 2], "neutral_site": [False, True]})


def plays(*overrides):
    return pd.DataFrame([{**BASE, "id": str(i), **o} for i, o in enumerate(overrides)])


@pytest.mark.parametrize(
    "down,distance,gained,expected",
    [(1, 10, 5, True), (1, 10, 4, False), (2, 10, 7, True), (2, 10, 6, False),
     (3, 4, 4, True), (3, 4, 3, False), (4, 1, 1, True), (4, 1, 0, False)],
)
def test_success_thresholds(down, distance, gained, expected):
    df = plays({"down": down, "distance": distance, "yards_gained": gained})
    assert bool(is_success(df).iloc[0]) is expected


def test_touchdowns_always_succeed_and_turnovers_never_do():
    df = plays(
        {"play_type": "Rushing Touchdown", "down": 3, "distance": 10, "yards_gained": 2},
        {"play_type": "Pass Interception Return", "yards_gained": 25},
        {"play_type": "Fumble Recovery (Opponent)", "yards_gained": 8},
    )
    assert list(is_success(df)) == [True, False, False]


@pytest.mark.parametrize(
    "period,margin,expected",
    [(1, 50, False), (2, 38, False), (2, 39, True), (3, 29, True), (3, -29, True),
     (3, 28, False), (4, 23, True), (4, 22, False), (5, 40, False)],
)
def test_garbage_time_default_thresholds(period, margin, expected):
    got = is_garbage(pd.Series([period]), pd.Series([margin]), GarbageTime())
    assert bool(got.iloc[0]) is expected


def test_garbage_time_is_configurable():
    custom = GarbageTime.parse("30,20,10")
    assert custom == GarbageTime(None, 30, 20, 10)
    assert bool(is_garbage(pd.Series([4]), pd.Series([11]), custom).iloc[0])
    assert not bool(is_garbage(pd.Series([4]), pd.Series([60]), GarbageTime.parse("off")).iloc[0])
    with pytest.raises(ValueError):
        GarbageTime.parse("30,20")


def test_score_state_buckets():
    got = score_state(pd.Series([-9, -8, -1, 0, 1, 8, 9]))
    assert list(got) == ["down 9+", "down 1-8", "down 1-8", "tied", "up 1-8", "up 1-8", "up 9+"]


def test_enrich_filters_to_scrimmage_plays_with_ppa():
    df = plays({"play_type": "Rush"}, {"play_type": "Punt"}, {"play_type": "Timeout"}, {"play_type": "Rush", "ppa": None})
    assert list(enrich_plays(df, GAMES)["id"]) == ["0"]


def test_enrich_classifies_rush_pass_and_fumbles():
    df = plays(
        {"play_type": "Rush"},
        {"play_type": "Sack"},
        {"play_type": "Pass Incompletion"},
        {"play_type": "Fumble Recovery (Own)", "play_text": "QB One pass complete to WR Two, fumbled"},
        {"play_type": "Fumble Recovery (Own)", "play_text": "RB One run for 3 yds, fumbled"},
    )
    assert list(enrich_plays(df, GAMES)["play_class"]) == ["rush", "pass", "pass", "pass", "rush"]


def test_explosive_uses_separate_rush_and_pass_thresholds():
    df = plays(
        {"play_type": "Rush", "yards_gained": 12},
        {"play_type": "Rush", "yards_gained": 11},
        {"play_type": "Pass Reception", "yards_gained": 16},
        {"play_type": "Pass Reception", "yards_gained": 15},
    )
    assert list(enrich_plays(df, GAMES)["explosive"]) == [True, False, True, False]
    custom = enrich_plays(df, GAMES, explosive=Explosive(rush_yards=10, pass_yards=20))
    assert list(custom["explosive"]) == [True, True, False, False]


def test_context_columns():
    df = plays(
        {"offense_score": 21, "defense_score": 0, "period": 3},
        {"offense": "Opp", "defense": "Penn State", "offense_score": 0, "defense_score": 35, "period": 3},
        {"game_id": 2, "period": 5},
    )
    out = enrich_plays(df, GAMES)
    assert list(out["venue"]) == ["home", "away", "neutral"]
    assert list(out["score_state"]) == ["up 9+", "down 9+", "tied"]
    assert list(out["garbage"]) == [False, True, False]
    assert list(out["quarter"]) == ["3", "3", "OT"]
    assert list(out["margin"]) == [21, -35, 0]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_transform.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.transform'`

- [ ] **Step 4: Implement `src/psu/transform.py`**

```python
"""Turn raw play-by-play into analysis-ready scrimmage plays.

Adds the rush/pass class, success, explosive, turnover and garbage-time flags,
plus game context (score state, quarter, venue) used by metrics and adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RUSH_TYPES = frozenset({"Rush", "Rushing Touchdown"})
PASS_TYPES = frozenset({
    "Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
})
FUMBLE_TYPES = frozenset({"Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown"})
SCRIMMAGE_TYPES = RUSH_TYPES | PASS_TYPES | FUMBLE_TYPES | {"Safety"}
TURNOVER_TYPES = frozenset({
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Opponent)", "Fumble Return Touchdown",
})
OFFENSIVE_TD_TYPES = frozenset({"Rushing Touchdown", "Passing Touchdown"})
SCORE_STATES = ("down 9+", "down 1-8", "tied", "up 1-8", "up 9+")

ENRICHED_COLUMNS = [
    "id", "game_id", "drive_id", "season", "week", "season_type",
    "offense", "offense_conference", "defense", "defense_conference",
    "period", "quarter", "down", "distance", "yards_to_goal", "yards_gained", "play_type", "ppa",
    "play_class", "success", "explosive", "turnover", "margin", "score_state", "garbage", "venue",
]


@dataclass(frozen=True)
class GarbageTime:
    """A play is garbage time when |score margin| is greater than its quarter's threshold. None = never."""

    q1: int | None = None
    q2: int | None = 38
    q3: int | None = 28
    q4: int | None = 22

    @classmethod
    def parse(cls, spec: str) -> GarbageTime:
        """Parse "38,28,22" (Q2,Q3,Q4 margins) or "off"."""
        if spec.strip().lower() == "off":
            return cls(None, None, None, None)
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected Q2,Q3,Q4 margins like '38,28,22' or 'off', got {spec!r}")
        q2, q3, q4 = (int(p) for p in parts)
        return cls(None, q2, q3, q4)


@dataclass(frozen=True)
class Explosive:
    rush_yards: int = 12
    pass_yards: int = 16


def success_threshold(down, distance) -> np.ndarray:
    down = np.asarray(down)
    fraction = np.select([down == 1, down == 2], [0.5, 0.7], default=1.0)
    return fraction * np.asarray(distance, dtype=float)


def is_success(plays: pd.DataFrame) -> pd.Series:
    """>=50% of yards to go on 1st down, >=70% on 2nd, 100% on 3rd/4th.

    Offensive touchdowns always succeed; turnovers never do.
    """
    gained = plays["yards_gained"] >= success_threshold(plays["down"], plays["distance"])
    touchdown = plays["play_type"].isin(OFFENSIVE_TD_TYPES)
    return (gained | touchdown) & ~plays["play_type"].isin(TURNOVER_TYPES)


def is_garbage(period: pd.Series, margin: pd.Series, garbage: GarbageTime) -> pd.Series:
    limits = pd.to_numeric(
        period.map({1: garbage.q1, 2: garbage.q2, 3: garbage.q3, 4: garbage.q4}), errors="coerce"
    )
    # NaN limit (overtime, or a quarter with no threshold) compares False: never garbage time.
    return (margin.abs() > limits).astype(bool)


def score_state(margin: pd.Series) -> pd.Series:
    buckets = np.select([margin <= -9, margin < 0, margin == 0, margin < 9], list(SCORE_STATES[:4]), default=SCORE_STATES[4])
    return pd.Series(buckets, index=margin.index)


def play_class(play_type: pd.Series, play_text: pd.Series) -> pd.Series:
    """Rush or pass. Fumbles and safeties are classed by their play text."""
    text = play_text.fillna("").str.lower()
    ambiguous = ~play_type.isin(RUSH_TYPES | PASS_TYPES)
    passing = play_type.isin(PASS_TYPES) | (ambiguous & (text.str.contains("pass") | text.str.contains("sack")))
    return pd.Series(np.where(passing, "pass", "rush"), index=play_type.index)


def enrich_plays(
    plays: pd.DataFrame,
    games: pd.DataFrame,
    *,
    garbage: GarbageTime = GarbageTime(),
    explosive: Explosive = Explosive(),
) -> pd.DataFrame:
    df = plays[plays["play_type"].isin(SCRIMMAGE_TYPES) & plays["ppa"].notna()].copy()
    df["play_class"] = play_class(df["play_type"], df["play_text"])
    df["success"] = is_success(df)
    df["turnover"] = df["play_type"].isin(TURNOVER_TYPES)
    long_enough = np.where(df["play_class"] == "rush", explosive.rush_yards, explosive.pass_yards)
    df["explosive"] = (df["yards_gained"] >= long_enough) & ~df["turnover"]
    df["margin"] = df["offense_score"] - df["defense_score"]
    df["garbage"] = is_garbage(df["period"], df["margin"], garbage)
    df["score_state"] = score_state(df["margin"])
    df["quarter"] = np.where(df["period"] > 4, "OT", df["period"].astype(str))
    neutral = df["game_id"].map(games.set_index("id")["neutral_site"]).eq(True)
    df["venue"] = np.where(neutral, "neutral", np.where(df["offense"] == df["home"], "home", "away"))
    return df[ENRICHED_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_transform.py`, then the whole suite: `.venv\Scripts\python -m pytest`
Expected: `24 passed`; full suite `83 passed`.

---

### Task 2: Team metrics (`metrics.py`)

**Files:**
- Create: `src/psu/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: the enriched columns from Task 1 (`offense, defense, season, ppa, success, explosive, turnover, play_class, down, garbage, yards_to_goal, drive_id`); raw `team_game_stats` (`game_id, season, team, category, stat`); raw `drives` (`id, season, offense, defense, start_offense_score, end_offense_score`).
- Produces:
  - `efficiency(enriched, side="offense", by=("season",), exclude_garbage=True) -> DataFrame` with columns `team, *by, plays, epa_per_play, rush_epa, pass_epa, success_rate, rush_success_rate, pass_success_rate, explosiveness, explosive_rate, third_down_rate, turnover_rate`
  - `havoc_rate(team_game_stats, enriched) -> DataFrame[season, team, havoc_events, defensive_plays, havoc_rate]`
  - `turnover_margin(team_game_stats) -> DataFrame[season, team, games, giveaways, takeaways, margin]`
  - `red_zone(enriched, drives, side="offense", by=("season",), line=20) -> DataFrame[team, *by, trips, touchdowns, points_per_trip, td_rate]`
  - `efficiency` and `red_zone` raise `ValueError` if `side` isn't `"offense"` or `"defense"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import pandas as pd
import pytest

from psu.metrics import efficiency, havoc_rate, red_zone, turnover_margin

BASE = dict(season=2024, offense="A", defense="B", ppa=0.0, success=False, explosive=False,
            turnover=False, play_class="rush", down=1, garbage=False, yards_to_goal=50, drive_id="d1")


def enriched(*rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


def box(rows):
    return pd.DataFrame([{"game_id": g, "season": 2024, "team": t, "category": c, "stat": s} for g, t, c, s in rows])


def test_efficiency_offense_basic():
    df = enriched(
        {"ppa": 1.0, "success": True, "play_class": "pass", "explosive": True},
        {"ppa": -0.5},
        {"ppa": 0.5, "success": True, "down": 3},
        {"ppa": 9.0, "success": True, "garbage": True},
    )
    row = efficiency(df, "offense").iloc[0]
    assert (row["team"], row["season"], row["plays"]) == ("A", 2024, 3)
    assert row["epa_per_play"] == pytest.approx(1 / 3)
    assert row["rush_epa"] == pytest.approx(0.0)
    assert row["pass_epa"] == pytest.approx(1.0)
    assert row["success_rate"] == pytest.approx(2 / 3)
    assert row["rush_success_rate"] == pytest.approx(0.5)
    assert row["pass_success_rate"] == pytest.approx(1.0)
    assert row["explosiveness"] == pytest.approx(0.75)
    assert row["explosive_rate"] == pytest.approx(1 / 3)
    assert row["third_down_rate"] == pytest.approx(1.0)


def test_efficiency_can_include_garbage_time():
    df = enriched({"ppa": 1.0}, {"ppa": 9.0, "garbage": True})
    assert efficiency(df, "offense", exclude_garbage=False).iloc[0]["plays"] == 2


def test_efficiency_defense_side_and_splits():
    df = enriched(
        {"offense": "A", "defense": "B", "down": 1, "ppa": 0.2},
        {"offense": "C", "defense": "B", "down": 2, "ppa": 0.4},
        {"offense": "B", "defense": "A", "ppa": -1.0},
    )
    out = efficiency(df, "defense", by=("season", "down"))
    b = out[out["team"] == "B"].sort_values("down")
    assert list(b["down"]) == [1, 2]
    assert list(b["epa_per_play"]) == pytest.approx([0.2, 0.4])


def test_missing_play_class_gives_nan_not_error():
    row = efficiency(enriched({"play_class": "pass", "ppa": 0.3}), "offense").iloc[0]
    assert pd.isna(row["rush_epa"]) and pd.isna(row["third_down_rate"])
    assert row["pass_epa"] == pytest.approx(0.3)


def test_efficiency_rejects_unknown_side():
    with pytest.raises(ValueError):
        efficiency(enriched({}), "special_teams")


def test_havoc_rate_counts_box_score_events_and_forced_fumbles():
    stats = box([
        (1, "A", "tacklesForLoss", "5"), (1, "A", "passesDeflected", "3"),
        (1, "A", "passesIntercepted", "1"), (1, "A", "totalFumbles", "0"),
        (1, "B", "tacklesForLoss", "2"), (1, "B", "totalFumbles", "2"),  # B has no passesDeflected row
    ])
    plays = enriched(*[{"offense": "B", "defense": "A"}] * 40, *[{"offense": "A", "defense": "B"}] * 50)
    out = havoc_rate(stats, plays).set_index("team")
    assert out.loc["A", "havoc_events"] == 11  # 5 TFL + 3 PD + 1 INT + 2 opponent fumbles
    assert out.loc["A", "havoc_rate"] == pytest.approx(11 / 40)
    assert out.loc["B", "havoc_events"] == 2
    assert out.loc["B", "havoc_rate"] == pytest.approx(2 / 50)


def test_turnover_margin():
    stats = box([
        (1, "A", "turnovers", "1"), (1, "B", "turnovers", "3"),
        (2, "A", "turnovers", "2"), (2, "C", "totalYards", "300"),  # C has no turnovers row: counts as 0
    ])
    out = turnover_margin(stats).set_index("team")
    assert out.loc["A", ["games", "giveaways", "takeaways", "margin"]].tolist() == [2, 3, 3, 0]
    assert out.loc["B", "margin"] == -2
    assert out.loc["C", "margin"] == 2


def test_red_zone_trips_touchdowns_and_points():
    plays = enriched(
        {"drive_id": "d1", "yards_to_goal": 15}, {"drive_id": "d1", "yards_to_goal": 3},
        {"drive_id": "d2", "yards_to_goal": 18}, {"drive_id": "d3", "yards_to_goal": 40},
    )
    drives = pd.DataFrame([
        {"id": "d1", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 0, "end_offense_score": 7},
        {"id": "d2", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 7, "end_offense_score": 10},
        {"id": "d3", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 10, "end_offense_score": 10},
    ])
    row = red_zone(plays, drives).iloc[0]
    assert (row["team"], row["trips"], row["touchdowns"]) == ("A", 2, 1)
    assert row["td_rate"] == pytest.approx(0.5)
    assert row["points_per_trip"] == pytest.approx(5.0)
    assert red_zone(plays, drives, "defense").iloc[0]["team"] == "B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.metrics'`

- [ ] **Step 3: Implement `src/psu/metrics.py`**

```python
"""Team efficiency metrics from enriched plays and box scores. Every function returns a tidy DataFrame."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

SIDES = ("offense", "defense")
HAVOC_CATEGORIES = ("tacklesForLoss", "passesDeflected", "passesIntercepted")


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be 'offense' or 'defense', got {side!r}")


def _masked_mean(df: pd.DataFrame, keys: list[str], value: str, mask: pd.Series) -> pd.Series:
    """Group mean of `value` over rows where mask is True; groups with no such rows get NaN."""
    return df[value].astype(float).where(mask).groupby([df[k] for k in keys]).mean()


def efficiency(
    enriched: pd.DataFrame,
    side: str = "offense",
    by: Sequence[str] = ("season",),
    exclude_garbage: bool = True,
) -> pd.DataFrame:
    _check_side(side)
    df = enriched[~enriched["garbage"]] if exclude_garbage else enriched
    keys = [side, *by]
    grouped = df.groupby(keys)
    rush = df["play_class"] == "rush"
    out = pd.DataFrame({
        "plays": grouped.size(),
        "epa_per_play": grouped["ppa"].mean(),
        "rush_epa": _masked_mean(df, keys, "ppa", rush),
        "pass_epa": _masked_mean(df, keys, "ppa", ~rush),
        "success_rate": grouped["success"].mean(),
        "rush_success_rate": _masked_mean(df, keys, "success", rush),
        "pass_success_rate": _masked_mean(df, keys, "success", ~rush),
        "explosiveness": _masked_mean(df, keys, "ppa", df["success"]),
        "explosive_rate": grouped["explosive"].mean(),
        "third_down_rate": _masked_mean(df, keys, "success", df["down"] == 3),
        "turnover_rate": grouped["turnover"].mean(),
    })
    return out.reset_index().rename(columns={side: "team"})


def _box_wide(team_game_stats: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """One row per (game, team) with a numeric column per category; missing categories count as 0."""
    teams = team_game_stats[["game_id", "season", "team"]].drop_duplicates()
    rows = team_game_stats[team_game_stats["category"].isin(categories)]
    values = rows.assign(value=pd.to_numeric(rows["stat"], errors="coerce")).pivot_table(
        index=["game_id", "season", "team"], columns="category", values="value", aggfunc="sum"
    )
    values = values.reindex(columns=list(categories))
    values.columns.name = None
    wide = teams.merge(values.reset_index(), on=["game_id", "season", "team"], how="left")
    wide[list(categories)] = wide[list(categories)].fillna(0.0)
    return wide


def _with_opponent(wide: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    opponent = wide[["game_id", "team", *columns]].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in columns}}
    )
    merged = wide.merge(opponent, on="game_id")
    return merged[merged["team"] != merged["opponent"]]


def havoc_rate(team_game_stats: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """(TFL + passes defended + interceptions + opponent fumbles) / defensive scrimmage plays, per season.

    Uses all scrimmage plays (garbage time included), matching CFBD's havoc rate.
    """
    wide = _with_opponent(_box_wide(team_game_stats, (*HAVOC_CATEGORIES, "totalFumbles")), ["totalFumbles"])
    wide["havoc_events"] = wide[list(HAVOC_CATEGORIES)].sum(axis=1) + wide["opp_totalFumbles"]
    events = wide.groupby(["season", "team"])["havoc_events"].sum()
    plays = enriched.groupby(["season", "defense"]).size().rename_axis(["season", "team"])
    out = pd.concat({"havoc_events": events, "defensive_plays": plays}, axis=1).dropna()
    out["havoc_rate"] = out["havoc_events"] / out["defensive_plays"]
    return out.reset_index()


def turnover_margin(team_game_stats: pd.DataFrame) -> pd.DataFrame:
    wide = _with_opponent(_box_wide(team_game_stats, ("turnovers",)), ["turnovers"])
    grouped = wide.groupby(["season", "team"])
    out = pd.DataFrame({
        "games": grouped.size(),
        "giveaways": grouped["turnovers"].sum(),
        "takeaways": grouped["opp_turnovers"].sum(),
    })
    out["margin"] = out["takeaways"] - out["giveaways"]
    return out.reset_index()


def red_zone(
    enriched: pd.DataFrame,
    drives: pd.DataFrame,
    side: str = "offense",
    by: Sequence[str] = ("season",),
    line: int = 20,
) -> pd.DataFrame:
    """Drives with a scrimmage play at or inside the `line`: trips, TD rate, and points per trip."""
    _check_side(side)
    reached = enriched.loc[enriched["yards_to_goal"] <= line, "drive_id"].unique()
    trips = drives[drives["id"].isin(reached)].copy()
    trips["points"] = (trips["end_offense_score"] - trips["start_offense_score"]).clip(lower=0)
    trips["touchdown"] = trips["points"] >= 6
    grouped = trips.groupby([side, *by])
    out = pd.DataFrame({
        "trips": grouped.size(),
        "touchdowns": grouped["touchdown"].sum(),
        "points_per_trip": grouped["points"].mean(),
    })
    out["td_rate"] = out["touchdowns"] / out["trips"]
    return out.reset_index().rename(columns={side: "team"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_metrics.py`, then `.venv\Scripts\python -m pytest`
Expected: `9 passed`; full suite `92 passed`.

---

### Task 3: Opponent adjustment (`adjust.py`)

**Files:**
- Create: `src/psu/adjust.py`
- Test: `tests/test_adjust.py`

**Interfaces:**
- Consumes: enriched columns `season, offense, defense, ppa, success, venue, garbage`
- Produces: `opponent_adjust(enriched, value="ppa", *, alpha=50.0, exclude_garbage=True) -> DataFrame` with columns exactly `season, team, off_raw, off_adj, def_raw, def_adj, off_plays, def_plays`. `value` is `"ppa"` or `"success"`; anything else raises `ValueError`. The output covers every team seen in the season, FBS or not.

**Model:** for each season, a ridge regression `value ~ intercept + offense_team + defense_team + home` on sparse one-hot columns. `home` is +1 when the offense is at home, −1 when away, and 0 at a neutral site.
- `off_adj = intercept + offense coefficient`. Higher is better.
- `def_adj = intercept + defense coefficient`. Lower is better.
- `alpha` shrinks teams with few plays (FCS opponents) toward the mean.

- [ ] **Step 1: Write the failing tests**

`tests/test_adjust.py`:
```python
import numpy as np
import pandas as pd
import pytest

from psu.adjust import opponent_adjust

OFFENSE = {"A": 0.3, "B": 0.0, "C": 0.0, "D": 0.0}
DEFENSE = {"A": 0.0, "B": 0.0, "C": 0.0, "D": -0.3}  # D is the elite defense
PAIRS = [("A", "B"), ("A", "D"), ("B", "D"), ("B", "C"), ("C", "A"), ("C", "B"), ("D", "A"), ("D", "C")]


def synthetic(seed=0, n=300, venue="neutral", home_boost=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for off, de in PAIRS:
        y = 0.1 + OFFENSE[off] + DEFENSE[de] + rng.normal(0, 0.05, n)
        v = venue if venue != "alternate" else None
        for i, value in enumerate(y):
            where = v or ("home" if i % 2 == 0 else "away")
            bump = home_boost if where == "home" else -home_boost if where == "away" else 0.0
            rows.append({"season": 2024, "offense": off, "defense": de, "ppa": value + bump,
                         "success": value > 0.1, "venue": where, "garbage": False})
    return pd.DataFrame(rows)


def test_adjustment_removes_schedule_strength():
    out = opponent_adjust(synthetic(), "ppa", alpha=1.0).set_index("team")
    # B's offense faced elite defense D; C's didn't. Raw says C > B; adjusted says they're equal.
    assert out.loc["C", "off_raw"] - out.loc["B", "off_raw"] > 0.1
    assert abs(out.loc["C", "off_adj"] - out.loc["B", "off_adj"]) < 0.03
    assert out.loc["A", "off_adj"] - out.loc["B", "off_adj"] == pytest.approx(0.3, abs=0.03)
    assert out["def_adj"].idxmin() == "D"


def test_success_adjustment_and_output_shape():
    out = opponent_adjust(synthetic(), "success", alpha=1.0)
    assert list(out.columns) == ["season", "team", "off_raw", "off_adj", "def_raw", "def_adj", "off_plays", "def_plays"]
    assert set(out["team"]) == {"A", "B", "C", "D"}
    assert (out["off_plays"] == 600).all() and (out["def_plays"] == 600).all()


def test_home_field_term_does_not_bias_neutral_games():
    boosted = opponent_adjust(synthetic(venue="alternate", home_boost=0.2), "ppa", alpha=1.0).set_index("team")
    neutral = opponent_adjust(synthetic(), "ppa", alpha=1.0).set_index("team")
    diff = boosted["off_adj"] - boosted.loc["B", "off_adj"]
    base = neutral["off_adj"] - neutral.loc["B", "off_adj"]
    assert (diff - base).abs().max() < 0.03  # the home term absorbs home advantage


def test_garbage_time_excluded_by_default_and_value_validated():
    df = pd.concat([synthetic(), pd.DataFrame([{"season": 2024, "offense": "B", "defense": "C", "ppa": 50.0,
                                               "success": True, "venue": "neutral", "garbage": True}])])
    out = opponent_adjust(df, "ppa", alpha=1.0).set_index("team")
    assert out.loc["B", "off_plays"] == 600
    with pytest.raises(ValueError):
        opponent_adjust(df, "yards")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_adjust.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.adjust'`

- [ ] **Step 3: Implement `src/psu/adjust.py`**

```python
"""Opponent adjustment: per-season ridge regression of play outcomes on offense and defense team effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

VALUES = ("ppa", "success")
COLUMNS = ["season", "team", "off_raw", "off_adj", "def_raw", "def_adj", "off_plays", "def_plays"]


def _one_hot(codes: np.ndarray, width: int) -> sparse.csr_matrix:
    n = len(codes)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, width))


def opponent_adjust(
    enriched: pd.DataFrame,
    value: str = "ppa",
    *,
    alpha: float = 50.0,
    exclude_garbage: bool = True,
) -> pd.DataFrame:
    """Fit value ~ intercept + offense + defense + home per season.

    off_adj = intercept + offense effect (higher is better); def_adj = intercept + defense effect
    (lower is better). Teams with few plays are shrunk toward average by `alpha`.
    """
    if value not in VALUES:
        raise ValueError(f"value must be one of {VALUES}, got {value!r}")
    df = enriched[~enriched["garbage"]] if exclude_garbage else enriched
    frames = []
    for season, s in df.groupby("season"):
        teams = pd.Index(sorted(set(s["offense"]) | set(s["defense"])))
        width = len(teams)
        home = np.select([s["venue"].eq("home"), s["venue"].eq("away")], [1.0, -1.0], 0.0)
        x = sparse.hstack([
            _one_hot(teams.get_indexer(s["offense"]), width),
            _one_hot(teams.get_indexer(s["defense"]), width),
            sparse.csr_matrix(home.reshape(-1, 1)),
        ]).tocsr()
        y = s[value].astype(float).to_numpy()
        model = Ridge(alpha=alpha).fit(x, y)
        values = s[value].astype(float)
        frames.append(pd.DataFrame({
            "season": season,
            "team": teams,
            "off_raw": values.groupby(s["offense"]).mean().reindex(teams).to_numpy(),
            "off_adj": model.intercept_ + model.coef_[:width],
            "def_raw": values.groupby(s["defense"]).mean().reindex(teams).to_numpy(),
            "def_adj": model.intercept_ + model.coef_[width:2 * width],
            "off_plays": s.groupby("offense").size().reindex(teams, fill_value=0).to_numpy(),
            "def_plays": s.groupby("defense").size().reindex(teams, fill_value=0).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_adjust.py`, then `.venv\Scripts\python -m pytest`
Expected: `4 passed`; full suite `96 passed`.

---

### Task 4: Build the derived tables (`build.py`)

**Files:**
- Create: `src/psu/build.py`
- Modify: `tests/conftest.py` (append `seed_raw_tables`)
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: `enrich_plays`, `GarbageTime`, `Explosive` (Task 1); `efficiency`, `havoc_rate`, `turnover_margin`, `red_zone` (Task 2); `opponent_adjust` (Task 3); `psu.db.SPECS`, `psu.db.upsert`, and `psu.db.connect` (Phase 1).
- Produces:
  - `fbs_teams(games: DataFrame) -> DataFrame[season, team]`
  - `SPLITS = ("down", "quarter", "score_state", "venue", "opp_conference")`
  - `DERIVED_TABLES = ("plays_enriched", "team_offense", "team_defense", "team_splits", "team_adjusted", "team_havoc", "team_turnovers", "team_red_zone")`
  - `build(con, *, garbage=GarbageTime(), explosive=Explosive(), alpha=50.0) -> dict[str, int]`: writes each `DERIVED_TABLES` entry with `CREATE OR REPLACE TABLE` and returns the row counts. Every table except `plays_enriched` holds FBS teams only.
  - Table shapes:
    - `team_splits`: `side, split, season, team, split_value (VARCHAR), plays, epa_per_play, ...` (the `efficiency` columns)
    - `team_adjusted`: `season, team, off_epa_raw, off_epa_adj, def_epa_raw, def_epa_adj, off_sr_raw, off_sr_adj, def_sr_raw, def_sr_adj, off_plays, def_plays`
    - `team_red_zone`: `side` plus the `red_zone` columns
  - `tests/conftest.py`: `seed_raw_tables(con) -> None` loads a tiny season into the raw tables (FBS teams "Alpha" and "Beta", and FCS team "Gamma").

- [ ] **Step 1: Append the seed helper to `tests/conftest.py`**

Add at the bottom of `tests/conftest.py`, after the existing fixtures:
```python
def seed_raw_tables(con):
    """Load a tiny 2024 season into the raw Phase 1 tables: FBS Alpha and Beta, FCS Gamma."""
    import pandas as pd

    from psu.db import SPECS, upsert

    games = pd.DataFrame([
        {"id": 1, "season": 2024, "week": 1, "season_type": "regular", "neutral_site": False,
         "home_team": "Alpha", "home_classification": "fbs", "home_points": 28,
         "away_team": "Beta", "away_classification": "fbs", "away_points": 14},
        {"id": 2, "season": 2024, "week": 2, "season_type": "regular", "neutral_site": False,
         "home_team": "Alpha", "home_classification": "fbs", "home_points": 42,
         "away_team": "Gamma", "away_classification": "fcs", "away_points": 3},
    ])
    plays, drives, box = [], [], []
    n = 0
    for game_id, home, away in ((1, "Alpha", "Beta"), (2, "Alpha", "Gamma")):
        for offense, defense in ((home, away), (away, home)):
            drive_id = f"{game_id}-{offense}"
            drives.append({"id": drive_id, "game_id": game_id, "season": 2024, "offense": offense,
                           "defense": defense, "start_offense_score": 0,
                           "end_offense_score": 7 if offense == "Alpha" else 0})
            for i in range(6):
                n += 1
                plays.append({
                    "id": str(n), "game_id": game_id, "drive_id": drive_id, "season": 2024, "week": game_id,
                    "season_type": "regular", "offense": offense, "defense": defense,
                    "offense_conference": None if offense == "Gamma" else "Big Ten",
                    "defense_conference": None if defense == "Gamma" else "Big Ten",
                    "home": home, "away": away, "period": 1 + i % 4, "down": 1 + i % 3, "distance": 10,
                    "yards_to_goal": 60 - i * 10, "yards_gained": 8 if offense == "Alpha" else 2,
                    "play_type": "Rush" if i % 2 else "Pass Reception", "play_text": "",
                    "ppa": 0.4 if offense == "Alpha" else -0.2, "offense_score": 0, "defense_score": 0,
                })
            plays.append({**plays[-1], "id": f"punt-{n}", "play_type": "Punt", "ppa": None})
            for category, stat in (("turnovers", "1" if offense == "Beta" else "0"), ("tacklesForLoss", "2")):
                box.append({"game_id": game_id, "season": 2024, "week": game_id, "season_type": "regular",
                            "team": offense, "category": category, "stat": stat})
    upsert(con, SPECS["games"], games)
    upsert(con, SPECS["plays"], pd.DataFrame(plays))
    upsert(con, SPECS["drives"], pd.DataFrame(drives))
    upsert(con, SPECS["team_game_stats"], pd.DataFrame(box))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_build.py`:
```python
import pandas as pd
import pytest

from conftest import seed_raw_tables
from psu.build import DERIVED_TABLES, build, fbs_teams
from psu.db import connect
from psu.transform import GarbageTime


def test_fbs_teams_uses_game_classifications():
    games = pd.DataFrame([
        {"season": 2024, "home_team": "Alpha", "home_classification": "fbs", "away_team": "Gamma", "away_classification": "fcs"},
        {"season": 2024, "home_team": "Beta", "home_classification": "fbs", "away_team": "Alpha", "away_classification": "fbs"},
    ])
    assert set(map(tuple, fbs_teams(games)[["season", "team"]].to_numpy())) == {(2024, "Alpha"), (2024, "Beta")}


def test_build_writes_fbs_only_tables_and_is_repeatable():
    con = connect(":memory:")
    seed_raw_tables(con)
    first = build(con, alpha=1.0)
    assert set(first) == set(DERIVED_TABLES)
    assert first["plays_enriched"] == 24  # punts and null-ppa plays filtered out
    for table in DERIVED_TABLES[1:]:
        teams = {r[0] for r in con.execute(f"SELECT DISTINCT team FROM {table}").fetchall()}
        assert teams <= {"Alpha", "Beta"} and teams, table
    assert build(con, alpha=1.0) == first  # CREATE OR REPLACE, not append
    alpha = con.execute("SELECT epa_per_play, success_rate FROM team_offense WHERE team = 'Alpha'").fetchone()
    assert alpha[0] == pytest.approx(0.4) and alpha[1] > 0.5
    splits = {r[0] for r in con.execute("SELECT DISTINCT split FROM team_splits").fetchall()}
    assert splits == {"down", "quarter", "score_state", "venue", "opp_conference"}
    adjusted = con.execute("SELECT off_epa_adj, def_epa_adj, off_sr_adj FROM team_adjusted WHERE team = 'Alpha'").fetchone()
    assert all(v is not None for v in adjusted)
    assert con.execute("SELECT margin FROM team_turnovers WHERE team = 'Alpha'").fetchone()[0] == 1


def test_build_honours_garbage_setting():
    con = connect(":memory:")
    seed_raw_tables(con)
    assert build(con, garbage=GarbageTime.parse("off"), alpha=1.0)["team_offense"] == 2
```

`from conftest import seed_raw_tables` works because pytest puts `tests/` on `sys.path` (rootdir-relative import mode, no `tests/__init__.py`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_build.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.build'`

- [ ] **Step 4: Implement `src/psu/build.py`**

```python
"""Materialize the Phase 2 metric tables in DuckDB from the raw Phase 1 tables (no API calls)."""
from __future__ import annotations

import duckdb
import pandas as pd

from psu.adjust import opponent_adjust
from psu.metrics import efficiency, havoc_rate, red_zone, turnover_margin
from psu.transform import Explosive, GarbageTime, enrich_plays

SPLITS = ("down", "quarter", "score_state", "venue", "opp_conference")
DERIVED_TABLES = (
    "plays_enriched", "team_offense", "team_defense", "team_splits",
    "team_adjusted", "team_havoc", "team_turnovers", "team_red_zone",
)
_PLAY_COLUMNS = (
    "id, game_id, drive_id, season, week, season_type, offense, offense_conference, defense, "
    "defense_conference, home, away, period, down, distance, yards_to_goal, yards_gained, play_type, "
    "play_text, ppa, offense_score, defense_score"
)


def fbs_teams(games: pd.DataFrame) -> pd.DataFrame:
    sides = [
        games.loc[games[f"{side}_classification"] == "fbs", ["season", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        for side in ("home", "away")
    ]
    return pd.concat(sides).drop_duplicates().reset_index(drop=True)


def _splits(enriched: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side, opp_conf in (("offense", "defense_conference"), ("defense", "offense_conference")):
        with_opp = enriched.assign(opp_conference=enriched[opp_conf].fillna("FCS/other"))
        for split in SPLITS:
            frame = efficiency(with_opp, side, by=("season", split)).rename(columns={split: "split_value"})
            frame["split_value"] = frame["split_value"].astype(str)
            frame.insert(0, "split", split)
            frame.insert(0, "side", side)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _adjusted(enriched: pd.DataFrame, alpha: float) -> pd.DataFrame:
    def renamed(value: str, tag: str) -> pd.DataFrame:
        return opponent_adjust(enriched, value, alpha=alpha).rename(columns={
            "off_raw": f"off_{tag}_raw", "off_adj": f"off_{tag}_adj",
            "def_raw": f"def_{tag}_raw", "def_adj": f"def_{tag}_adj",
        })

    epa = renamed("ppa", "epa")
    sr = renamed("success", "sr").drop(columns=["off_plays", "def_plays"])
    merged = epa.merge(sr, on=["season", "team"])
    return merged[[
        "season", "team", "off_epa_raw", "off_epa_adj", "def_epa_raw", "def_epa_adj",
        "off_sr_raw", "off_sr_adj", "def_sr_raw", "def_sr_adj", "off_plays", "def_plays",
    ]]


def _red_zone(enriched: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side in ("offense", "defense"):
        frame = red_zone(enriched, drives, side)
        frame.insert(0, "side", side)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register("_derived", df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _derived')
    finally:
        con.unregister("_derived")


def build(
    con: duckdb.DuckDBPyConnection,
    *,
    garbage: GarbageTime = GarbageTime(),
    explosive: Explosive = Explosive(),
    alpha: float = 50.0,
) -> dict[str, int]:
    plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays").df()
    games = con.execute(
        "SELECT id, season, neutral_site, home_team, away_team, home_classification, away_classification FROM games"
    ).df()
    drives = con.execute(
        "SELECT id, season, offense, defense, start_offense_score, end_offense_score FROM drives"
    ).df()
    box = con.execute("SELECT game_id, season, team, category, stat FROM team_game_stats").df()

    enriched = enrich_plays(plays, games, garbage=garbage, explosive=explosive)
    fbs = fbs_teams(games)

    def fbs_only(df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(fbs, on=["season", "team"])

    tables = {
        "plays_enriched": enriched,
        "team_offense": fbs_only(efficiency(enriched, "offense")),
        "team_defense": fbs_only(efficiency(enriched, "defense")),
        "team_splits": fbs_only(_splits(enriched)),
        "team_adjusted": fbs_only(_adjusted(enriched, alpha)),
        "team_havoc": fbs_only(havoc_rate(box, enriched)),
        "team_turnovers": fbs_only(turnover_margin(box)),
        "team_red_zone": fbs_only(_red_zone(enriched, drives)),
    }
    for name, df in tables.items():
        _write(con, name, df)
    return {name: len(df) for name, df in tables.items()}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_build.py`, then `.venv\Scripts\python -m pytest`
Expected: `3 passed`; full suite `99 passed`.

---

### Task 5: `psu build` command

**Files:**
- Modify: `src/psu/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `build`, `DERIVED_TABLES` (Task 4); `GarbageTime.parse` (Task 1); `db.connect`, `db.row_counts`, `_print_counts` (Phase 1)
- Produces:
  - `psu build [--garbage 38,28,22|off] [--alpha 50]`. It prints one row count per derived table.
  - Exit codes: 0 = ok; 2 = a bad `--garbage` value, or no plays loaded yet (the message tells you to run `psu ingest`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:
```python
def test_build_requires_ingested_data(settings, capsys):
    assert cli.main(["build"]) == 2
    assert "psu ingest" in capsys.readouterr().err


def test_build_rejects_bad_garbage_spec(settings, capsys):
    assert cli.main(["build", "--garbage", "30,20"]) == 2
    assert "38,28,22" in capsys.readouterr().err


def test_build_writes_tables(settings, capsys):
    from conftest import seed_raw_tables
    from psu import db

    con = db.connect(settings.db_path)
    seed_raw_tables(con)
    con.close()
    assert cli.main(["build", "--alpha", "1"]) == 0
    out = capsys.readouterr().out
    assert "team_offense" in out and "team_adjusted" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`
Expected: the 3 new tests FAIL (`argparse` exits with "invalid choice: 'build'", i.e. `SystemExit: 2` raised rather than returned).

- [ ] **Step 3: Implement the subcommand in `src/psu/cli.py`**

After the `sub.add_parser("status", ...)` line, add:
```python
    bld = sub.add_parser("build", help="Compute metric tables from ingested data (no API calls)")
    bld.add_argument("--garbage", default="38,28,22", help='Q2,Q3,Q4 garbage-time margins, or "off"')
    bld.add_argument("--alpha", type=float, default=50.0, help="Ridge shrinkage for opponent adjustment")
```
After the `if args.command == "status": ...` block, add:
```python
    if args.command == "build":
        try:
            garbage = GarbageTime.parse(args.garbage)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        con = db.connect(settings.db_path)
        try:
            if db.row_counts(con)["plays"] == 0:
                print("error: no plays loaded yet; run `psu ingest` first", file=sys.stderr)
                return 2
            _print_counts(build(con, garbage=garbage, alpha=args.alpha))
        finally:
            con.close()
        return 0
```
Add these imports at the top with the others:
```python
from psu.build import build
from psu.transform import GarbageTime
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`, then `.venv\Scripts\python -m pytest`
Expected: `9 passed` in test_cli.py; full suite `102 passed`.

---

### Task 6: Validate against CFBD advanced stats on the real database

**Files:**
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: `enrich_plays`, `efficiency`, `havoc_rate`; `_PLAY_COLUMNS` from `psu.build`; the real `data/psu.duckdb` (read-only)
- Produces: a validation test that skips itself when the database is absent.

- [ ] **Step 1: Write the test**

`tests/test_validation.py`:
```python
"""Checks our metrics against CFBD's own advanced season stats for Penn State.

Needs the real data/psu.duckdb from `psu ingest`; skipped when it is absent. CFBD's advanced stats
include garbage time, so the comparison uses exclude_garbage=False.
"""
from pathlib import Path

import duckdb
import pytest

from psu.build import _PLAY_COLUMNS
from psu.metrics import efficiency, havoc_rate
from psu.transform import enrich_plays

DB = Path(__file__).resolve().parents[1] / "data" / "psu.duckdb"
SEASONS = (2022, 2023, 2024, 2025)
TEAM = "Penn State"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="needs data/psu.duckdb from `psu ingest`")


@pytest.fixture(scope="module")
def ours_and_cfbd():
    con = duckdb.connect(str(DB), read_only=True)
    try:
        season_list = ", ".join(map(str, SEASONS))
        plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays WHERE season IN ({season_list})").df()
        games = con.execute("SELECT id, neutral_site FROM games").df()
        box = con.execute(
            f"SELECT game_id, season, team, category, stat FROM team_game_stats WHERE season IN ({season_list})"
        ).df()
        cfbd = con.execute(f"SELECT * FROM advanced_season WHERE team = '{TEAM}'").df().set_index("season")
    finally:
        con.close()
    enriched = enrich_plays(plays, games)
    offense = efficiency(enriched, "offense", exclude_garbage=False).set_index(["team", "season"]).loc[TEAM]
    defense = efficiency(enriched, "defense", exclude_garbage=False).set_index(["team", "season"]).loc[TEAM]
    havoc = havoc_rate(box, enriched).set_index(["team", "season"]).loc[TEAM]
    return offense, defense, havoc, cfbd


@pytest.mark.parametrize("season", SEASONS)
@pytest.mark.parametrize("side", ["offense", "defense"])
def test_efficiency_matches_cfbd(ours_and_cfbd, season, side):
    offense, defense, _, cfbd = ours_and_cfbd
    ours = (offense if side == "offense" else defense).loc[season]
    theirs = cfbd.loc[season]
    assert ours["plays"] == pytest.approx(theirs[f"{side}_plays"], rel=0.01)
    assert ours["epa_per_play"] == pytest.approx(theirs[f"{side}_ppa"], abs=0.01)
    assert ours["success_rate"] == pytest.approx(theirs[f"{side}_success_rate"], abs=0.015)
    assert ours["explosiveness"] == pytest.approx(theirs[f"{side}_explosiveness"], abs=0.03)


@pytest.mark.parametrize("season", SEASONS)
def test_havoc_matches_cfbd(ours_and_cfbd, season):
    _, _, havoc, cfbd = ours_and_cfbd
    assert havoc.loc[season, "havoc_rate"] == pytest.approx(cfbd.loc[season, "defense_havoc_total"], abs=0.02)
```

- [ ] **Step 2: Run it**

Run: `.venv\Scripts\python -m pytest tests/test_validation.py -v`
Expected: `12 passed` (2024 was verified by hand while planning; the other seasons are expected to match as well). **If any season is outside tolerance, don't loosen the tolerances or change the metric code.** Report each failing season with our value and CFBD's value, and the orchestrator will decide.

---

### Task 7: Build the real tables, README, and PR (orchestrator)

**Run by the orchestrator, not a subagent. It reads the real database and makes no API calls.**

**Files:**
- Modify: `README.md` (append a Phase 2 section)

- [ ] **Step 1: Build the real tables**

Run: `.venv\Scripts\psu build`
Expected: counts for all 8 tables. Rough sizes: `plays_enriched` ~500k; `team_offense` / `team_defense` ~680 (about 136 FBS teams × 5 seasons); `team_splits` in the tens of thousands.

- [ ] **Step 2: Sanity-check Penn State**

Run:
```powershell
@'
import duckdb
con = duckdb.connect("data/psu.duckdb", read_only=True)
print(con.execute("""
    SELECT o.season, round(o.epa_per_play, 3) AS off_epa, round(o.success_rate, 3) AS off_sr,
           round(d.epa_per_play, 3) AS def_epa, round(a.off_epa_adj, 3) AS off_adj,
           round(a.def_epa_adj, 3) AS def_adj, round(h.havoc_rate, 3) AS havoc, t.margin AS to_margin
    FROM team_offense o
    JOIN team_defense d USING (season, team)
    JOIN team_adjusted a USING (season, team)
    JOIN team_havoc h USING (season, team)
    JOIN team_turnovers t USING (season, team)
    WHERE team = 'Penn State' ORDER BY season
""").df().to_string(index=False))
print(con.execute("""
    SELECT season, team, round(off_epa_adj - def_epa_adj, 3) AS net_adj
    FROM team_adjusted WHERE season = 2024 ORDER BY net_adj DESC LIMIT 10
""").df().to_string(index=False))
'@ | .venv\Scripts\python -
```
Expected: Penn State shows positive offensive EPA and negative defensive EPA every season. The 2024 top 10 by net adjusted EPA should look like a plausible top 10, including the CFP teams (Penn State, Ohio State, Notre Dame, Texas, Oregon...). Report anything implausible.

- [ ] **Step 3: Append the Phase 2 section to `README.md`**

````markdown
## Phase 2: Metrics layer

```powershell
.venv\Scripts\psu build                      # no API calls; rebuilds every metric table
.venv\Scripts\psu build --garbage 30,20,15   # custom garbage-time margins (Q2,Q3,Q4), or --garbage off
.venv\Scripts\python -m pytest tests/test_validation.py   # Penn State vs CFBD's advanced stats
```

`psu build` reads the Phase 1 tables and writes these DuckDB tables (FBS teams only):

| Table | What's in it |
|---|---|
| `plays_enriched` | scrimmage plays with `play_class`, `success`, `explosive`, `turnover`, `garbage`, `score_state`, `quarter`, `venue` |
| `team_offense` / `team_defense` | EPA/play (overall, rush, pass), success rate, explosiveness, explosive rate, 3rd-down rate, turnover rate, per season; garbage time excluded |
| `team_splits` | the same metrics split by down, quarter, score state, venue, and opponent conference |
| `team_adjusted` | opponent-adjusted EPA/play and success rate: a per-season ridge regression with offense, defense, and home-field terms (`--alpha` sets shrinkage) |
| `team_havoc` | (TFL + passes defended + INT + forced fumbles) / defensive plays, from box scores |
| `team_turnovers` | giveaways, takeaways, and margin |
| `team_red_zone` | trips inside the 20, TD rate, and points per trip, for offense and defense |

Definitions:
- **Success:** 50% of the yards needed on 1st down, 70% on 2nd, and 100% on 3rd/4th. TDs always succeed; turnovers never do.
- **Explosiveness:** the mean EPA of successful plays.
- **Explosive play:** a rush of 12+ yards or a pass of 16+ yards.
- **Garbage time:** a margin above 38 in Q2, above 28 in Q3, or above 22 in Q4.
````

- [ ] **Step 4: Full suite, commit, push, and open the PR**

```powershell
.venv\Scripts\python -m pytest
git add README.md
git commit -m "docs: add Phase 2 metrics guide to README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/phase2-metrics
& "C:\Program Files\GitHub CLI\gh.exe" pr create --base main --head feat/phase2-metrics --title "Phase 2: metrics layer" --body-file <body file>
```
The PR body should cover a summary, the validation results, the Step 2 Penn State table, the top-10 sanity check, and a test plan. Then stop for the user's review.
