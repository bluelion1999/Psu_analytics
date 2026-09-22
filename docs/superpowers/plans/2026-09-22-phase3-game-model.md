# Phase 3 — Game Prediction Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **In this repo:** implementers and reviewers follow `.claude/agents/psu-implementer.md` and `.claude/agents/psu-reviewer.md`. Give each agent only its own task, plus **Global Constraints** and **Review Focus**.

**Goal:** `psu train` does three things for every FBS-vs-FBS game:
- builds pregame features (no leakage)
- backtests a margin model against the closing Vegas line using time-based splits
- writes predicted margins and win probabilities, for played and upcoming games, to `game_predictions`

It also saves the model and a metrics report.

**Architecture:**
- `features.py` computes each team's ratings *as of* each slate (week), fitting the Phase 2 ridge adjustment only on plays from earlier slates of the same season. Early in a season those ratings are blended with the previous season's full-season ratings.
- `models/game_predict.py` is pure modelling: two model kinds (linear ridge, and XGBoost), a normal-CDF win probability, scoring, and a backtest that validates on season N−1 and tests on season N.
- `train.py` does the DuckDB and file I/O. `psu train` wraps it.

**Tech Stack:** pandas, numpy, scipy (`norm`), scikit-learn (Pipeline, SimpleImputer, StandardScaler, Ridge, cross_val_predict), xgboost, joblib, DuckDB.

**Spec:** `docs/superpowers/specs/psu-analytics-build-plan.md` ("Phase 3 — Game prediction model")

## Global Constraints

- Branch `feat/phase3-game-model`. One Conventional Commit per task (message given in the dispatch), staging only that task's files by explicit path, and ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No CFBD API calls.** Everything reads `data/psu.duckdb`. Never run `psu ingest`. Unit tests use synthetic frames or in-memory DuckDB.
- **Target:** `margin = home_points − away_points`, for completed FBS-vs-FBS games (both teams `fbs` in `games`).
- **Win probability:** `P(home wins) = Φ(predicted_margin / σ)`. σ is the standard deviation of out-of-fold residuals on the training set.
- **No leakage** (verified by tests):
  - Efficiency ratings for a game use only plays from earlier slates of the same season, plus the previous season's full-season ratings.
  - SP+ is the **previous** season's rating, because CFBD's same-season SP+ reflects the whole season.
  - Talent is the same season's (a preseason recruiting composite).
  - Rest days come from earlier game dates.
- **Slates:** regular-season week w is slate w. Postseason week w is slate `last_regular_week + w`.
- **Blending:** `rating = w·current + (1−w)·prior`, with `w = plays/(plays + shrink_plays)` (default 300). A team with no prior-season rating uses the previous season's league mean. The first season in the data has no prior at all, so it's excluded from training and evaluation.
- **Features** (home minus away unless noted): `home_field` (0 at a neutral site, else 1), `d_off_epa`, `d_def_epa`, `d_off_sr`, `d_def_sr`, `d_prior_sp`, `d_talent`, `d_rest` (rest days, capped at 21).
- **Vegas baseline:** `vegas_margin = −median(spread)` across providers. CFBD's `spread` is the home line; negative means the home team is favoured. Vegas's win probability uses `σ_vegas` = the standard deviation of `margin − vegas_margin` on the training seasons.
- **Time-based evaluation:** usable seasons are complete seasons (earlier than the current season) other than the first season in the data. With data for 2022–2026 that's 2023, 2024 and 2025.
  - **Validation:** train on seasons before the second-to-last usable season, and evaluate on it (train 2023 → validate 2024). This picks the model kind by MAE.
  - **Test:** train on seasons before the last usable season, and evaluate on it (train 2023–2024 → test 2025).
  - Report MAE and Brier for the model and for Vegas, overall and for Penn State.
  - **Final model:** refit on every completed usable game, including the current season's played games.
- New runtime dependency: `xgboost>=2.0` (joblib ships with scikit-learn).
- Generated outputs go to `data/models/game_model.joblib`, `data/reports/game_model.json` and `data/reports/game_model.md`. Both directories are gitignored.
- Anything printed to the console must be ASCII only (the Windows console is cp1252).

## Review Focus

1. **Future data leaking into a game's features:** altering the plays of slate k or later must not change any rating used for slate k. Tested in Task 1 (`test_rolling_ratings_use_only_earlier_slates`) and Task 3 (`test_backtest_folds_are_time_ordered`).
2. **A season with no prior** (the first in the data) or a team with no prior (newly FBS): early-slate ratings fall back to the previous season's league mean, the first season is never trained on or scored, and nothing crashes. Tested in Task 1 (`test_first_slate_uses_last_season_and_league_mean_for_new_teams`) and Task 3 (`test_backtest_needs_three_complete_seasons`).
3. **Missing feature values** (no prior SP+ or talent row): they're imputed and predictions stay finite. Tested in Task 3 (`test_fit_learns_signal_and_estimates_sigma`).
4. **Upcoming (not yet played) games** get predictions but are never used for training or scoring. Tested in Task 4 (`test_train_and_save_writes_predictions_model_and_report`, `test_final_model_trains_on_every_completed_game_after_the_first_season`).
5. **Games without a betting line:** Vegas is compared with the model only on games that have a line (`model_on_lined_games`), never on different game sets. Tested in Task 3 (`test_backtest_folds_are_time_ordered`).

## File Map

| File | Responsibility |
|---|---|
| `src/psu/features.py` | `slate_index`, `rest_days`, `vegas_margin`, `league_means`, `season_ratings`, `rolling_ratings`, `game_features`, `FEATURES` |
| `src/psu/models/__init__.py` | package marker |
| `src/psu/models/game_predict.py` | `MODEL_KINDS`, `make_pipeline`, `GameModel`, `fit`, `win_prob`, `scores`, `training_rows`, `backtest` |
| `src/psu/train.py` | `load_features`, `train_and_save`, `report_markdown` |
| `src/psu/cli.py` | adds `psu train` |
| `tests/conftest.py` | adds `synthetic_features()` |

---

### Task 1: Pregame rating features (`features.py`, part 1)

**Files:**
- Modify: `pyproject.toml` (add `"xgboost>=2.0"` to `dependencies`)
- Create: `src/psu/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `psu.adjust.opponent_adjust(enriched, value, *, alpha, exclude_garbage=True)`. It returns `season, team, off_raw, off_adj, def_raw, def_adj, off_plays, def_plays`, centred on the league mean.
- Produces:
  - `MAX_REST = 21`, `RATING_COLUMNS = ["off_epa", "def_epa", "off_sr", "def_sr"]`
  - `slate_index(games) -> DataFrame[id, season, slate]` (games need `id, season, week, season_type`)
  - `rest_days(games) -> DataFrame[id, home_rest, away_rest]` (games need `id, season, start_date, home_team, away_team`)
  - `vegas_margin(lines) -> DataFrame[game_id, vegas_margin]`
  - `league_means(enriched) -> {"epa": float, "sr": float}`, over non-garbage plays (NaN when empty)
  - `season_ratings(enriched, alpha) -> DataFrame indexed by team: off_epa, def_epa, off_sr, def_sr, off_plays, def_plays`
  - `rolling_ratings(enriched, games, *, alpha=50.0, shrink_plays=300) -> DataFrame[season, slate, team, off_epa, def_epa, off_sr, def_sr]`, with one row per team playing in each slate
- Enriched plays need `game_id, season, offense, defense, ppa, success, venue, garbage`.

- [ ] **Step 1: Add the dependency and install**

In `pyproject.toml`, add `"xgboost>=2.0",` to `dependencies` (after `scipy`).
Run: `.venv\Scripts\python -m pip install -e ".[dev]"`. Expected: `Successfully installed ... xgboost-...`.

- [ ] **Step 2: Write the failing tests**

`tests/test_features.py`:
```python
import numpy as np
import pandas as pd
import pytest

from psu.features import (
    MAX_REST, league_means, rest_days, rolling_ratings, season_ratings, slate_index, vegas_margin,
)

SCHEDULE = [(1, "A", "B"), (1, "C", "D"), (2, "A", "C"), (2, "B", "D"), (3, "A", "D"), (3, "B", "C")]
QUALITY = {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4, "E": 0.0}


def make_games(season, schedule=SCHEDULE, start_id=0):
    return pd.DataFrame([
        {"id": season * 100 + start_id + i, "season": season, "week": week, "season_type": "regular",
         "start_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
         "neutral_site": False, "completed": True, "home_team": home, "away_team": away,
         "home_classification": "fbs", "away_classification": "fbs", "home_points": 28, "away_points": 21}
        for i, (week, home, away) in enumerate(schedule)
    ])


def make_plays(games, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in games.itertuples():
        for off, de in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            for v in QUALITY[off] - QUALITY[de] + rng.normal(0, 0.1, n):
                rows.append({"game_id": g.id, "season": g.season, "offense": off, "defense": de,
                             "ppa": v, "success": v > 0, "venue": "neutral", "garbage": False})
    return pd.DataFrame(rows)


def test_slate_index_puts_postseason_after_regular_season():
    games = pd.DataFrame([
        {"id": 1, "season": 2024, "week": 14, "season_type": "regular"},
        {"id": 2, "season": 2024, "week": 1, "season_type": "postseason"},
        {"id": 3, "season": 2024, "week": 3, "season_type": "regular"},
    ])
    assert slate_index(games).set_index("id")["slate"].to_dict() == {1: 14, 2: 15, 3: 3}


def test_rest_days_between_games_and_capped_for_openers():
    rest = rest_days(make_games(2024)).set_index("id")
    assert rest.loc[202400, "home_rest"] == MAX_REST
    assert rest.loc[202402, ["home_rest", "away_rest"]].tolist() == [7, 7]


def test_vegas_margin_is_minus_median_spread():
    lines = pd.DataFrame({"game_id": [1, 1, 1, 2], "spread": [-7.0, -6.5, -7.5, None]})
    assert vegas_margin(lines).set_index("game_id")["vegas_margin"].to_dict() == {1: 7.0}


def test_league_means_skip_garbage():
    plays = pd.DataFrame({"ppa": [1.0, 0.0, 9.0], "success": [True, False, True], "garbage": [False, False, True]})
    assert league_means(plays) == {"epa": 0.5, "sr": 0.5}


def test_rolling_ratings_use_only_earlier_slates():
    games = make_games(2024)
    plays = make_plays(games)
    base = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10)

    late = plays.copy()
    late.loc[late["game_id"].isin(games.loc[games["week"] == 3, "id"]), "ppa"] = 100.0
    pd.testing.assert_frame_equal(base, rolling_ratings(late, games, alpha=1.0, shrink_plays=10))

    mid = plays.copy()
    mid.loc[mid["game_id"].isin(games.loc[games["week"] == 2, "id"]), "ppa"] = 100.0
    changed = rolling_ratings(mid, games, alpha=1.0, shrink_plays=10).set_index(["season", "slate", "team"])
    b = base.set_index(["season", "slate", "team"])
    pd.testing.assert_frame_equal(b.xs(2, level="slate"), changed.xs(2, level="slate"))
    assert not b.xs(3, level="slate").equals(changed.xs(3, level="slate"))


def test_first_slate_uses_last_season_and_league_mean_for_new_teams():
    g23 = make_games(2023)
    games = pd.concat([g23, make_games(2024, schedule=[(1, "A", "E")])], ignore_index=True)
    plays = make_plays(g23)
    out = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10).set_index(["season", "slate", "team"])
    prior = season_ratings(plays, alpha=1.0)
    means = league_means(plays)
    assert out.loc[(2024, 1, "A"), "off_epa"] == pytest.approx(prior.loc["A", "off_epa"])
    assert out.loc[(2024, 1, "E"), "off_epa"] == pytest.approx(means["epa"])
    assert out.loc[(2024, 1, "E"), "def_sr"] == pytest.approx(means["sr"])
    first = out.xs(2023, level="season").xs(1, level="slate")
    assert first.isna().all().all()  # first season, first slate: nothing known yet


def test_shrinkage_blends_prior_and_current():
    g23, g24 = make_games(2023), make_games(2024)
    games = pd.concat([g23, g24], ignore_index=True)
    plays = pd.concat([make_plays(g23), make_plays(g24, seed=1)], ignore_index=True)
    early_2024 = plays[plays["game_id"].isin(g24.loc[g24["week"] < 3, "id"])]
    current = season_ratings(early_2024, alpha=1.0)
    prior = season_ratings(plays[plays["season"] == 2023], alpha=1.0)
    no_shrink = rolling_ratings(plays, games, alpha=1.0, shrink_plays=0).set_index(["season", "slate", "team"])
    all_prior = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10**9).set_index(["season", "slate", "team"])
    assert no_shrink.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(current.loc["A", "off_epa"])
    assert all_prior.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(prior.loc["A", "off_epa"], abs=1e-6)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_features.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.features'`

- [ ] **Step 4: Implement `src/psu/features.py` (part 1)**

```python
"""Pregame features for game prediction. Every feature for a game uses only information from before its slate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from psu.adjust import opponent_adjust

MAX_REST = 21
RATING_COLUMNS = ["off_epa", "def_epa", "off_sr", "def_sr"]
_RATING_FRAME_COLUMNS = [*RATING_COLUMNS, "off_plays", "def_plays"]


def slate_index(games: pd.DataFrame) -> pd.DataFrame:
    """Regular-season week w is slate w; postseason week w follows the season's last regular week."""
    g = games[["id", "season", "week", "season_type"]].copy()
    last_regular = g[g["season_type"] == "regular"].groupby("season")["week"].max()
    offset = g["season"].map(last_regular).fillna(0)
    g["slate"] = np.where(g["season_type"] == "postseason", offset + g["week"], g["week"]).astype(int)
    return g[["id", "season", "slate"]]


def rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game in the same season, capped at MAX_REST (openers get MAX_REST)."""
    sides = []
    for side in ("home", "away"):
        part = games[["id", "season", "start_date", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        sides.append(part.assign(side=side))
    long = pd.concat(sides, ignore_index=True)
    long["start_date"] = pd.to_datetime(long["start_date"])
    long = long.sort_values("start_date")
    previous = long.groupby(["season", "team"])["start_date"].shift()
    long["rest"] = (long["start_date"] - previous).dt.days.clip(upper=MAX_REST).fillna(MAX_REST)
    wide = long.pivot(index="id", columns="side", values="rest")
    wide.columns.name = None
    return wide.rename(columns={"home": "home_rest", "away": "away_rest"}).reset_index()


def vegas_margin(lines: pd.DataFrame) -> pd.DataFrame:
    """Market-expected home margin: minus the median closing home spread across providers."""
    spreads = lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median()
    return (-spreads).rename("vegas_margin").reset_index()


def league_means(enriched: pd.DataFrame) -> dict[str, float]:
    plays = enriched[~enriched["garbage"]]
    return {"epa": float(plays["ppa"].mean()), "sr": float(plays["success"].astype(float).mean())}


def season_ratings(enriched: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Opponent-adjusted EPA and success rate per team over the given plays (one season)."""
    if enriched.empty:
        return pd.DataFrame(columns=_RATING_FRAME_COLUMNS, index=pd.Index([], name="team"), dtype=float)
    epa = opponent_adjust(enriched, "ppa", alpha=alpha).set_index("team")
    sr = opponent_adjust(enriched, "success", alpha=alpha).set_index("team")
    return pd.DataFrame({
        "off_epa": epa["off_adj"], "def_epa": epa["def_adj"],
        "off_sr": sr["off_adj"], "def_sr": sr["def_adj"],
        "off_plays": epa["off_plays"], "def_plays": epa["def_plays"],
    })


def rolling_ratings(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    *,
    alpha: float = 50.0,
    shrink_plays: int = 300,
) -> pd.DataFrame:
    """Ratings for each team as of the start of each slate, blended with last season's final ratings."""
    slates = slate_index(games)
    plays = enriched.merge(slates.rename(columns={"id": "game_id"})[["game_id", "slate"]], on="game_id")
    frames = []
    for season in sorted(slates["season"].unique()):
        previous = plays[plays["season"] == season - 1]
        prior = season_ratings(previous, alpha)
        means = league_means(previous) if not previous.empty else {"epa": np.nan, "sr": np.nan}
        this_season = plays[plays["season"] == season]
        season_games = games[games["season"] == season].merge(slates[["id", "slate"]], on="id")
        for slate, slate_games in season_games.groupby("slate"):
            teams = pd.Index(sorted(set(slate_games["home_team"]) | set(slate_games["away_team"])), name="team")
            current = season_ratings(this_season[this_season["slate"] < slate], alpha).reindex(teams)
            prior_now = prior.reindex(teams)
            out = pd.DataFrame(index=teams)
            for column in RATING_COLUMNS:
                side, kind = column.split("_")
                n = current[f"{side}_plays"].astype(float).fillna(0.0)
                weight = (n / (n + shrink_plays)).fillna(0.0)
                p = prior_now[column].astype(float).fillna(means[kind])
                c = current[column].astype(float)
                blended = weight * c.fillna(0.0) + (1 - weight) * p
                out[column] = blended.where(p.notna(), c)
            out = out.reset_index()
            out.insert(0, "slate", int(slate))
            out.insert(0, "season", int(season))
            frames.append(out)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_features.py`, then `.venv\Scripts\python -m pytest`
Expected: `7 passed`; full suite `131 passed`.

---

### Task 2: Game-level feature table (`features.py`, part 2)

**Files:**
- Modify: `src/psu/features.py` (append)
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
  - `FEATURES = ["home_field", "d_off_epa", "d_def_epa", "d_off_sr", "d_def_sr", "d_prior_sp", "d_talent", "d_rest"]`
  - `GAME_COLUMNS = ["game_id", "season", "week", "season_type", "slate", "start_date", "home_team", "away_team", "neutral_site", "completed", "home_points", "away_points", "margin", "vegas_margin"]`
  - `game_features(enriched, games, lines, sp, talent, *, alpha=50.0, shrink_plays=300) -> DataFrame` with columns `GAME_COLUMNS + FEATURES` and one row per FBS-vs-FBS game.
    - `margin = home_points − away_points` for completed games, NaN otherwise.
    - `d_prior_sp` uses SP+ from `year = season − 1`.
    - `d_talent` uses talent from `year = season`.
- Inputs:
  - `games`: `id, season, week, season_type, start_date, neutral_site, completed, home_team, away_team, home_classification, away_classification, home_points, away_points`
  - `lines`: `game_id, spread`
  - `sp`: `year, team, rating`
  - `talent`: `year, team, talent`

- [ ] **Step 1: Append the failing test to `tests/test_features.py`**

```python
from psu.features import FEATURES, game_features  # noqa: E402  (keep with the other imports if you prefer)


def test_game_features_are_home_minus_away_and_fbs_only():
    g23, g24 = make_games(2023), make_games(2024)
    g24.loc[0, "neutral_site"] = True
    g24.loc[5, ["completed", "home_points", "away_points"]] = [False, np.nan, np.nan]
    fcs = make_games(2024, schedule=[(4, "A", "F")], start_id=50)
    fcs["away_classification"] = "fcs"
    games = pd.concat([g23, g24, fcs], ignore_index=True)
    plays = make_plays(pd.concat([g23, g24], ignore_index=True))
    lines = pd.DataFrame({"game_id": [202400], "spread": [-3.5]})
    sp = pd.DataFrame({"year": [2023, 2023, 2024], "team": ["A", "B", "A"], "rating": [20.0, 5.0, 99.0]})
    talent = pd.DataFrame({"year": [2024, 2024], "team": ["A", "B"], "talent": [900.0, 700.0]})

    f = game_features(plays, games, lines, sp, talent, alpha=1.0, shrink_plays=10).set_index("game_id")
    assert set(FEATURES) <= set(f.columns)
    assert 202450 not in f.index  # FCS opponent excluded
    row = f.loc[202400]  # 2024 week 1: A (home) vs B, neutral site
    assert row["home_field"] == 0 and row["margin"] == 7 and row["vegas_margin"] == 3.5
    assert row["d_prior_sp"] == 15.0  # 2023 SP+, not 2024's 99
    assert row["d_talent"] == 200.0
    assert row["d_off_epa"] > 0  # A's 2023 offense beat B's
    assert f.loc[202401, "home_field"] == 1
    assert np.isnan(f.loc[202405, "margin"])
    assert np.isnan(f.loc[202401, "vegas_margin"])
```
Move the new import line to the top-of-file import block, so the file has a single import section.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_features.py`
Expected: FAIL with `ImportError: cannot import name 'FEATURES'`

- [ ] **Step 3: Append to `src/psu/features.py`**

```python
FEATURES = ["home_field", "d_off_epa", "d_def_epa", "d_off_sr", "d_def_sr", "d_prior_sp", "d_talent", "d_rest"]
GAME_COLUMNS = [
    "game_id", "season", "week", "season_type", "slate", "start_date", "home_team", "away_team",
    "neutral_site", "completed", "home_points", "away_points", "margin", "vegas_margin",
]


def game_features(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    *,
    alpha: float = 50.0,
    shrink_plays: int = 300,
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game: pregame home-minus-away features, the Vegas margin, and the result."""
    ratings = rolling_ratings(enriched, games, alpha=alpha, shrink_plays=shrink_plays)
    g = games[(games["home_classification"] == "fbs") & (games["away_classification"] == "fbs")]
    g = g.merge(slate_index(games)[["id", "slate"]], on="id")
    prior_sp = sp.assign(season=sp["year"] + 1)[["season", "team", "rating"]]
    season_talent = talent.rename(columns={"year": "season"})[["season", "team", "talent"]]
    for side in ("home", "away"):
        team = f"{side}_team"
        g = g.merge(
            ratings.rename(columns={"team": team, **{c: f"{side}_{c}" for c in RATING_COLUMNS}}),
            on=["season", "slate", team], how="left",
        )
        g = g.merge(prior_sp.rename(columns={"team": team, "rating": f"{side}_prior_sp"}), on=["season", team], how="left")
        g = g.merge(season_talent.rename(columns={"team": team, "talent": f"{side}_talent"}), on=["season", team], how="left")
    g = g.merge(rest_days(games), on="id", how="left")
    g = g.merge(vegas_margin(lines).rename(columns={"game_id": "id"}), on="id", how="left")
    g["home_field"] = (~g["neutral_site"].eq(True)).astype(int)
    for column in RATING_COLUMNS:
        g[f"d_{column}"] = g[f"home_{column}"] - g[f"away_{column}"]
    g["d_prior_sp"] = g["home_prior_sp"] - g["away_prior_sp"]
    g["d_talent"] = g["home_talent"] - g["away_talent"]
    g["d_rest"] = g["home_rest"] - g["away_rest"]
    g["margin"] = np.where(g["completed"].eq(True), g["home_points"] - g["away_points"], np.nan)
    return g.rename(columns={"id": "game_id"})[GAME_COLUMNS + FEATURES].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_features.py`, then `.venv\Scripts\python -m pytest`
Expected: `8 passed`; full suite `132 passed`.

---

### Task 3: Margin model, win probability, and backtest (`models/game_predict.py`)

**Files:**
- Create: `src/psu/models/__init__.py`, `src/psu/models/game_predict.py`
- Modify: `tests/conftest.py` (append `synthetic_features`)
- Test: `tests/test_game_predict.py`

**Interfaces:**
- Consumes: `psu.features.FEATURES`. The input frames have `season, home_team, away_team, margin, vegas_margin` plus `FEATURES`.
- Produces:
  - `MODEL_KINDS = ("linear", "xgboost")`
  - `make_pipeline(kind) -> sklearn Pipeline`: a median `SimpleImputer(keep_empty_features=True)`, then either StandardScaler + Ridge(alpha=1.0), or XGBRegressor.
  - `@dataclass GameModel(kind: str, pipeline, sigma: float)` with `.predict_margin(df) -> ndarray` and `.win_prob(df) -> ndarray`
  - `fit(train, kind) -> GameModel`. σ is the std (ddof=1) of 5-fold out-of-fold residuals on `train`.
  - `win_prob(margin, sigma) -> ndarray`
  - `scores(actual, predicted, prob) -> {"n", "mae", "brier"}`. For n = 0 it returns `{"n": 0, "mae": None, "brier": None}`.
  - `usable_seasons(features, current_season) -> list[int]`: seasons with completed games that are before `current_season` and after the first season in the frame.
  - `training_rows(features) -> DataFrame`: completed games (margin not NaN) after the first season in the frame, including the current season.
  - `backtest(features, current_season, team="Penn State") -> dict` with keys:
    - `model_kind`, `validation_season`, `validation` (`{kind: scores}`), `test_season`, `train_seasons`, `sigma`, `vegas_sigma`
    - `test`: `{"all": block, team: block}`, where each block is `{"model", "model_on_lined_games", "vegas"}`
    - All seasons are Python ints and all numbers are Python floats.
  - It raises `ValueError` when fewer than 3 usable seasons exist.
  - `tests/conftest.py`: `synthetic_features(seed=0, games_per_season=240) -> DataFrame` for seasons 2022–2026, where only half of 2026 is completed.

- [ ] **Step 1: Append the synthetic generator to `tests/conftest.py`**

```python
def synthetic_features(seed=0, games_per_season=240):
    """Feature rows for 2022-2026 with a known linear signal; half of 2026 is still to be played."""
    import numpy as np
    import pandas as pd

    from psu.features import FEATURES

    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(30)] + ["Penn State"]
    rows, game_id = [], 0
    for season in range(2022, 2027):
        for i in range(games_per_season):
            game_id += 1
            home, away = rng.choice(teams, 2, replace=False)
            x = {c: float(rng.normal()) for c in FEATURES}
            x["home_field"] = float(rng.random() < 0.9)
            signal = 8 * x["d_off_epa"] - 6 * x["d_def_epa"] + 4 * x["d_prior_sp"] + 2.5 * x["home_field"]
            completed = season < 2026 or i < games_per_season // 2
            rows.append({
                "game_id": game_id, "season": season, "week": 1 + i % 15, "season_type": "regular",
                "home_team": str(home), "away_team": str(away), "completed": completed,
                "margin": signal + rng.normal(0, 12) if completed else np.nan,
                "vegas_margin": signal + rng.normal(0, 3) if i % 10 else np.nan,
                **x,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_game_predict.py`:
```python
import numpy as np
import pytest

from conftest import synthetic_features
from psu.models import game_predict as gp


def test_win_prob_is_a_normal_cdf_of_margin():
    assert gp.win_prob(0.0, 14.0) == pytest.approx(0.5)
    p = gp.win_prob(np.array([-14.0, 7.0, 14.0]), 14.0)
    assert p[0] == pytest.approx(0.1587, abs=1e-4) and p[2] == pytest.approx(0.8413, abs=1e-4)
    assert p[0] < p[1] < p[2]


def test_scores():
    assert gp.scores([7, -3], [3, -1], [0.6, 0.4]) == {"n": 2, "mae": pytest.approx(3.0), "brier": pytest.approx(0.16)}
    assert gp.scores([], [], []) == {"n": 0, "mae": None, "brier": None}


@pytest.mark.parametrize("kind", gp.MODEL_KINDS)
def test_fit_learns_signal_and_estimates_sigma(kind):
    f = synthetic_features()
    train = f[f["season"].isin([2023, 2024])].copy()
    test = f[f["season"] == 2025].copy()
    train.loc[train.index[:20], "d_prior_sp"] = np.nan  # missing values are imputed
    test.loc[test.index[:5], ["d_prior_sp", "d_talent"]] = np.nan
    model = gp.fit(train, kind)
    predicted = model.predict_margin(test)
    assert np.isfinite(predicted).all()
    assert np.mean(np.abs(test["margin"] - predicted)) < np.mean(np.abs(test["margin"]))
    assert 8 < model.sigma < 20


def test_backtest_folds_are_time_ordered(monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind):
        seen.append(set(train["season"]))
        return real_fit(train, kind)

    monkeypatch.setattr(gp, "fit", spy)
    report = gp.backtest(synthetic_features(), current_season=2026)
    assert (report["validation_season"], report["test_season"]) == (2024, 2025)
    assert report["train_seasons"] == [2023, 2024]
    assert seen[: len(gp.MODEL_KINDS)] == [{2023}] * len(gp.MODEL_KINDS)
    assert seen[-1] == {2023, 2024}
    assert report["model_kind"] in gp.MODEL_KINDS
    for block in ("all", "Penn State"):
        assert set(report["test"][block]) == {"model", "model_on_lined_games", "vegas"}
        assert report["test"][block]["model"]["n"] > 0
        assert report["test"][block]["vegas"]["n"] == report["test"][block]["model_on_lined_games"]["n"]
    assert report["test"]["all"]["vegas"]["n"] > 0 and report["test"]["all"]["vegas"]["mae"] is not None
    assert isinstance(report["test_season"], int) and isinstance(report["sigma"], float)


def test_backtest_needs_three_complete_seasons():
    f = synthetic_features()
    with pytest.raises(ValueError):
        gp.backtest(f[f["season"] >= 2024], current_season=2026)


def test_training_rows_skip_first_season_and_unplayed_games():
    rows = gp.training_rows(synthetic_features())
    assert set(rows["season"]) == {2023, 2024, 2025, 2026}
    assert rows["margin"].notna().all()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_game_predict.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.models'`

- [ ] **Step 4: Implement**

`src/psu/models/__init__.py`:
```python
"""Predictive models."""
```

`src/psu/models/game_predict.py`:
```python
"""Game margin model: pipelines, win probability, scoring, and a time-based backtest against Vegas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from psu.features import FEATURES

MODEL_KINDS = ("linear", "xgboost")


def make_pipeline(kind: str) -> Pipeline:
    if kind == "linear":
        steps = [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
    elif kind == "xgboost":
        steps = [("model", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, random_state=0, n_jobs=1))]
    else:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), *steps])


def win_prob(margin, sigma: float) -> np.ndarray:
    """P(home team wins) given a predicted home margin and the margin's error spread."""
    return norm.cdf(np.asarray(margin, dtype=float) / sigma)


@dataclass
class GameModel:
    kind: str
    pipeline: Pipeline
    sigma: float

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[FEATURES])

    def win_prob(self, df: pd.DataFrame) -> np.ndarray:
        return win_prob(self.predict_margin(df), self.sigma)


def fit(train: pd.DataFrame, kind: str) -> GameModel:
    x, y = train[FEATURES], train["margin"]
    out_of_fold = cross_val_predict(make_pipeline(kind), x, y, cv=5)
    sigma = float(np.std(y - out_of_fold, ddof=1))
    return GameModel(kind, make_pipeline(kind).fit(x, y), sigma)


def scores(actual, predicted, prob) -> dict:
    actual = np.asarray(actual, dtype=float)
    if len(actual) == 0:
        return {"n": 0, "mae": None, "brier": None}
    predicted = np.asarray(predicted, dtype=float)
    prob = np.asarray(prob, dtype=float)
    home_won = (actual > 0).astype(float)
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "brier": float(np.mean((prob - home_won) ** 2)),
    }


def usable_seasons(features: pd.DataFrame, current_season: int) -> list[int]:
    first = features["season"].min()
    played = features.loc[features["margin"].notna(), "season"].unique()
    return sorted(int(s) for s in played if first < s < current_season)


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    return features[features["margin"].notna() & (features["season"] > features["season"].min())]


def _vegas_sigma(train: pd.DataFrame) -> float:
    lined = train[train["vegas_margin"].notna()]
    return float(np.std(lined["margin"] - lined["vegas_margin"], ddof=1))


def _score_model(model: GameModel, games: pd.DataFrame) -> dict:
    if games.empty:
        return scores([], [], [])
    return scores(games["margin"], model.predict_margin(games), model.win_prob(games))


def backtest(features: pd.DataFrame, current_season: int, team: str = "Penn State") -> dict:
    seasons = usable_seasons(features, current_season)
    if len(seasons) < 3:
        raise ValueError(f"Need at least 3 complete seasons after the first; have {seasons}")
    played = features[features["margin"].notna()]
    validate_season, test_season = seasons[-2], seasons[-1]

    def split(eval_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        before = [s for s in seasons if s < eval_season]
        return played[played["season"].isin(before)], played[played["season"] == eval_season]

    train, valid = split(validate_season)
    validation = {kind: _score_model(fit(train, kind), valid) for kind in MODEL_KINDS}
    kind = min(MODEL_KINDS, key=lambda k: validation[k]["mae"])

    train, test = split(test_season)
    model = fit(train, kind)
    vegas_sigma = _vegas_sigma(train)

    def block(games: pd.DataFrame) -> dict:
        lined = games[games["vegas_margin"].notna()]
        return {
            "model": _score_model(model, games),
            "model_on_lined_games": _score_model(model, lined),
            "vegas": scores(lined["margin"], lined["vegas_margin"], win_prob(lined["vegas_margin"], vegas_sigma)),
        }

    is_team = test["home_team"].eq(team) | test["away_team"].eq(team)
    return {
        "model_kind": kind,
        "validation_season": validate_season,
        "validation": validation,
        "test_season": test_season,
        "train_seasons": [s for s in seasons if s < test_season],
        "sigma": model.sigma,
        "vegas_sigma": vegas_sigma,
        "test": {"all": block(test), team: block(test[is_team])},
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_game_predict.py`, then `.venv\Scripts\python -m pytest`
Expected: `7 passed`; full suite `139 passed`.

---

### Task 4: Training run, predictions table, saved model and report (`train.py`)

**Files:**
- Create: `src/psu/train.py`
- Modify: `.gitignore` (add `data/models/` and `data/reports/`)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `game_features`, `FEATURES` (Tasks 1–2)
  - `gp.backtest`, `gp.fit`, `gp.training_rows` (Task 3), called through the module as `gp.<name>` so tests can monkeypatch them
  - `enrich_plays(plays, games, drives)` (Phase 2)
  - `psu.build._PLAY_COLUMNS`, `psu.config.TEAM`
- Produces:
  - `PREDICTION_COLUMNS = ["game_id", "season", "week", "season_type", "home_team", "away_team", "completed", "margin", "vegas_margin", "pred_margin", "home_win_prob"]`
  - `load_features(con, *, alpha=50.0, shrink_plays=300) -> DataFrame` reads the raw tables and returns `game_features(...)`.
  - `train_and_save(con, features, *, current_season, out_dir, team=TEAM) -> dict` does the following and returns the report:
    - runs the backtest
    - fits the final model on `gp.training_rows(features)` with the chosen kind
    - writes `game_predictions` (`CREATE OR REPLACE`, one row per feature row)
    - saves `out_dir/models/game_model.joblib`, `out_dir/reports/game_model.json` and `out_dir/reports/game_model.md`
    - adds `final_train_games` (int) to the report
  - `report_markdown(report) -> str`: ASCII-only markdown.

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
import json

import joblib

from conftest import synthetic_features
from psu.db import connect
from psu.models import game_predict as gp
from psu.train import PREDICTION_COLUMNS, report_markdown, train_and_save


def test_train_and_save_writes_predictions_model_and_report(tmp_path):
    con = connect(":memory:")
    features = synthetic_features()
    report = train_and_save(con, features, current_season=2026, out_dir=tmp_path)

    preds = con.execute("SELECT * FROM game_predictions").df()
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == len(features)
    assert preds["home_win_prob"].between(0, 1).all()
    upcoming = preds[preds["margin"].isna()]
    assert len(upcoming) > 0 and upcoming["pred_margin"].notna().all()

    model = joblib.load(tmp_path / "models" / "game_model.joblib")
    assert model.kind == report["model_kind"]
    saved = json.loads((tmp_path / "reports" / "game_model.json").read_text(encoding="utf-8"))
    assert saved["test_season"] == 2025
    assert saved["final_train_games"] == len(gp.training_rows(features))
    md = (tmp_path / "reports" / "game_model.md").read_text(encoding="utf-8")
    assert "Vegas" in md and "Penn State" in md
    report_markdown(report).encode("ascii")  # console-safe


def test_final_model_trains_on_every_completed_game_after_the_first_season(tmp_path, monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind):
        seen.append(set(train["season"]))
        return real_fit(train, kind)

    monkeypatch.setattr(gp, "fit", spy)
    train_and_save(connect(":memory:"), synthetic_features(), current_season=2026, out_dir=tmp_path)
    assert seen[-1] == {2023, 2024, 2025, 2026}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_train.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.train'`

- [ ] **Step 3: Implement `src/psu/train.py`**

```python
"""Train the game model from DuckDB: features, backtest, final fit, predictions table, saved model and report."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from psu.build import _PLAY_COLUMNS
from psu.config import TEAM
from psu.features import game_features
from psu.models import game_predict as gp
from psu.transform import enrich_plays

PREDICTION_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team", "completed",
    "margin", "vegas_margin", "pred_margin", "home_win_prob",
]


def load_features(con: duckdb.DuckDBPyConnection, *, alpha: float = 50.0, shrink_plays: int = 300) -> pd.DataFrame:
    plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays").df()
    games = con.execute(
        "SELECT id, season, week, season_type, start_date, neutral_site, completed, home_team, away_team, "
        "home_classification, away_classification, home_points, away_points FROM games"
    ).df()
    drives = con.execute("SELECT id, offense, start_offense_score, start_defense_score FROM drives").df()
    lines = con.execute("SELECT game_id, spread FROM lines").df()
    sp = con.execute("SELECT year, team, rating FROM ratings_sp").df()
    talent = con.execute("SELECT year, team, talent FROM talent").df()
    enriched = enrich_plays(plays, games, drives)
    return game_features(enriched, games, lines, sp, talent, alpha=alpha, shrink_plays=shrink_plays)


def _fmt(value, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def report_markdown(report: dict) -> str:
    validation = ", ".join(f"{kind} {_fmt(s['mae'], 2)}" for kind, s in report["validation"].items())
    seasons = "-".join(str(s) for s in (report["train_seasons"][0], report["train_seasons"][-1]))
    lines = [
        "# Game model report",
        "",
        f"Model: {report['model_kind']} (chosen on {report['validation_season']} validation MAE: {validation}).",
        f"Trained on {seasons}, tested on {report['test_season']}; final model refit on "
        f"{report.get('final_train_games', '?')} completed games.",
        f"Win probability = NormalCDF(margin / {report['sigma']:.1f}); Vegas uses sigma {report['vegas_sigma']:.1f}.",
        "",
        "| Games | N (lined) | Model MAE | Vegas MAE | Model Brier | Vegas Brier | Model MAE (all games) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, block in report["test"].items():
        lined, vegas, every = block["model_on_lined_games"], block["vegas"], block["model"]
        lines.append(
            f"| {name} | {lined['n']} | {_fmt(lined['mae'], 2)} | {_fmt(vegas['mae'], 2)} | "
            f"{_fmt(lined['brier'], 3)} | {_fmt(vegas['brier'], 3)} | {_fmt(every['mae'], 2)} (n={every['n']}) |"
        )
    return "\n".join(lines) + "\n"


def train_and_save(
    con: duckdb.DuckDBPyConnection,
    features: pd.DataFrame,
    *,
    current_season: int,
    out_dir: Path,
    team: str = TEAM,
) -> dict:
    report = gp.backtest(features, current_season, team=team)
    train = gp.training_rows(features)
    model = gp.fit(train, report["model_kind"])
    report["final_train_games"] = int(len(train))

    predictions = features.copy()
    predictions["pred_margin"] = model.predict_margin(features)
    predictions["home_win_prob"] = model.win_prob(features)
    predictions = predictions[PREDICTION_COLUMNS]
    con.register("_predictions", predictions)
    try:
        con.execute("CREATE OR REPLACE TABLE game_predictions AS SELECT * FROM _predictions")
    finally:
        con.unregister("_predictions")

    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "models" / "game_model.joblib")
    (out_dir / "reports" / "game_model.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "reports" / "game_model.md").write_text(report_markdown(report), encoding="utf-8")
    return report
```

Append to `.gitignore`:
```
data/models/
data/reports/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_train.py`, then `.venv\Scripts\python -m pytest`
Expected: `2 passed`; full suite `141 passed`.

---

### Task 5: `psu train` command

**Files:**
- Modify: `src/psu/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `load_features`, `train_and_save`, `report_markdown` (Task 4); `db.connect`, `db.row_counts` (Phase 1); `settings.current_season`, `settings.db_path`
- Produces:
  - `psu train [--alpha 50] [--shrink-plays 300]`. It writes outputs under `settings.db_path.parent` (i.e. `data/`) and prints the markdown report.
  - Exit codes: 0 = ok; 2 = no plays loaded, or too few seasons (the `ValueError` message is printed).
  - `cli.load_features` is imported into the cli module's namespace, so tests can monkeypatch it.

- [ ] **Step 1: Append the failing tests to `tests/test_cli.py`**

```python
def test_train_requires_ingested_data(settings, capsys):
    assert cli.main(["train"]) == 2
    assert "psu ingest" in capsys.readouterr().err


def test_train_reports_and_writes_outputs(settings, capsys, monkeypatch):
    from conftest import seed_raw_tables, synthetic_features
    from psu import db

    con = db.connect(settings.db_path)
    seed_raw_tables(con)
    con.close()
    monkeypatch.setattr(cli, "load_features", lambda con, **kw: synthetic_features())
    assert cli.main(["train"]) == 0
    assert "Vegas MAE" in capsys.readouterr().out
    assert (settings.db_path.parent / "models" / "game_model.joblib").exists()


def test_train_with_too_few_seasons_is_a_clean_error(settings, capsys, monkeypatch):
    from conftest import seed_raw_tables, synthetic_features
    from psu import db

    con = db.connect(settings.db_path)
    seed_raw_tables(con)
    con.close()
    few = synthetic_features()
    monkeypatch.setattr(cli, "load_features", lambda con, **kw: few[few["season"] >= 2024])
    assert cli.main(["train"]) == 2
    assert "3 complete seasons" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`
Expected: the 3 new tests FAIL (argparse `invalid choice: 'train'` raises `SystemExit`).

- [ ] **Step 3: Implement in `src/psu/cli.py`**

Imports (with the others):
```python
from psu.train import load_features, report_markdown, train_and_save
```
Parser, after the `build` subparser:
```python
    trn = sub.add_parser("train", help="Backtest and train the game model; write predictions (no API calls)")
    trn.add_argument("--alpha", type=float, default=50.0, help="Ridge shrinkage for rolling team ratings")
    trn.add_argument("--shrink-plays", type=int, default=300, help="Plays before a season's own data outweighs last season")
```
Handler, after the `build` handler:
```python
    if args.command == "train":
        con = db.connect(settings.db_path)
        try:
            if db.row_counts(con)["plays"] == 0:
                print("error: no plays loaded yet; run `psu ingest` first", file=sys.stderr)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`, then `.venv\Scripts\python -m pytest`
Expected: `12 passed` in test_cli.py; full suite `144 passed`.

---

### Task 6: Real run, tuning check, README, PR (orchestrator)

**Run by the orchestrator. It reads and writes only `data/` locally, with no API calls.**

- [ ] **Step 1: Train on the real data:** `.venv\Scripts\psu train`. Record the report, and check that runtime is under about 3 minutes.
- [ ] **Step 2: Tune the rating shrinkage (a Phase 2 follow-up):** run `.venv\Scripts\psu train --alpha 200`.
  - Compare the **validation** MAE (the 2024 fold only; never choose on the test season).
  - Keep the better `--alpha` as the default in `cli.py` and `train.py`. If it changes, make that its own commit (`feat(train): default rating alpha to N (lower 2024 validation MAE)`), then re-run `psu train` with the chosen default.
- [ ] **Step 3: Sanity checks** (read-only queries on `game_predictions`):
  - The Penn State 2026 upcoming games have sensible margins and win probabilities.
  - `corr(pred_margin, vegas_margin)` over the 2025 games is above 0.7.
  - The average home win probability is about 0.55–0.6.
- [ ] **Step 4: README Phase 3 section:** cover how to run it, what the model uses, the no-leakage design, the report location, and the test-season results table (model vs Vegas, overall and Penn State).
- [ ] **Step 5:** Full suite, commit the README (`docs: add Phase 3 game model guide to README`), push `feat/phase3-game-model`, and open the PR with `gh`. Then stop for the user's review.
