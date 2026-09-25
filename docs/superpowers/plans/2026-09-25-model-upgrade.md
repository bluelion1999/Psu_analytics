# Model Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve early-season predictions and win-probability calibration, and record how the season simulation changes week to week.

**Architecture:**
- **Priors:** CFBD returning production is ingested into a new table. A new pure module, `psu/priors.py`, turns last season's final ratings into a projected preseason prior, and `features.rolling_ratings` uses it.
- **Calibration:** `game_predict` estimates per-phase sigmas from walk-forward out-of-sample residuals and reports reliability. The simulator samples with those per-phase sigmas.
- **History:** a `sim_history` table records one row per (season, as-of slate, team). `psu simulate --backfill` replays earlier weeks using ratings frozen at each week, and the dashboard charts the result.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn (Ridge only), xgboost, DuckDB, Streamlit + Altair, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-25-model-upgrade-design.md`

## Global Constraints

- Seasons are 2022 and later (`config.FIRST_SEASON = 2022`). The 2022 season keeps the `no_prior` split.
- Tests never call the CFBD API. Use `tests/conftest.py`'s `FakeCFBD` and synthetic frames.
- The new endpoint uses the 7-day `SLOW_REFRESH`, like talent and recruiting.
- Never import `sklearn.impute` or `sklearn.neighbors`, because Windows Smart App Control blocks them.
- The tuned defaults `TRAIN_ALPHA = 20.0` and `SHRINK_PLAYS = 75` come only from `psu.config`.
- Refresh stays manual; add no scheduling.
- Style: ruff with line length 120. Run `.venv/Scripts/ruff check .` and `.venv/Scripts/ruff format --check .` before each commit.
- Commits: Conventional Commits, one per task, and every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run tests with `.venv/Scripts/python -m pytest -q`. The full suite must pass at the end of every task.

## Review Focus

1. **A database built before this change** has no `returning_production` table. `psu train` must still work, treating returning production as unknown (Task 3 test `test_load_feature_inputs_without_returning_table`).
2. **A model report written before this change** has no `sigma_by_phase`. `psu simulate` must still run using the pooled sigma (Task 5 test `test_load_sigmas_falls_back_for_old_reports`).
3. **A team with no returning-production row**, or with no rating from last season (new to FBS), must get a finite prior: the season-median returning share, or the league mean (Task 2 `test_returning_pct_fills_missing_teams_with_the_season_median`; the existing Task 3 new-team test is kept).
4. **`psu simulate --backfill` before any game is played** must write nothing, print a message and exit 0 (Task 6 `test_backfill_before_any_games_writes_nothing`).
5. **Running `psu simulate` twice in the same week** must leave exactly one history row for that week, not two (Task 5 `test_write_history_replaces_the_same_week_and_keeps_others`).

---

### Task 1: Ingest returning production

**Files:**
- Modify: `src/psu/cfbd_api.py` (`ENDPOINTS`)
- Modify: `src/psu/db.py` (`SPECS`: add a `TableSpec` after `recruiting`)
- Modify: `src/psu/flatten.py` (`FLATTENERS`)
- Modify: `src/psu/ingest.py` (`season_requests`, the slow-refresh endpoint set)
- Modify: `tests/conftest.py` (`FakeCFBD`: answer the new endpoint)
- Test: `tests/test_flatten.py`, `tests/test_db.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces:
  - Endpoint and table name `returning_production`, key `("season", "team")`.
  - Columns: `season INTEGER, team VARCHAR, conference VARCHAR, percent_ppa, percent_passing_ppa, percent_receiving_ppa, percent_rushing_ppa, usage` (all DOUBLE). Undeclared numeric columns (`total_ppa`, `passing_usage`, ...) are kept as DOUBLE.

- [ ] **Step 1: Write the failing tests**

In `tests/test_flatten.py`, add:

```python
def test_returning_production_flattens_to_snake_case():
    from psu.flatten import FLATTENERS

    df = FLATTENERS["returning_production"](
        [{"season": 2025, "team": "Penn State", "conference": "Big Ten", "percentPPA": 0.62, "totalPassingPPA": 80.5}],
        {"year": 2025},
    )
    assert df.loc[0, "season"] == 2025 and df.loc[0, "percent_ppa"] == 0.62
    assert df.loc[0, "total_passing_ppa"] == 80.5
```

In `tests/test_db.py`, add:

```python
def test_returning_production_round_trip(con):
    spec = SPECS["returning_production"]
    assert spec.key == ("season", "team")
    df = pd.DataFrame({"season": [2025], "team": ["Penn State"], "percent_ppa": [0.62], "total_ppa": [140.0]})
    upsert(con, spec, df)
    upsert(con, spec, df.assign(percent_ppa=0.7))  # same key: replaced, not duplicated
    assert rows(con, "SELECT season, team, percent_ppa, total_ppa FROM returning_production") == [
        (2025, "Penn State", 0.7, 140.0)
    ]
```

In `tests/conftest.py` `FakeCFBD`, add this before the final `raise AssertionError`:

```python
        if endpoint == "returning_production":
            return [{"season": y, "team": "Penn State", "conference": "Big Ten", "percentPPA": 0.6, "usage": 0.55}]
```

In `tests/test_ingest.py`:
- Change `SEASON_LEVEL = 7` to `SEASON_LEVEL = 8` and update its comment to list `returning_production`.
- In `test_current_season_skips_future_weeks_and_refreshes_only_live_data`, after `assert second.api_calls == 5 + WEEKLY`, add:

```python
    assert "returning_production" not in {e for e, _ in fake_cfbd.calls}  # slow refresh, like talent
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_flatten.py tests/test_db.py tests/test_ingest.py -q`
Expected: FAIL. The failures are a `KeyError: 'returning_production'` and the ingest call counts being off by one per season.

- [ ] **Step 3: Implement**

`src/psu/cfbd_api.py`: add to `ENDPOINTS`:

```python
    "returning_production": ("PlayersApi", "get_returning_production"),
```

`src/psu/db.py`: add after the `recruiting` `TableSpec`:

```python
        TableSpec(
            "returning_production",
            ("season", "team"),
            {
                "season": "INTEGER",
                "team": "VARCHAR",
                "conference": "VARCHAR",
                "percent_ppa": "DOUBLE",
                "percent_passing_ppa": "DOUBLE",
                "percent_receiving_ppa": "DOUBLE",
                "percent_rushing_ppa": "DOUBLE",
                "usage": "DOUBLE",
            },
            extra_type="DOUBLE",
        ),
```

`src/psu/flatten.py`: add to `FLATTENERS`:

```python
    "returning_production": flatten_wide,
```

`src/psu/ingest.py`:
- Add `("returning_production", {"year": season}),` as the last entry of `season_requests`.
- Replace the hard-coded `("talent", "recruiting")` in `ingest()` with a module constant:

```python
SLOW_ENDPOINTS = ("talent", "recruiting", "returning_production")  # change at most a few times a year
```

and in `ingest()`:

```python
            Request(e, p, SLOW_REFRESH if e in SLOW_ENDPOINTS else refresh_after, season_final_at)
```

Also update the `SLOW_REFRESH` comment to say "talent, recruiting, returning production and the calendar change rarely".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass. `test_every_endpoint_maps_to_a_real_cfbd_method` now also covers `PlayersApi.get_returning_production`.

- [ ] **Step 5: Commit**

```bash
git add src/psu/cfbd_api.py src/psu/db.py src/psu/flatten.py src/psu/ingest.py tests/conftest.py tests/test_flatten.py tests/test_db.py tests/test_ingest.py
git commit -m "feat(ingest): pull CFBD returning production into returning_production"
```

---

### Task 2: Preseason prior projection (`psu/priors.py`)

**Files:**
- Create: `src/psu/priors.py`
- Test: `tests/test_priors.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. It works on plain DataFrames; the `returning` frame has columns `season, team, percent_ppa`.
- Produces, all importable from `psu.priors`:
  - `FALLBACK_B0: float = 0.6`, `DEFAULT_RETURNING: float = 0.5`, `MIN_PAIRS: int = 20`
  - `@dataclass(frozen=True) class Projection(b0: float, b1: float = 0.0)` with method `factor(ret: np.ndarray) -> np.ndarray` (clipped to [0, 1])
  - `empty_returning() -> pd.DataFrame` (columns `season, team, percent_ppa`)
  - `returning_pct(returning: pd.DataFrame, season: int, teams) -> pd.Series` (indexed by team)
  - `projection_pairs(finals: dict[int, pd.DataFrame], means: dict[int, dict[str, float]], returning: pd.DataFrame, column: str) -> pd.DataFrame` (columns `season, team, x, y, ret`)
  - `fit_projection(pairs: pd.DataFrame, column: str) -> Projection`
  - `season_projections(pairs: dict[str, pd.DataFrame], season: int) -> dict[str, Projection]`
  - `project_prior(last: pd.DataFrame, means: dict[str, float], ret: pd.Series, projections: dict[str, Projection]) -> pd.DataFrame`
- Conventions:
  - `finals[s]` is indexed by team and has rating columns like `off_epa`.
  - `means[s]` is `{"epa": float, "sr": float}`, the `league_means` shape.
  - A column's kind is `column.split("_")[1]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_priors.py`:

```python
import numpy as np
import pandas as pd
import pytest

from psu.priors import (
    DEFAULT_RETURNING,
    FALLBACK_B0,
    MIN_PAIRS,
    Projection,
    empty_returning,
    fit_projection,
    project_prior,
    projection_pairs,
    returning_pct,
    season_projections,
)

MEANS = {s: {"epa": 0.0, "sr": 0.4} for s in (2022, 2023, 2024)}


def synthetic(b0=0.3, b1=0.5, d=0.55, n=200, seed=0, first=2022):
    """Final ratings for `first` and `first + 1` where next = (b0 + b1*ret)*last (offense) and d*last (defense)."""
    rng = np.random.default_rng(seed)
    teams = pd.Index([f"T{i}" for i in range(n)], name="team")
    ret = pd.Series(rng.uniform(0.2, 0.9, n), index=teams)
    last = pd.DataFrame({"off_epa": rng.normal(0, 0.1, n), "def_epa": rng.normal(0, 0.1, n)}, index=teams)
    nxt = pd.DataFrame(
        {
            "off_epa": (b0 + b1 * ret) * last["off_epa"] + rng.normal(0, 0.002, n),
            "def_epa": d * last["def_epa"] + rng.normal(0, 0.002, n),
        },
        index=teams,
    )
    returning = pd.DataFrame({"season": first + 1, "team": teams, "percent_ppa": ret.to_numpy()})
    return {first: last, first + 1: nxt}, returning


def test_fit_recovers_offense_and_defense_regression():
    finals, ret = synthetic()
    off = fit_projection(projection_pairs(finals, MEANS, ret, "off_epa"), "off_epa")
    de = fit_projection(projection_pairs(finals, MEANS, ret, "def_epa"), "def_epa")
    assert off.b0 == pytest.approx(0.3, abs=0.03) and off.b1 == pytest.approx(0.5, abs=0.05)
    assert de.b0 == pytest.approx(0.55, abs=0.02) and de.b1 == 0.0  # defense ignores returning production


def test_pairs_are_relative_to_each_season_mean():
    finals, ret = synthetic()
    means = {2022: {"epa": 0.1, "sr": 0.0}, 2023: {"epa": -0.1, "sr": 0.0}}
    pairs = projection_pairs(finals, means, ret, "off_epa")
    assert list(pairs.columns) == ["season", "team", "x", "y", "ret"]
    row = pairs.iloc[0]
    assert row["season"] == 2023
    assert row["x"] == pytest.approx(finals[2022].loc[row["team"], "off_epa"] - 0.1)
    assert row["y"] == pytest.approx(finals[2023].loc[row["team"], "off_epa"] + 0.1)


def test_too_few_pairs_fall_back():
    finals, ret = synthetic(n=MIN_PAIRS - 1)
    assert fit_projection(projection_pairs(finals, MEANS, ret, "off_epa"), "off_epa") == Projection(FALLBACK_B0)
    lone = {2023: finals[2023]}  # no previous season: no pairs at all
    assert projection_pairs(lone, MEANS, ret, "off_epa").empty
    assert fit_projection(projection_pairs(lone, MEANS, ret, "off_epa"), "off_epa") == Projection(FALLBACK_B0)


def test_season_projections_use_only_earlier_seasons():
    early, r1 = synthetic(b0=0.3, b1=0.0, d=0.3, first=2022)
    late, r2 = synthetic(b0=0.9, b1=0.0, d=0.9, first=2023, seed=1)
    # 2024's finals come from an unrelated draw, so 2024 pairs are noise: only the 2023 pairs carry b0 = 0.3.
    finals = {2022: early[2022], 2023: early[2023], 2024: late[2024]}
    returning = pd.concat([r1, r2], ignore_index=True)
    pairs = {c: projection_pairs(finals, MEANS, returning, c) for c in ("off_epa", "def_epa")}
    assert season_projections(pairs, 2023)["off_epa"] == Projection(FALLBACK_B0)  # nothing before 2023
    assert season_projections(pairs, 2024)["off_epa"].b0 == pytest.approx(0.3, abs=0.03)  # 2023 pairs only
    assert season_projections(pairs, 2024)["def_epa"].b0 == pytest.approx(0.3, abs=0.03)


def test_factor_is_clipped_to_unit_interval():
    np.testing.assert_allclose(Projection(0.9, 0.5).factor(np.array([0.0, 0.5, 1.0])), [0.9, 1.0, 1.0])
    np.testing.assert_allclose(Projection(-0.2).factor(np.array([0.3])), [0.0])


def test_returning_pct_fills_missing_teams_with_the_season_median():
    ret = pd.DataFrame(
        {"season": [2024, 2024, 2024, 2023], "team": ["A", "B", "C", "A"], "percent_ppa": [0.2, 0.4, 0.9, 0.99]}
    )
    out = returning_pct(ret, 2024, ["A", "Z"])
    assert list(out.index) == ["A", "Z"] and list(out) == [0.2, 0.4]
    assert list(returning_pct(empty_returning(), 2024, ["A"])) == [DEFAULT_RETURNING]


def test_project_prior_regresses_toward_the_mean():
    last = pd.DataFrame({"off_epa": [0.3, -0.1], "def_epa": [0.2, 0.0]}, index=pd.Index(["A", "B"], name="team"))
    ret = pd.Series([1.0, 0.0], index=["A", "B"])
    projections = {"off_epa": Projection(0.2, 0.6), "def_epa": Projection(0.5)}
    prior = project_prior(last, {"epa": 0.1, "sr": 0.4}, ret, projections)
    assert prior.loc["A", "off_epa"] == pytest.approx(0.1 + 0.8 * 0.2)
    assert prior.loc["B", "off_epa"] == pytest.approx(0.1 + 0.2 * -0.2)
    assert prior.loc["A", "def_epa"] == pytest.approx(0.1 + 0.5 * 0.1)
    assert list(prior.index) == ["A", "B"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_priors.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.priors'`.

- [ ] **Step 3: Implement `src/psu/priors.py`**

```python
"""Preseason priors: last season's final ratings, regressed toward the league mean by how much production returns.

Offense ratings regress by b0 + b1 * (share of last season's offensive PPA that returns); CFBD's returning
production covers offense only, so defense ratings use b0 alone. Coefficients for a season are fitted only on
earlier season pairs, so a prior never sees the season it predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FALLBACK_B0 = 0.6  # used when there are too few earlier season pairs to fit
DEFAULT_RETURNING = 0.5  # returning share when a season has no returning-production data at all
MIN_PAIRS = 20
PAIR_COLUMNS = ["season", "team", "x", "y", "ret"]


@dataclass(frozen=True)
class Projection:
    b0: float
    b1: float = 0.0

    def factor(self, ret) -> np.ndarray:
        """How much of last season's distance from the mean carries over, clipped to [0, 1]."""
        return np.clip(self.b0 + self.b1 * np.asarray(ret, dtype=float), 0.0, 1.0)


def empty_returning() -> pd.DataFrame:
    return pd.DataFrame(
        {"season": pd.Series(dtype="int64"), "team": pd.Series(dtype=object), "percent_ppa": pd.Series(dtype=float)}
    )


def returning_pct(returning: pd.DataFrame, season: int, teams) -> pd.Series:
    """Each team's returning share of offensive PPA for `season`; unknown teams get that season's median."""
    teams = pd.Index(teams, name="team")
    rows = returning.loc[returning["season"] == season].drop_duplicates("team").set_index("team")["percent_ppa"]
    known = rows.astype(float).dropna()
    fill = float(known.median()) if len(known) else DEFAULT_RETURNING
    return rows.astype(float).reindex(teams).fillna(fill)


def projection_pairs(
    finals: dict[int, pd.DataFrame], means: dict[int, dict[str, float]], returning: pd.DataFrame, column: str
) -> pd.DataFrame:
    """One row per team per season S: x = last(S-1) - mean(S-1), y = final(S) - mean(S), ret = returning share in S."""
    kind = column.split("_")[1]
    frames = []
    for season in sorted(finals):
        if season - 1 not in finals:
            continue
        both = pd.concat(
            {"last": finals[season - 1][column], "next": finals[season][column]}, axis=1, join="inner"
        ).dropna()
        if both.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team": both.index.to_numpy(),
                    "x": (both["last"] - means[season - 1][kind]).to_numpy(float),
                    "y": (both["next"] - means[season][kind]).to_numpy(float),
                    "ret": returning_pct(returning, season, both.index).to_numpy(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)


def fit_projection(pairs: pd.DataFrame, column: str) -> Projection:
    """Least squares on y = (b0 + b1*ret) * x for offense columns, y = b0 * x for defense columns."""
    pairs = pairs.dropna(subset=["x", "y", "ret"])
    if len(pairs) < MIN_PAIRS:
        return Projection(FALLBACK_B0)
    x, y, ret = (pairs[c].to_numpy(float) for c in ("x", "y", "ret"))
    if column.startswith("off_"):
        (b0, b1), *_ = np.linalg.lstsq(np.column_stack([x, x * ret]), y, rcond=None)
        return Projection(float(b0), float(b1))
    denom = float(x @ x)
    return Projection(float(x @ y / denom)) if denom > 0 else Projection(FALLBACK_B0)


def season_projections(pairs: dict[str, pd.DataFrame], season: int) -> dict[str, Projection]:
    """Projections for `season`, fitted only on pairs whose target season is earlier."""
    return {column: fit_projection(p[p["season"] < season], column) for column, p in pairs.items()}


def project_prior(
    last: pd.DataFrame, means: dict[str, float], ret: pd.Series, projections: dict[str, Projection]
) -> pd.DataFrame:
    """prior = mean + factor * (last - mean), per rating column in `projections`."""
    out = last.copy()
    r = ret.reindex(last.index).to_numpy(float)
    for column, projection in projections.items():
        mean = means[column.split("_")[1]]
        out[column] = mean + projection.factor(r) * (last[column].astype(float) - mean)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_priors.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/priors.py tests/test_priors.py
git commit -m "feat(priors): project preseason ratings from last season and returning production"
```

---

### Task 3: Use projected priors and `d_returning` in the features

**Files:**
- Modify: `src/psu/features.py`
  - `rolling_ratings` gains a `returning` argument and uses the projected prior.
  - `game_features` is split into `game_features` plus a new `assemble_features`.
  - `d_returning` is added to `FEATURES`.
- Modify: `src/psu/train.py` (new `FeatureInputs` and `load_feature_inputs`; `load_features` uses them)
- Modify: `tests/test_features.py`: update the two tests that assumed the raw prior, and add new tests.
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes (Task 2): `empty_returning`, `returning_pct`, `projection_pairs`, `season_projections`, `project_prior` from `psu.priors`.
- Produces:
  - `rolling_ratings(enriched, games, *, alpha: float, shrink_plays: int, returning: pd.DataFrame | None = None) -> pd.DataFrame`. Same columns as before: `season, slate, team, off_epa, def_epa, off_sr, def_sr`.
  - `assemble_features(ratings, games, lines, sp, talent, returning=None) -> pd.DataFrame`. This is everything `game_features` did after computing ratings, plus `d_returning`.
  - `game_features(enriched, games, lines, sp, talent, *, alpha, shrink_plays, returning=None)`, which is `assemble_features(rolling_ratings(...), ...)`.
  - `FEATURES` gains `"d_returning"` as its last entry.
  - In `psu.train`: `@dataclass(frozen=True) class FeatureInputs(enriched, games, lines, sp, talent, returning)` and `load_feature_inputs(con) -> FeatureInputs`.

- [ ] **Step 1: Write and update the tests**

In `tests/test_features.py`, add `from psu.priors import FALLBACK_B0` to the imports and **replace** the bodies of two existing tests. Their priors are now projected; with only one earlier season there are no pairs, so the fallback `b0` is used.

```python
def test_first_slate_uses_last_season_and_league_mean_for_new_teams():
    g23 = make_games(2023)
    games = pd.concat([g23, make_games(2024, schedule=[(1, "A", "E")])], ignore_index=True)
    plays = make_plays(g23)
    out = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10).set_index(["season", "slate", "team"])
    prior = season_ratings(plays, alpha=1.0)
    means = league_means(plays)
    projected = means["epa"] + FALLBACK_B0 * (prior.loc["A", "off_epa"] - means["epa"])
    assert out.loc[(2024, 1, "A"), "off_epa"] == pytest.approx(projected)
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
    last = plays[plays["season"] == 2023]
    prior = season_ratings(last, alpha=1.0)
    mean = league_means(last)["epa"]
    projected = mean + FALLBACK_B0 * (prior.loc["A", "off_epa"] - mean)
    no_shrink = rolling_ratings(plays, games, alpha=1.0, shrink_plays=0).set_index(["season", "slate", "team"])
    all_prior = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10**9).set_index(["season", "slate", "team"])
    assert no_shrink.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(current.loc["A", "off_epa"])
    assert all_prior.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(projected, abs=1e-6)
```

Add these new tests:

```python
def test_prior_with_few_teams_falls_back_and_ignores_returning_production():
    seasons = [2021, 2022, 2023, 2024]
    games = pd.concat([make_games(s) for s in seasons], ignore_index=True)
    plays = pd.concat([make_plays(make_games(s), seed=s) for s in seasons[:-1]], ignore_index=True)
    teams = ["A", "B", "C", "D"]
    high = pd.DataFrame({"season": 2024, "team": teams, "percent_ppa": 0.95})
    low = high.assign(percent_ppa=0.05)
    hi = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10, returning=high).set_index(["season", "slate", "team"])
    lo = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10, returning=low).set_index(["season", "slate", "team"])
    assert np.isfinite(hi.loc[(2024, 1, "A"), "off_epa"])
    # Only 4 teams, so no season has MIN_PAIRS pairs: the fallback ignores returning production.
    assert hi.loc[(2024, 1, "A"), "off_epa"] == pytest.approx(lo.loc[(2024, 1, "A"), "off_epa"])


def test_rolling_ratings_fit_each_season_on_earlier_pairs_only(monkeypatch):
    import psu.features as features

    seen = []
    real = features.season_projections

    def spy(pairs, season):
        seen.append(season)
        return real(pairs, season)

    monkeypatch.setattr(features, "season_projections", spy)
    games = pd.concat([make_games(s) for s in (2022, 2023, 2024)], ignore_index=True)
    plays = make_plays(games)
    rolling_ratings(plays, games, alpha=1.0, shrink_plays=10)
    assert seen == [2023, 2024]  # 2022 has no previous season, so no prior to project
    # season_projections itself keeps only pairs before `season` (tested in test_priors)


def test_game_features_include_d_returning():
    g23, g24 = make_games(2023), make_games(2024)
    games = pd.concat([g23, g24], ignore_index=True)
    plays = make_plays(pd.concat([g23, g24], ignore_index=True))
    lines = pd.DataFrame({"game_id": [202400], "spread": [-3.5]})
    sp = pd.DataFrame({"year": [2023], "team": ["A"], "rating": [1.0]})  # non-empty: keeps merge dtypes numeric
    talent = pd.DataFrame({"year": [2024], "team": ["A"], "talent": [1.0]})
    returning = pd.DataFrame({"season": [2024, 2024], "team": ["A", "B"], "percent_ppa": [0.8, 0.3]})
    f = game_features(
        plays, games, lines, sp, talent, alpha=1.0, shrink_plays=10, returning=returning
    ).set_index("game_id")
    assert FEATURES[-1] == "d_returning"
    assert f.loc[202400, "d_returning"] == pytest.approx(0.5)  # 2024 week 1: A (home) vs B
    assert np.isnan(f.loc[202401, "d_returning"])  # C vs D: no rows, left for the imputer
```

In `tests/test_train.py`, add:

```python
def test_load_feature_inputs_without_returning_table():
    from conftest import seed_raw_tables

    from psu.train import load_feature_inputs

    con = connect(":memory:")
    seed_raw_tables(con)
    inputs = load_feature_inputs(con)  # database built before returning production existed
    assert list(inputs.returning.columns) == ["season", "team", "percent_ppa"] and inputs.returning.empty
```

Before writing this test, check that `seed_raw_tables` creates `plays`, `games`, `drives`, `lines`, `ratings_sp` and `talent`. If it does not create `lines`, `ratings_sp` or `talent`, upsert one empty-but-typed row set for each with `db.upsert(con, db.SPECS[name], pd.DataFrame(columns=list(db.SPECS[name].columns)))`, and note that in the report.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_features.py tests/test_train.py -q`
Expected: FAIL. The updated prior tests fail because of the projection value, `returning=` is an unexpected keyword, `d_returning` is missing, and `load_feature_inputs` does not exist.

- [ ] **Step 3: Implement**

In `src/psu/features.py`:

1. Add the imports:

```python
from psu.priors import empty_returning, project_prior, projection_pairs, returning_pct, season_projections
```

2. Replace `rolling_ratings` with:

```python
def rolling_ratings(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    *,
    alpha: float,
    shrink_plays: int,
    returning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ratings for each team as of the start of each slate, blended with a projected prior from last season."""
    returning = empty_returning() if returning is None else returning
    slates = slate_index(games)
    plays = enriched.merge(slates.rename(columns={"id": "game_id"})[["game_id", "slate"]], on="game_id")
    seasons = [int(s) for s in sorted(slates["season"].unique())]
    by_season = {s: plays[plays["season"] == s] for s in seasons}
    finals = {s: season_ratings(p, alpha) for s, p in by_season.items() if not p.empty}
    means = {s: league_means(p) for s, p in by_season.items() if not p.empty}
    pairs = {c: projection_pairs(finals, means, returning, c) for c in RATING_COLUMNS}
    no_means = {"epa": np.nan, "sr": np.nan}
    frames = []
    for season in seasons:
        last = finals.get(season - 1)
        season_means = means.get(season - 1, no_means)
        if last is None:
            prior = season_ratings(plays.iloc[:0], alpha)
        else:
            prior = project_prior(
                last, season_means, returning_pct(returning, season, last.index), season_projections(pairs, season)
            )
        this_season = by_season[season]
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
                p = prior_now[column].astype(float).fillna(season_means[kind])
                c = current[column].astype(float)
                blended = weight * c.fillna(0.0) + (1 - weight) * p
                out[column] = blended.where(p.notna(), c)
            out = out.reset_index()
            out.insert(0, "slate", int(slate))
            out.insert(0, "season", int(season))
            frames.append(out)
    return pd.concat(frames, ignore_index=True)
```

This keeps the old behaviour except for the prior: a season with no previous season gets an empty prior with NaN means, so its ratings are current-only.

3. Change `FEATURES` to end with `"d_returning"`:

```python
FEATURES = [
    "home_field",
    "d_off_epa",
    "d_def_epa",
    "d_off_sr",
    "d_def_sr",
    "d_prior_sp",
    "d_talent",
    "d_rest",
    "d_returning",
]
```

4. Replace `game_features` with a thin wrapper around a new `assemble_features`:

```python
def assemble_features(
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    returning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game from precomputed slate ratings: home-minus-away features, Vegas, result."""
    returning = empty_returning() if returning is None else returning
    g = games[(games["home_classification"] == "fbs") & (games["away_classification"] == "fbs")]
    g = g.merge(slate_index(games)[["id", "slate"]], on="id")
    prior_sp = sp.assign(season=sp["year"] + 1)[["season", "team", "rating"]]
    season_talent = talent.rename(columns={"year": "season"})[["season", "team", "talent"]]
    season_returning = returning[["season", "team", "percent_ppa"]].drop_duplicates(["season", "team"])
    for side in ("home", "away"):
        team = f"{side}_team"
        g = g.merge(
            ratings.rename(columns={"team": team, **{c: f"{side}_{c}" for c in RATING_COLUMNS}}),
            on=["season", "slate", team],
            how="left",
        )
        g = g.merge(
            prior_sp.rename(columns={"team": team, "rating": f"{side}_prior_sp"}), on=["season", team], how="left"
        )
        g = g.merge(
            season_talent.rename(columns={"team": team, "talent": f"{side}_talent"}), on=["season", team], how="left"
        )
        g = g.merge(
            season_returning.rename(columns={"team": team, "percent_ppa": f"{side}_returning"}),
            on=["season", team],
            how="left",
        )
    g = g.merge(rest_days(games), on="id", how="left")
    g = g.merge(vegas_margin(lines).rename(columns={"game_id": "id"}), on="id", how="left")
    g["home_field"] = (~g["neutral_site"].eq(True)).astype(int)
    for column in RATING_COLUMNS:
        g[f"d_{column}"] = g[f"home_{column}"] - g[f"away_{column}"]
    g["d_prior_sp"] = g["home_prior_sp"] - g["away_prior_sp"]
    g["d_talent"] = g["home_talent"] - g["away_talent"]
    g["d_rest"] = g["home_rest"] - g["away_rest"]
    g["d_returning"] = g["home_returning"].astype(float) - g["away_returning"].astype(float)
    g["margin"] = np.where(g["completed"].eq(True), g["home_points"] - g["away_points"], np.nan)
    return g.rename(columns={"id": "game_id"})[GAME_COLUMNS + FEATURES].reset_index(drop=True)


def game_features(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    *,
    alpha: float,
    shrink_plays: int,
    returning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game: pregame home-minus-away features, the Vegas margin, and the result."""
    ratings = rolling_ratings(enriched, games, alpha=alpha, shrink_plays=shrink_plays, returning=returning)
    return assemble_features(ratings, games, lines, sp, talent, returning)
```

In `src/psu/train.py`, add `from dataclasses import dataclass` and `from psu.priors import empty_returning`, then replace `load_features` with:

```python
@dataclass(frozen=True)
class FeatureInputs:
    enriched: pd.DataFrame
    games: pd.DataFrame
    lines: pd.DataFrame
    sp: pd.DataFrame
    talent: pd.DataFrame
    returning: pd.DataFrame


def _returning(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'returning_production'"
    ).fetchone()[0]
    if not exists:
        return empty_returning()  # database built before returning production was ingested
    return con.execute("SELECT season, team, percent_ppa FROM returning_production").df()


def load_feature_inputs(con: duckdb.DuckDBPyConnection) -> FeatureInputs:
    plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays").df()
    games = con.execute(
        "SELECT id, season, week, season_type, start_date, neutral_site, completed, home_team, away_team, "
        "home_classification, away_classification, home_points, away_points FROM games"
    ).df()
    drives = con.execute("SELECT id, offense, start_offense_score, start_defense_score FROM drives").df()
    return FeatureInputs(
        enriched=enrich_plays(plays, games, drives),
        games=games,
        lines=con.execute("SELECT game_id, spread FROM lines").df(),
        sp=con.execute("SELECT year, team, rating FROM ratings_sp").df(),
        talent=con.execute("SELECT year, team, talent FROM talent").df(),
        returning=_returning(con),
    )


def load_features(con: duckdb.DuckDBPyConnection, *, alpha: float = 20.0, shrink_plays: int = 75) -> pd.DataFrame:
    i = load_feature_inputs(con)
    return game_features(
        i.enriched, i.games, i.lines, i.sp, i.talent, alpha=alpha, shrink_plays=shrink_plays, returning=i.returning
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass. `synthetic_features()` in conftest builds every `FEATURES` column from random normals, so `d_returning` is picked up automatically.

- [ ] **Step 5: Commit**

```bash
git add src/psu/features.py src/psu/train.py tests/test_features.py tests/test_train.py
git commit -m "feat(features): project preseason priors with returning production; add d_returning"
```

---

### Task 4: Phase sigmas and calibration report

**Files:**
- Modify: `src/psu/models/game_predict.py`
- Modify: `src/psu/train.py` (`train_and_save`, `report_markdown`, `BASELINE`)
- Modify: `tests/conftest.py` (`synthetic_features`: add a `slate` column)
- Test: `tests/test_game_predict.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: feature frames with `season_type` and `slate` columns, which Task 3's `assemble_features` and `GAME_COLUMNS` already provide.
- Produces, in `psu.models.game_predict`:
  - `PHASES = ("early", "mid", "post")`, `EARLY_SLATES = 4`, `MIN_PHASE_N = 30`
  - `phase_of(season_type, slate) -> np.ndarray` of phase strings. Postseason is `post`; a regular-season slate of 4 or less is `early`; anything else is `mid`.
  - `GameModel.sigma_by_phase: dict[str, float] | None = None` (new dataclass field, last)
  - `GameModel.row_sigma(df) -> np.ndarray`. `GameModel.win_prob(df)` uses per-row sigma.
  - `oos_residuals(played, seasons, kind) -> pd.DataFrame[season, phase, residual]`
  - `phase_sigmas(residuals, fallback: float, min_n: int = MIN_PHASE_N) -> dict[str, float]`
  - `reliability(prob, won, bins: int = 10) -> pd.DataFrame[bin, n, mean_pred, actual]`
  - `ece(table) -> float | None`
- New keys in the `backtest()` report: `"sigma_by_phase"`, `report["test"]["early"]`, and `"calibration": {"model": {"ece", "bins"}, "vegas": {"ece", "bins"}}`. `bins` holds JSON-safe records.
- In `psu.train`: `BASELINE` dict. The saved `GameModel` carries `sigma_by_phase`.

- [ ] **Step 1: Write the failing tests**

In `tests/conftest.py` `synthetic_features`, add `"slate": 1 + i % 15,` next to `"week": 1 + i % 15,`.

In `tests/test_game_predict.py`, add:

```python
import pandas as pd


def test_phase_of():
    phases = gp.phase_of(["regular", "regular", "regular", "postseason"], [1, 4, 5, 16])
    assert list(phases) == ["early", "early", "mid", "post"]


def test_phase_sigmas_use_rms_and_fall_back_below_min_n():
    residuals = pd.DataFrame(
        {"season": 2024, "phase": ["early"] * 40 + ["mid"] * 40 + ["post"] * 5, "residual": [20.0] * 40 + [10.0] * 45}
    )
    sig = gp.phase_sigmas(residuals, fallback=99.0)
    assert sig["early"] == pytest.approx(20.0) and sig["mid"] == pytest.approx(10.0)
    pooled = np.sqrt((40 * 400 + 45 * 100) / 85)
    assert sig["post"] == pytest.approx(pooled)  # only 5 postseason residuals
    empty = pd.DataFrame(columns=["season", "phase", "residual"])
    assert gp.phase_sigmas(empty, fallback=15.0) == {"early": 15.0, "mid": 15.0, "post": 15.0}


def test_oos_residuals_never_train_on_the_season_they_score(monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind):
        seen.append(set(train["season"]))
        return real_fit(train, kind)

    monkeypatch.setattr(gp, "fit", spy)
    f = synthetic_features()
    played = f[f["margin"].notna()]
    res = gp.oos_residuals(played, [2023, 2024, 2025], "linear")
    assert seen == [{2023}, {2023, 2024}]
    assert set(res["season"]) == {2024, 2025}
    assert set(res["phase"]) <= {"early", "mid"}


def test_reliability_bins_and_ece():
    table = gp.reliability([0.05, 0.15, 0.95, 0.99, 1.0], [0, 1, 1, 1, 0])
    assert list(table.columns) == ["bin", "n", "mean_pred", "actual"]
    assert list(table["bin"]) == [0.0, 0.1, 0.9] and list(table["n"]) == [1, 1, 3]
    assert table.loc[2, "mean_pred"] == pytest.approx((0.95 + 0.99 + 1.0) / 3)
    assert table.loc[2, "actual"] == pytest.approx(2 / 3)
    expected = (1 * 0.05 + 1 * 0.85 + 3 * abs((0.95 + 0.99 + 1.0) / 3 - 2 / 3)) / 5
    assert gp.ece(table) == pytest.approx(expected)
    assert gp.ece(gp.reliability([], [])) is None


def test_model_win_prob_uses_phase_sigma():
    f = synthetic_features()
    model = gp.fit(f[f["season"].isin([2023, 2024])], "linear")
    rows = f[f["season"] == 2025].head(20)
    base = model.win_prob(rows)
    model.sigma_by_phase = {"early": 1e6, "mid": model.sigma, "post": model.sigma}
    wide = model.win_prob(rows)
    early = gp.phase_of(rows["season_type"], rows["slate"]) == "early"
    assert np.allclose(wide[early], 0.5, atol=1e-3)  # a huge sigma makes early games coin flips
    assert np.allclose(wide[~early], base[~early])
```

Also extend `test_backtest_folds_are_time_ordered`:
- Change the loop to `for block in ("all", "early", "Penn State"):`.
- Keep `assert seen[-1] == {2023, 2024}`; the test-season model is still fitted last.
- Append:

```python
    assert set(report["sigma_by_phase"]) == {"early", "mid", "post"}
    for source in ("model", "vegas"):
        cal = report["calibration"][source]
        assert 0 <= cal["ece"] <= 1 and sum(b["n"] for b in cal["bins"]) > 0
    json.dumps(report)  # the report is written as JSON
```

Add `import json` at the top of the file if it is missing.

In `tests/test_train.py`, add:

```python
def test_saved_model_and_markdown_carry_phase_sigmas(tmp_path):
    con = connect(":memory:")
    report = train_and_save(con, synthetic_features(), current_season=2026, out_dir=tmp_path)
    model = joblib.load(tmp_path / "models" / "game_model.joblib")
    assert model.sigma_by_phase == report["sigma_by_phase"]
    text = report_markdown(report)
    assert "by phase" in text and "## Calibration" in text and "Before the model upgrade" in text
    assert "| early |" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_game_predict.py tests/test_train.py -q`
Expected: FAIL because `phase_of`, `phase_sigmas`, `oos_residuals`, `reliability` and `ece` do not exist yet.

- [ ] **Step 3: Implement**

In `src/psu/models/game_predict.py`:

```python
PHASES = ("early", "mid", "post")
EARLY_SLATES = 4  # slates 1-4 lean hardest on the preseason prior
MIN_PHASE_N = 30


def phase_of(season_type, slate) -> np.ndarray:
    season_type = np.asarray(season_type, dtype=object)
    slate = np.asarray(slate, dtype=float)
    return np.where(season_type == "postseason", "post", np.where(slate <= EARLY_SLATES, "early", "mid"))
```

Change `GameModel` to:

```python
@dataclass
class GameModel:
    kind: str
    pipeline: Pipeline
    sigma: float
    sigma_by_phase: dict[str, float] | None = None

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[FEATURES])

    def row_sigma(self, df: pd.DataFrame) -> np.ndarray:
        if not self.sigma_by_phase or not {"season_type", "slate"} <= set(df.columns):
            return np.full(len(df), self.sigma, dtype=float)
        phases = phase_of(df["season_type"], df["slate"])
        return np.array([self.sigma_by_phase.get(p, self.sigma) for p in phases], dtype=float)

    def win_prob(self, df: pd.DataFrame) -> np.ndarray:
        return win_prob(self.predict_margin(df), self.row_sigma(df))
```

Add the calibration helpers after `scores`:

```python
def oos_residuals(played: pd.DataFrame, seasons: list[int], kind: str) -> pd.DataFrame:
    """Walk-forward residuals: each season after the first, predicted by a model fitted on earlier seasons only."""
    frames = []
    for season in seasons[1:]:
        train = played[played["season"].isin([s for s in seasons if s < season])]
        scored = played[played["season"] == season]
        if train.empty or scored.empty:
            continue
        model = fit(train, kind)
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "phase": phase_of(scored["season_type"], scored["slate"]),
                    "residual": scored["margin"].to_numpy(float) - model.predict_margin(scored),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["season", "phase", "residual"])


def phase_sigmas(residuals: pd.DataFrame, fallback: float, min_n: int = MIN_PHASE_N) -> dict[str, float]:
    """RMS residual per phase; phases with fewer than min_n residuals use the pooled RMS (or fallback if none)."""
    r = residuals["residual"].astype(float).to_numpy()
    pooled = float(np.sqrt(np.mean(r**2))) if len(r) else float(fallback)
    out = {}
    for phase in PHASES:
        rp = r[(residuals["phase"] == phase).to_numpy()]
        out[phase] = float(np.sqrt(np.mean(rp**2))) if len(rp) >= min_n else pooled
    return out


def reliability(prob, won, bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins (labelled by lower edge): count, mean predicted and actual home-win rate."""
    prob = np.asarray(prob, dtype=float)
    won = np.asarray(won, dtype=float)
    edge = np.minimum((prob * bins).astype(int), bins - 1) / bins
    df = pd.DataFrame({"bin": edge, "prob": prob, "won": won})
    table = df.groupby("bin").agg(n=("prob", "size"), mean_pred=("prob", "mean"), actual=("won", "mean"))
    return table.reset_index()[["bin", "n", "mean_pred", "actual"]]


def ece(table: pd.DataFrame) -> float | None:
    """Expected calibration error: bin-size-weighted mean |mean predicted - actual|."""
    n = table["n"].sum()
    if n == 0:
        return None
    return float((table["n"] * (table["mean_pred"] - table["actual"]).abs()).sum() / n)


def _records(table: pd.DataFrame) -> list[dict]:
    return [
        {"bin": float(r.bin), "n": int(r.n), "mean_pred": float(r.mean_pred), "actual": float(r.actual)}
        for r in table.itertuples()
    ]
```

In `backtest`, change the section from `kind = min(...)` to the end so that residuals are computed **before** the test-season model is fitted. That keeps the test-season fit as the last `fit` call.

```python
    kind = min(MODEL_KINDS, key=lambda k: validation[k]["mae"])
    residuals = oos_residuals(played, seasons, kind)

    train, test = split(test_season)
    model = fit(train, kind)
    model.sigma_by_phase = phase_sigmas(residuals[residuals["season"] < test_season], model.sigma)
    vegas_sigma = _vegas_sigma(train)

    def block(games: pd.DataFrame) -> dict:
        lined = games[games["vegas_margin"].notna()]
        return {
            "model": _score_model(model, games),
            "model_on_lined_games": _score_model(model, lined),
            "vegas": scores(lined["margin"], lined["vegas_margin"], win_prob(lined["vegas_margin"], vegas_sigma)),
        }

    def calibration(prob, won) -> dict:
        table = reliability(prob, won)
        return {"ece": ece(table), "bins": _records(table)}

    is_team = test["home_team"].eq(team) | test["away_team"].eq(team)
    is_early = phase_of(test["season_type"], test["slate"]) == "early"
    lined = test[test["vegas_margin"].notna()]
    won = (lined["margin"] > 0).astype(float).to_numpy()
    return {
        "model_kind": kind,
        "validation_season": validate_season,
        "validation": validation,
        "test_season": test_season,
        "train_seasons": [s for s in seasons if s < test_season],
        "sigma": model.sigma,
        "sigma_by_phase": phase_sigmas(residuals, model.sigma),
        "vegas_sigma": vegas_sigma,
        "test": {"all": block(test), "early": block(test[is_early]), team: block(test[is_team])},
        "calibration": {
            "model": calibration(model.win_prob(lined), won),
            "vegas": calibration(win_prob(lined["vegas_margin"], vegas_sigma), won),
        },
    }
```

In `src/psu/train.py`:

- Add below `PREDICTION_COLUMNS`:

```python
# 2025 test-season scores before the model upgrade (preseason priors, phase sigmas), kept to show the change.
BASELINE = {"model_mae": 12.52, "model_brier": 0.185, "vegas_mae": 11.82, "vegas_brier": 0.175}
```

- In `train_and_save`, right after `model = gp.fit(train, report["model_kind"])`:

```python
    model.sigma_by_phase = report["sigma_by_phase"]
```

- In `report_markdown`, replace the `Win probability = ...` line with the following, using `.get` so older report dicts still render:

```python
    phases = report.get("sigma_by_phase")
    sigma_text = (
        "by phase: " + ", ".join(f"{p} {phases[p]:.1f}" for p in ("early", "mid", "post"))
        if phases
        else f"{report['sigma']:.1f}"
    )
```

and use:

```python
        f"Win probability = NormalCDF(margin / sigma), sigma {sigma_text}; Vegas uses sigma {report['vegas_sigma']:.1f}.",
```

After the score table rows, append:

```python
    lines += [
        "",
        f"Before the model upgrade ({report['test_season']} test, all games): model MAE {BASELINE['model_mae']:.2f}, "
        f"Brier {BASELINE['model_brier']:.3f}; Vegas MAE {BASELINE['vegas_mae']:.2f}, "
        f"Brier {BASELINE['vegas_brier']:.3f}.",
    ]
    cal = report.get("calibration")
    if cal:
        lines += [
            "",
            "## Calibration (test season, games with a Vegas line)",
            "",
            f"Expected calibration error: model {_fmt(cal['model']['ece'], 3)}, Vegas {_fmt(cal['vegas']['ece'], 3)}.",
            "",
            "| Source | Bin | N | Mean predicted | Actual |",
            "|---|---|---|---|---|",
        ]
        for source in ("model", "vegas"):
            lines += [
                f"| {source} | {b['bin']:.1f} | {b['n']} | {b['mean_pred']:.3f} | {b['actual']:.3f} |"
                for b in cal[source]["bins"]
            ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass. If an existing `test_train.py` or `test_cli.py` assertion checks the old `Win probability = NormalCDF(margin / 16.3)` wording, update it to the new wording and mention that in the report.

- [ ] **Step 5: Commit**

```bash
git add src/psu/models/game_predict.py src/psu/train.py tests/conftest.py tests/test_game_predict.py tests/test_train.py
git commit -m "feat(model): phase-specific sigmas from walk-forward residuals; reliability and ECE in the report"
```

---

### Task 5: Simulator uses phase sigmas and writes `sim_history`

**Files:**
- Modify: `src/psu/sim/season.py` (`draw_margins`: sigma may be per game)
- Modify: `src/psu/simulate.py`
  - `load_sigma` becomes `load_sigmas`.
  - `run_simulation` accepts a phase dict.
  - `SimResult.as_of_slate`, `current_slate`, `history_rows` and `write_history` are added.
  - `write_results` also writes history.
- Modify: `src/psu/cli.py` (`_simulate` uses `load_sigmas`)
- Modify: `src/psu/dashboard/common.py` (`SOURCES["sim_history"] = "psu simulate"`)
- Test: `tests/test_sim_season.py`, `tests/test_simulate.py`, `tests/test_simulate_io.py`

**Interfaces:**
- Consumes (Task 4): `PHASES` and `phase_of` from `psu.models.game_predict`.
- Produces, in `psu.simulate`:
  - `load_sigmas(out_dir) -> dict[str, float]` (keys `early`, `mid`, `post`)
  - `run_simulation(..., sigma: float | Mapping[str, float], ...)`
  - `SimResult.as_of_slate: int = 1` (new last field)
  - `current_slate(games: pd.DataFrame, season: int) -> int`
  - `HISTORY_COLUMNS: list[str]`
  - `history_rows(results: list[SimResult], *, backfilled: bool, run_at: pd.Timestamp | None = None) -> pd.DataFrame`
  - `write_history(con, rows: pd.DataFrame) -> None`, which upserts on `(season, as_of_slate, team)`
- The `sim_history` table has columns `season INTEGER, as_of_slate INTEGER, team VARCHAR, run_at TIMESTAMP, mean_wins, p_10_plus, p_title_game, p_conf_champ, p_cfp DOUBLE, backfilled BOOLEAN`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_sim_season.py`, add:

```python
def test_draw_margins_accepts_one_sigma_per_game_and_matches_scalar_when_equal():
    strengths = np.zeros((4000, 3))
    pred, home, away = np.zeros(2), np.array([0, 0]), np.array([1, 2])
    a = draw_margins(pred, home, away, strengths, sigma=SIGMA, tau=0.0, rng=np.random.default_rng(1))
    b = draw_margins(pred, home, away, strengths, sigma=np.array([SIGMA, SIGMA]), tau=0.0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)  # same seed, same draws
    c = draw_margins(pred, home, away, strengths, sigma=np.array([1.0, 30.0]), tau=0.0, rng=np.random.default_rng(1))
    assert c[:, 0].std() == pytest.approx(1.0, rel=0.1) and c[:, 1].std() == pytest.approx(30.0, rel=0.1)
```

In `tests/test_simulate.py`, add:

```python
def test_phase_sigmas_equal_to_a_scalar_give_identical_results():
    a = run(sigma=SIGMA, seed=2)
    b = run(sigma={"early": SIGMA, "mid": SIGMA, "post": SIGMA}, seed=2)
    pd.testing.assert_frame_equal(a.win_totals, b.win_totals)
    pd.testing.assert_frame_equal(a.conference, b.conference)


def test_bad_tau_for_any_phase_fails_fast():
    with pytest.raises(ValueError, match="tau"):
        run(sigma={"early": 3.0, "mid": SIGMA, "post": SIGMA}, tau=5.0)


def test_as_of_slate_is_the_first_unfinished_week():
    assert run().as_of_slate == 2  # synthetic league: week 1 played, week 2 onward open
```

In `tests/test_simulate_io.py`:
- Replace the `load_sigma` import with `load_sigmas`.
- Add `HISTORY_COLUMNS`, `history_rows` and `write_history` to the `psu.simulate` import.
- Update the three existing `load_sigma` tests:
  - `load_sigma(tmp_path)` becomes `load_sigmas(tmp_path)`.
  - The value check `== 16.25` becomes `== {"early": 16.25, "mid": 16.25, "post": 16.25}`.

Then add:

```python
def test_load_sigmas_falls_back_for_old_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "game_model.json").write_text(
        '{"sigma": 16.0, "sigma_by_phase": {"early": 18.5, "mid": 15.0}}', encoding="utf-8"
    )
    assert load_sigmas(tmp_path) == {"early": 18.5, "mid": 15.0, "post": 16.0}


def test_write_history_replaces_the_same_week_and_keeps_others():
    con = connect(":memory:")
    games, upcoming = synthetic_league()
    first = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=100, seed=1)
    second = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=100, seed=2)
    stamp = pd.Timestamp("2026-09-20 12:00")
    write_history(con, history_rows([first], backfilled=False, run_at=stamp))
    write_history(con, history_rows([second], backfilled=False, run_at=stamp + pd.Timedelta(days=1)))
    other_week = history_rows([first], backfilled=True, run_at=stamp).assign(as_of_slate=1)
    write_history(con, other_week)
    rows = con.execute("SELECT * FROM sim_history ORDER BY as_of_slate").df()
    assert list(rows.columns) == HISTORY_COLUMNS
    assert list(rows["as_of_slate"]) == [1, 2]  # one row per week, not one per run
    assert rows.loc[1, "mean_wins"] == pytest.approx(second.mean_wins)
    assert list(rows["backfilled"]) == [True, False]


def test_write_results_also_records_history(tmp_path):
    con = connect(":memory:")
    games, upcoming = synthetic_league()
    result = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=100)
    write_results(con, result, tmp_path)
    write_results(con, result, tmp_path)
    assert con.execute("SELECT count(*), min(as_of_slate), bool_and(NOT backfilled) FROM sim_history").fetchone() == (
        1,
        2,
        True,
    )
```

Use the imports the file already has, such as `connect`, `synthetic_league`, `run_simulation` and `write_results`. Add any that are missing, plus `import pandas as pd`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_sim_season.py tests/test_simulate.py tests/test_simulate_io.py -q`
Expected: FAIL. `load_sigmas`, `write_history`, `history_rows` and `as_of_slate` do not exist yet, and array sigma is rejected.

- [ ] **Step 3: Implement**

`src/psu/sim/season.py`: replace `draw_margins` with:

```python
def draw_margins(
    pred_margin, home_idx, away_idx, strengths: np.ndarray, *, sigma, tau: float, rng: np.random.Generator
) -> np.ndarray:
    """Home-minus-away margins, shape (n_sims, n_games): every simulation plays every game.

    sigma is one value or one per game; equal values give the same draws as a single scalar.
    """
    pred = np.asarray(pred_margin, dtype=float)
    per_game = np.broadcast_to(np.asarray(sigma, dtype=float), pred.shape)
    s = np.array([game_noise_sd(float(v), tau) for v in per_game.ravel()]).reshape(pred.shape)
    noise = rng.normal(0.0, s, size=(strengths.shape[0], pred.size))
    return pred + strengths[:, home_idx] - strengths[:, away_idx] + noise
```

`src/psu/simulate.py`:

- Imports: `from collections.abc import Mapping`, `from psu.models.game_predict import PHASES, phase_of`.
- Add `as_of_slate: int = 1` as the last field of `SimResult`.
- Add near `makes_cfp`:

```python
def current_slate(games: pd.DataFrame, season: int) -> int:
    """First regular-season week with an unfinished game (one past the last week once all are played)."""
    regular = games[(games["season"] == season) & (games["season_type"] == "regular")]
    open_weeks = regular.loc[~regular["completed"].fillna(False).astype(bool), "week"]
    if len(open_weeks):
        return int(open_weeks.min())
    return int(regular["week"].max()) + 1 if len(regular) else 1
```

- In `run_simulation`:
  - Change the signature to `sigma: float | Mapping[str, float],`.
  - Replace the line `game_noise_sd(sigma, tau)  # fail fast ...` with:

```python
    sigmas = {p: float(sigma[p]) for p in PHASES} if isinstance(sigma, Mapping) else dict.fromkeys(PHASES, float(sigma))
    for value in sigmas.values():
        game_noise_sd(value, tau)  # fail fast on a bad tau, before any draw
```

  - In the `if predicted.any():` block, pass per-game sigmas. Regular-season week equals slate.

```python
        rows = relevant.loc[predicted]
        game_sigma = np.array([sigmas[p] for p in phase_of(rows["season_type"], rows["week"])])
        margins = draw_margins(
            rows["pred_margin"].to_numpy(float),
            home[predicted],
            away[predicted],
            strengths,
            sigma=game_sigma,
            tau=tau,
            rng=rng,
        )
```

  - In the title-game `draw_matchups(...)` call, use `sigma=sigmas["mid"],`.
  - In the `return SimResult(...)`, add `as_of_slate=current_slate(games, season),`.

- Replace `load_sigma` with:

```python
def load_sigmas(out_dir: Path) -> dict[str, float]:
    """Per-phase win-probability sigmas from the model report; older reports fall back to the pooled sigma."""
    path = Path(out_dir) / "reports" / "game_model.json"
    if not path.exists():
        raise MissingModel(f"{path} not found; run `psu train` first")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        pooled = float(report["sigma"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise MissingModel(f"{path} has no usable sigma; run `psu train` first") from e
    by_phase = report.get("sigma_by_phase") or {}
    return {p: float(by_phase.get(p, pooled)) for p in PHASES}
```

- In `simulate_season`, change the parameter annotation to `sigma: float | Mapping[str, float],`.
- Add after `SUMMARY_COLUMNS`:

```python
HISTORY_COLUMNS = [
    "season",
    "as_of_slate",
    "team",
    "run_at",
    "mean_wins",
    "p_10_plus",
    "p_title_game",
    "p_conf_champ",
    "p_cfp",
    "backfilled",
]
_HISTORY_DDL = (
    "CREATE TABLE IF NOT EXISTS sim_history (season INTEGER, as_of_slate INTEGER, team VARCHAR, run_at TIMESTAMP, "
    "mean_wins DOUBLE, p_10_plus DOUBLE, p_title_game DOUBLE, p_conf_champ DOUBLE, p_cfp DOUBLE, backfilled BOOLEAN)"
)


def history_rows(results: list[SimResult], *, backfilled: bool, run_at: pd.Timestamp | None = None) -> pd.DataFrame:
    run_at = pd.Timestamp.now().floor("s") if run_at is None else run_at
    return pd.DataFrame(
        [
            {
                "season": r.season,
                "as_of_slate": r.as_of_slate,
                "team": r.team,
                "run_at": run_at,
                "mean_wins": r.mean_wins,
                "p_10_plus": r.p_10_plus,
                "p_title_game": r.p_title_game,
                "p_conf_champ": r.p_conf_champ,
                "p_cfp": r.p_cfp,
                "backfilled": backfilled,
            }
            for r in results
        ],
        columns=HISTORY_COLUMNS,
    )


def write_history(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> None:
    """Upsert on (season, as_of_slate, team): a re-run of the same week replaces that week's row."""
    if rows.empty:
        return
    con.execute(_HISTORY_DDL)
    con.register("_history", rows[HISTORY_COLUMNS])
    try:
        con.execute(
            "DELETE FROM sim_history WHERE EXISTS (SELECT 1 FROM _history n WHERE n.season = sim_history.season "
            "AND n.as_of_slate = sim_history.as_of_slate AND n.team = sim_history.team)"
        )
        con.execute(f"INSERT INTO sim_history SELECT {', '.join(HISTORY_COLUMNS)} FROM _history")
    finally:
        con.unregister("_history")
```

- In `write_results`, inside the `try:` after the three `_write` calls, add:

```python
        write_history(con, history_rows([result], backfilled=False))
```

`src/psu/cli.py`: change the import to `load_sigmas`, and in `_simulate` use `sigma = load_sigmas(settings.db_path.parent)`.

`src/psu/dashboard/common.py`: add `"sim_history": "psu simulate",` to `SOURCES`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass. `test_cli.py` writes `{"sigma": 16.0}`, which is an old-style report, so it exercises the fallback.

- [ ] **Step 5: Commit**

```bash
git add src/psu/sim/season.py src/psu/simulate.py src/psu/cli.py src/psu/dashboard/common.py tests/test_sim_season.py tests/test_simulate.py tests/test_simulate_io.py
git commit -m "feat(sim): sample with phase sigmas and record each run in sim_history"
```

---

### Task 6: `psu simulate --backfill`

**Files:**
- Modify: `src/psu/features.py` (add `frozen_ratings`)
- Create: `src/psu/backfill.py`
- Modify: `src/psu/cli.py` (`--backfill` flag, `_backfill`)
- Test: `tests/test_features.py`, `tests/test_backfill.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes:
  - Task 3: `rolling_ratings(..., returning=)`, `assemble_features`, `load_feature_inputs`, `FeatureInputs`.
  - Task 5: `run_simulation`, `load_sigmas`, `current_slate`, `history_rows`, `write_history`, `MissingModel`, `SimResult`, and `simulate.load_inputs(con, season)`, which returns `(games, predictions)`.
- Produces:
  - `features.frozen_ratings(ratings: pd.DataFrame, season: int, as_of_slate: int) -> pd.DataFrame`
  - `backfill.replay_games(games: pd.DataFrame, as_of_slate: int) -> pd.DataFrame`
  - `backfill.simulate_as_of(games, predict: Callable[[int], pd.DataFrame], *, season, sigma, team, n_sims, seed, tau) -> list[SimResult]`
  - `backfill.backfill_history(con, *, season, out_dir, alpha, shrink_plays, team, n_sims, seed, tau) -> pd.DataFrame` (the rows written)
  - CLI: `psu simulate --backfill`

- [ ] **Step 1: Write the failing tests**

In `tests/test_features.py`, add `frozen_ratings` to the `psu.features` import and add:

```python
def test_frozen_ratings_never_use_later_slates():
    ratings = pd.DataFrame(
        {
            "season": 2024,
            "slate": [1, 1, 2, 3, 3, 4],
            "team": ["A", "B", "A", "A", "C", "A"],
            "off_epa": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "def_epa": 0.0,
            "off_sr": 0.0,
            "def_sr": 0.0,
        }
    )
    other = ratings.assign(season=2023, off_epa=9.0)
    frozen = frozen_ratings(pd.concat([ratings, other], ignore_index=True), 2024, 2).set_index(
        ["season", "slate", "team"]
    )
    assert frozen.loc[(2024, 1, "A"), "off_epa"] == 0.1  # before the as-of slate: unchanged
    assert frozen.loc[(2024, 2, "A"), "off_epa"] == 0.3  # the as-of slate itself
    assert frozen.loc[(2024, 3, "A"), "off_epa"] == 0.3  # later slates: frozen at slate 2
    assert frozen.loc[(2024, 4, "A"), "off_epa"] == 0.3
    assert frozen.loc[(2024, 3, "C"), "off_epa"] == 0.5  # C's first game is after slate 2: its row is prior-only
    assert (frozen.xs(2023, level="season")["off_epa"] == 9.0).all()  # other seasons untouched
```

Create `tests/test_backfill.py`:

```python
import pandas as pd
from conftest import synthetic_league

from psu.backfill import replay_games, simulate_as_of


def all_predictions(games):
    """A prediction for every synthetic-league game, including the two already played."""
    fixed = {1: 5.0, 2: 3.0, 9: 30.0}
    _, upcoming = synthetic_league()
    played = pd.DataFrame(
        [
            {"game_id": g.id, "home_team": g.home_team, "away_team": g.away_team, "neutral_site": False,
             "pred_margin": fixed[g.id]}
            for g in games.itertuples()
            if g.id in fixed
        ]
    )
    return pd.concat([upcoming, played], ignore_index=True)


def test_replay_games_reopens_the_as_of_week_and_later():
    games, _ = synthetic_league()
    replay = replay_games(games, 1)
    assert not replay["completed"].any()
    assert replay["home_points"].isna().all()
    assert replay_games(games, 2)["completed"].tolist() == games["completed"].tolist()  # week 1 stays played


def test_simulate_as_of_replays_each_finished_week():
    games, _ = synthetic_league()
    seen = []

    def predict(n):
        seen.append(n)
        return all_predictions(games)

    results = simulate_as_of(games, predict, season=2026, sigma=16.0, team="B", n_sims=500, seed=0, tau=0.1)
    assert seen == [1]  # week 1 is the only finished week in the synthetic league
    assert [r.as_of_slate for r in results] == [1]
    # As of week 1, B's loss to A hasn't happened yet, so B can still win all three of its games.
    totals = results[0].win_totals
    assert totals.loc[totals["wins"] == totals["wins"].max(), "prob"].item() > 0


def test_backfill_before_any_games_writes_nothing():
    games, _ = synthetic_league()
    unplayed = replay_games(games, 1)
    results = simulate_as_of(
        unplayed, lambda n: all_predictions(games), season=2026, sigma=16.0, team="A", n_sims=50, seed=0, tau=0.1
    )
    assert results == []
```

In `tests/test_cli.py`, add a test that `psu simulate --backfill` with no model returns 2 and prints an error. Put it after `test_simulate_requires_trained_model`, and reuse that test's `settings` fixture and pattern:

```python
def test_simulate_backfill_requires_trained_model(settings, capsys):
    assert cli.main(["simulate", "--backfill"]) == 2
    assert "psu train" in capsys.readouterr().err
```

Also add `--backfill` to the parser test (around line 178): `assert s.backfill is False`, and `assert parser.parse_args(["simulate", "--backfill"]).backfill is True`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_features.py tests/test_backfill.py tests/test_cli.py -q`
Expected: FAIL because `frozen_ratings` and `psu.backfill` do not exist and `--backfill` is not recognised.

- [ ] **Step 3: Implement**

`src/psu/features.py`: add after `rolling_ratings`:

```python
def frozen_ratings(ratings: pd.DataFrame, season: int, as_of_slate: int) -> pd.DataFrame:
    """Ratings as they stood at the start of as_of_slate, copied onto every later slate of `season`.

    Each team takes its row from the latest slate at or before as_of_slate. A team whose first game comes later
    uses that first row, which holds only its preseason prior because it had played no games. Other seasons and
    earlier slates are unchanged.
    """
    this = ratings[ratings["season"] == season].sort_values("slate")
    known = this[this["slate"] <= as_of_slate].drop_duplicates("team", keep="last").set_index("team")
    first = this.drop_duplicates("team", keep="first").set_index("team")
    snapshot = pd.concat([known, first[~first.index.isin(known.index)]])[RATING_COLUMNS]
    out = ratings.copy()
    later = (out["season"] == season) & (out["slate"] >= as_of_slate)
    out.loc[later, RATING_COLUMNS] = out.loc[later, ["team"]].join(snapshot, on="team")[RATING_COLUMNS].to_numpy()
    return out
```

Create `src/psu/backfill.py`:

```python
"""Replay earlier weeks of a season: simulate as of each finished week, with team ratings frozen at that week.

The ratings behind every replayed prediction use only plays from before the as-of week. The model's coefficients
were fitted with this season's finished games in view, so replays are close to, but not exactly, what the model
would have said at the time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from psu.config import TEAM
from psu.features import assemble_features, frozen_ratings, rolling_ratings
from psu.simulate import (
    MissingModel,
    SimResult,
    current_slate,
    history_rows,
    load_inputs,
    load_sigmas,
    run_simulation,
    write_history,
)
from psu.train import load_feature_inputs

PREDICTION_COLUMNS = ["game_id", "home_team", "away_team", "neutral_site", "pred_margin"]


def replay_games(games: pd.DataFrame, as_of_slate: int) -> pd.DataFrame:
    """The season as it stood before as_of_slate: that week and everything after it is unplayed again."""
    replay = games.copy()
    later = (replay["season_type"] != "regular") | (replay["week"] >= as_of_slate)
    replay.loc[later, "completed"] = False
    replay[["home_points", "away_points"]] = replay[["home_points", "away_points"]].astype("Float64")
    replay.loc[later, ["home_points", "away_points"]] = pd.NA
    return replay


def simulate_as_of(
    games: pd.DataFrame,
    predict: Callable[[int], pd.DataFrame],
    *,
    season: int,
    sigma: float | Mapping[str, float],
    team: str,
    n_sims: int,
    seed: int,
    tau: float,
) -> list[SimResult]:
    """One simulation per finished week N (1 .. current - 1), from predict(N) and the games as of N."""
    return [
        run_simulation(
            replay_games(games, n), predict(n), season=season, sigma=sigma, team=team, n_sims=n_sims, seed=seed, tau=tau
        )
        for n in range(1, current_slate(games, season))
    ]


def backfill_history(
    con: duckdb.DuckDBPyConnection,
    *,
    season: int,
    out_dir: Path,
    alpha: float,
    shrink_plays: int,
    team: str = TEAM,
    n_sims: int,
    seed: int,
    tau: float,
) -> pd.DataFrame:
    model_path = Path(out_dir) / "models" / "game_model.joblib"
    if not model_path.exists():
        raise MissingModel(f"{model_path} not found; run `psu train` first")
    sigmas = load_sigmas(out_dir)
    model = joblib.load(model_path)
    games, _ = load_inputs(con, season)
    inputs = load_feature_inputs(con)
    ratings = rolling_ratings(
        inputs.enriched, inputs.games, alpha=alpha, shrink_plays=shrink_plays, returning=inputs.returning
    )
    season_games = inputs.games[inputs.games["season"] == season]

    def predict(n: int) -> pd.DataFrame:
        feats = assemble_features(
            frozen_ratings(ratings, season, n), season_games, inputs.lines, inputs.sp, inputs.talent, inputs.returning
        )
        feats = feats[feats["slate"] >= n]
        return feats.assign(pred_margin=model.predict_margin(feats))[PREDICTION_COLUMNS]

    results = simulate_as_of(
        games, predict, season=season, sigma=sigmas, team=team, n_sims=n_sims, seed=seed, tau=tau
    )
    rows = history_rows(results, backfilled=True)
    con.execute("BEGIN TRANSACTION")
    try:
        write_history(con, rows)
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
    return rows
```

`src/psu/cli.py`:
- Import `from psu.backfill import backfill_history`.
- Add to the `simulate` parser:

```python
    sim.add_argument(
        "--backfill", action="store_true", help="Also replay each finished week of the season into sim_history"
    )
```

- Add `_backfill` after `_simulate`:

```python
def _backfill(settings: config.Settings, *, team: str, n_sims: int, seed: int, tau: float) -> int:
    season = settings.current_season
    try:
        con = db.connect(settings.db_path)
        try:
            rows = backfill_history(
                con,
                season=season,
                out_dir=settings.db_path.parent,
                alpha=config.TRAIN_ALPHA,
                shrink_plays=config.SHRINK_PLAYS,
                team=team,
                n_sims=n_sims,
                seed=seed,
                tau=tau,
            )
        finally:
            con.close()
    except (MissingModel, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if rows.empty:
        print(f"backfill: no finished weeks in {season} yet; nothing to replay")
    else:
        print(f"backfill: replayed {len(rows)} week(s) of {season} into sim_history")
    return 0
```

- Change `cmd_simulate` to:

```python
def cmd_simulate(args: argparse.Namespace, settings: config.Settings) -> int:
    code = _simulate(settings, team=args.team, n_sims=args.sims, seed=args.seed, tau=args.tau)[0]
    if code or not args.backfill:
        return code
    return _backfill(settings, team=args.team, n_sims=args.sims, seed=args.seed, tau=args.tau)
```

`test_simulate_backfill_requires_trained_model` exits 2 from `_simulate` before the backfill runs, because the report file is missing. That is the intended behaviour.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/psu/features.py src/psu/backfill.py src/psu/cli.py tests/test_features.py tests/test_backfill.py tests/test_cli.py
git commit -m "feat(sim): psu simulate --backfill replays finished weeks with frozen ratings"
```

---

### Task 7: Dashboard "Odds over time" and docs

**Files:**
- Modify: `src/psu/dashboard/predictions.py` (add `sim_history`)
- Modify: `app/views/predictions.py` (chart)
- Modify: `README.md` (document returning production, phase sigmas, `--backfill`)
- Test: `tests/test_dashboard_predictions.py`

**Interfaces:**
- Consumes (Task 5): the `sim_history` table and `HISTORY_COLUMNS`.
- Produces: `psu.dashboard.predictions.sim_history(con, season: int, team: str) -> pd.DataFrame` with columns `as_of_slate, p_title_game, p_conf_champ, p_cfp, backfilled`, ordered by `as_of_slate`. It returns an empty frame with those columns when the table does not exist.

- [ ] **Step 1: Write the failing tests**

In `tests/test_dashboard_predictions.py`, add `sim_history` to the import and add:

```python
HISTORY_VIEW = ["as_of_slate", "p_title_game", "p_conf_champ", "p_cfp", "backfilled"]


def test_sim_history_is_empty_without_the_table(con):
    h = sim_history(con, 2026, "Penn State")
    assert h.empty and list(h.columns) == HISTORY_VIEW


def test_sim_history_rows_are_ordered_by_week():
    import pandas as pd

    from psu.simulate import HISTORY_COLUMNS, write_history

    c = connect(":memory:")
    base = {"team": "Penn State", "run_at": pd.Timestamp("2026-09-20"), "mean_wins": 9.0, "p_10_plus": 0.4,
            "p_title_game": 0.5, "p_conf_champ": 0.3}
    rows = pd.DataFrame(
        [
            {**base, "season": 2026, "as_of_slate": 3, "p_cfp": 0.6, "backfilled": False},
            {**base, "season": 2026, "as_of_slate": 1, "p_cfp": 0.5, "backfilled": True},
            {**base, "season": 2025, "as_of_slate": 1, "p_cfp": 0.9, "backfilled": True},
        ]
    )[HISTORY_COLUMNS]
    write_history(c, rows)
    h = sim_history(c, 2026, "Penn State")
    assert list(h.columns) == HISTORY_VIEW
    assert list(h["as_of_slate"]) == [1, 3] and list(h["p_cfp"]) == [0.5, 0.6]
```

The module `con` fixture is the seeded dashboard DB, which has no `sim_history` table. That covers the empty case.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dashboard_predictions.py -q`
Expected: FAIL with an ImportError for `sim_history`.

- [ ] **Step 3: Implement**

`src/psu/dashboard/predictions.py`: import `has_table` from `psu.dashboard.common` and add:

```python
HISTORY_VIEW_COLUMNS = ["as_of_slate", "p_title_game", "p_conf_champ", "p_cfp", "backfilled"]


def sim_history(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    """Season-outlook odds by week; empty (not an error) before the first `psu simulate` that records history."""
    if not has_table(con, "sim_history"):
        return pd.DataFrame(columns=HISTORY_VIEW_COLUMNS)
    return con.execute(
        f"SELECT {', '.join(HISTORY_VIEW_COLUMNS)} FROM sim_history WHERE season = ? AND team = ? ORDER BY as_of_slate",
        [season, team],
    ).df()
```

`app/views/predictions.py`: insert this block right before `with st.expander("Next week's FBS games"):`:

```python
st.subheader("Odds over time")
history = load_or_note("predictions.sim_history", season, TEAM)
if history is not None:
    if history.empty:
        st.info("No simulation history yet. Run `psu simulate --backfill` to replay earlier weeks.")
    else:
        labels = {"p_cfp": "CFP", "p_conf_champ": "Big Ten champ", "p_title_game": "Title game"}
        long = history.melt(id_vars=["as_of_slate"], value_vars=list(labels), var_name="odds", value_name="prob")
        long["odds"] = long["odds"].map(labels)
        st.altair_chart(
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("as_of_slate:O", title="Before week"),
                y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("odds:N", title=None),
                tooltip=["as_of_slate:O", "odds:N", alt.Tooltip("prob:Q", format=".1%")],
            ),
            use_container_width=True,
        )
        if history["backfilled"].any():
            st.caption(
                "Earlier weeks are replays: team ratings as of each week, but model coefficients fitted with "
                "this season's results in view."
            )
```

`README.md`: in the section that documents the CLI commands (next to `psu refresh`), add:
- `psu simulate --backfill`: after the normal run, replays each finished week of the current season into `sim_history` for the dashboard's "Odds over time" chart. It makes no API calls.
- Two notes in the model/pipeline section:
  - Preseason priors now regress last season's ratings using CFBD returning production (one extra API call per season, refreshed weekly).
  - Win probabilities use separate sigmas for weeks 1–4, week 5 on, and the postseason. `data/reports/game_model.md` shows reliability and ECE.

Match the README's existing heading and bullet style.

- [ ] **Step 4: Run the tests and check the dashboard**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass, including `test_dashboard_app.py`. That test renders the pages; the new block shows the "No simulation history yet" note on the seeded DB.

- [ ] **Step 5: Commit**

```bash
git add src/psu/dashboard/predictions.py app/views/predictions.py README.md tests/test_dashboard_predictions.py
git commit -m "feat(dashboard): odds-over-time chart from sim_history; document backfill and priors"
```

---

## After all tasks (orchestrator, not a subagent task)

1. With the user's OK (it costs 5 CFBD calls), run `psu ingest --seasons 2022-2026` to pull returning production.
2. Run `psu refresh --skip-ingest`, then `psu simulate --backfill`.
3. Compare `data/reports/game_model.md` with the baseline line. Put the before and after numbers (all games and early season) in the PR description.
4. Open the PR from `feat/model-upgrade` into `main`.
