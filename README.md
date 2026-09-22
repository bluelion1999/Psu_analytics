# PSU Analytics

Penn State football analytics built on the [CollegeFootballData](https://collegefootballdata.com) API:
a cached data pipeline into DuckDB, efficiency metrics, predictive models, and a Streamlit dashboard.
The full build plan is in `docs/superpowers/specs/psu-analytics-build-plan.md`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env   # then paste your free key from https://collegefootballdata.com/key
.venv\Scripts\python -m pytest
```

## Phase 1: Data pipeline

```powershell
.venv\Scripts\psu ingest --seasons 2024        # one season (~60 API calls)
.venv\Scripts\psu ingest --seasons 2022-2026   # everything (~295 calls the first time)
.venv\Scripts\psu status                       # row counts per table
```

- Every API response is cached in `data/raw/<endpoint>/<params>.json`. Past seasons are never
  re-requested. For the current season, data less than 24 hours old is reused, weeks that haven't
  started are skipped, and a week is treated as final 3 days after it ends.
- The CFBD free tier has a monthly call limit. Each run stops before it exceeds `--max-calls`
  (default 300, or `PSU_MAX_CALLS` in `.env`). Whatever was fetched is kept, so re-running continues
  where it stopped. A full 2022–2026 pull (~295 calls) fits in one default run. To pull more
  history, lower `FIRST_SEASON` in `src/psu/config.py` (about 60 calls per extra season).
- Data lands in `data/psu.duckdb`: `games`, `plays`, `drives`, `team_game_stats`,
  `player_game_stats`, `advanced_season`, `ratings_sp`, `talent`, `recruiting`, `lines`. Game-level
  stats are stored long (one row per team/stat or player/stat). Start dates are UTC.
- Deleting `data/psu.duckdb` is safe: the next `psu ingest` rebuilds it from the cache without
  calling the API.
- Refreshing the current season costs about 5 calls per daily run plus 3 per live week (talent,
  recruiting and the calendar refresh weekly, with about 3 more calls on those days). Running once
  a day in-season uses roughly 250-350 calls a month.
- Data quirks for Phase 2:
  - `ratings_sp` includes a `team = 'nationalAverages'` row.
  - `player_game_stats` includes team-total rows with a negative `athlete_id` and
    `athlete_name = ' Team'`.
  - About a quarter of `plays` have null `ppa` (non-scrimmage plays such as kickoffs, penalties and
    timeouts).

## Phase 2: Metrics layer

```powershell
.venv\Scripts\psu build                      # no API calls; rebuilds every metric table (~30 s)
.venv\Scripts\psu build --garbage 30,20,15   # custom garbage-time margins (Q2,Q3,Q4), or --garbage off
.venv\Scripts\python -m pytest tests/test_validation.py   # Penn State vs CFBD's advanced stats
```

`psu build` reads the Phase 1 tables and writes these DuckDB tables (FBS teams only):

| Table | What's in it |
|---|---|
| `plays_enriched` | scrimmage plays with `play_class`, `success`, `explosive`, `turnover`, `garbage`, `score_state`, `quarter`, `venue` |
| `team_offense` / `team_defense` | EPA/play (overall, rush, pass), success rate, explosiveness, explosive rate, 3rd-down rate, turnover rate, per season; garbage time excluded |
| `team_splits` | the same metrics split by down, quarter, score state, venue, and opponent conference (`side`, `split`, `split_value`) |
| `team_adjusted` | opponent-adjusted EPA/play and success rate: a per-season ridge regression with offense, defense, and home-field terms (`--alpha` sets shrinkage) |
| `team_havoc` | (TFL + passes defended + INT + forced fumbles) / defensive plays, from box scores |
| `team_turnovers` | giveaways, takeaways, and margin |
| `team_red_zone` | trips inside the 20, TD rate, and points per trip, for offense and defense |

Definitions:
- **Scrimmage play:** a rush, pass, sack, or fumble/interception play with a PPA value.
- **Success:** 50% of the yards needed on 1st down, 70% on 2nd, and 100% on 3rd/4th. TDs always succeed; turnovers never do.
- **Explosiveness:** the mean EPA of successful plays.
- **Explosive play:** a rush of 12+ yards or a pass of 16+ yards.
- **Garbage time:** a margin above 38 in Q2, above 28 in Q3, or above 22 in Q4.

**How this compares with CFBD:** CFBD's advanced stats count every play that has a PPA value,
including End Period rows and punt-return or blocked-kick touchdowns charged to the defense.
Counted that way, our data reproduces CFBD's play counts and EPA/play exactly for Penn State
2022–2025. Our metrics use scrimmage plays only, so they differ slightly: up to about 2% of plays
and about 0.025 EPA. CFBD doesn't publish its explosiveness formula; ours differs by up to about
0.11 on defense.
