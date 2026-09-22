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
