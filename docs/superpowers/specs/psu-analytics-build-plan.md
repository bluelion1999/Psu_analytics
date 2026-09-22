# Penn State Football Analytics Platform — Build Plan

## Goal
Build a local analytics platform for Penn State football using the CollegeFootballData (CFBD) API: a data pipeline, a local database, efficiency metrics, predictive models, and an interactive dashboard.

## Assumptions (change these if needed)

* Language: Python 3.11+
* Scope: personal/hobby project, runs locally
* Data source: CFBD API (free key from https://collegefootballdata.com/key), stored in env var `CFBD_API_KEY`
* Seasons: the last 5 seasons, 2022 through the current season (2026) (narrowed from 2014–2026 on 2026-09-22 to save API quota)
* Team of focus: Penn State, but pull all FBS data so opponent adjustments work

## Tech stack

* Data access: official `cfbd` Python package (check current docs at https://api.collegefootballdata.com for endpoint names and params before coding against them)
* Storage: DuckDB (single file, `data/psu.duckdb`)
* Analysis: pandas or polars, numpy
* Modeling: scikit-learn, xgboost
* Dashboard: Streamlit + Plotly
* Config: `.env` via python-dotenv; never commit the API key
* Tests: pytest

## Repo structure

```
psu-analytics/
├── .env.example            # CFBD_API_KEY=
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/                # cached JSON responses, one file per request
│   └── psu.duckdb
├── src/psu/
│   ├── config.py           # team name, seasons, paths
│   ├── client.py           # CFBD client with caching + rate limiting
│   ├── ingest.py           # pull + load raw data into DuckDB
│   ├── transform.py        # clean tables, derived columns
│   ├── metrics.py          # EPA, success rate, explosiveness, etc.
│   ├── models/
│   │   ├── game_predict.py
│   │   └── season_sim.py
│   └── cli.py              # `psu ingest`, `psu build`, `psu train`
├── app/
│   └── streamlit_app.py
├── notebooks/              # exploration only
└── tests/
```

## Important constraints

* The free CFBD tier has a monthly API call limit. Cache every response to `data/raw/` and never re-request something already cached (except the current season's latest week). Prefer season-level and week-level bulk requests over per-game requests.
* Make ingestion idempotent: re-running should upsert, not duplicate.
* Handle missing data (older seasons lack some fields; some endpoints only start in 2025).

## Phase 1 — Data pipeline

Pull (all FBS, each season 2022–2026):

* Games and results (scores, home/away, neutral site, conference game flag)
* Plays (play-by-play with down, distance, yard line, play type, PPA/EPA where provided)
* Drives
* Team game stats and player game stats
* Season advanced team stats
* SP+ ratings, team talent composite, recruiting rankings
* Betting lines (spreads, totals)

Load into DuckDB tables: `games`, `plays`, `drives`, `team_game_stats`, `player_game_stats`, `advanced_season`, `ratings_sp`, `talent`, `recruiting`, `lines`.

Done when: `psu ingest --seasons 2022-2026` completes, row counts are logged per table, and a re-run makes zero new API calls for completed seasons.

## Phase 2 — Metrics layer

Compute for Penn State offense and defense, and for every FBS team for comparison:

* EPA per play (overall, rush, pass)
* Success rate (standard definition: ≥50% of yards needed on 1st down, ≥70% on 2nd, 100% on 3rd/4th)
* Explosiveness (EPA on successful plays), explosive play rate
* Havoc rate (defense), turnover margin
* Red zone and 3rd-down efficiency
* Splits: by down, quarter, score state, home/away, opponent conference
* Garbage-time filter (make the thresholds configurable)
* Opponent-adjusted versions of EPA/play and success rate (e.g., ridge regression of play results on offense + defense team effects per season)

Done when: `metrics.py` functions return tidy DataFrames, unit tests cover success-rate and garbage-time logic, and PSU season totals roughly match CFBD's advanced stats.

## Phase 3 — Game prediction model

* Target: final margin (PSU points − opponent points); derive win probability from it.
* Features: opponent-adjusted efficiency (rolling, using only data available before the game), SP+ ratings, talent composite, home/neutral, rest days.
* Train on all FBS games, evaluate on PSU games and overall.
* Validation: time-based splits only (train on earlier seasons, test on later). No leakage from future games.
* Report MAE on margin, Brier score on win probability, and compare against the closing Vegas spread as a baseline.

Done when: `psu train` saves a model and a metrics report, and the report includes the Vegas comparison.

## Phase 4 — Season simulator

* Use the game model to simulate PSU's remaining schedule (10,000 runs).
* Output: distribution of final win totals, odds of 10+ wins, odds of a Big Ten title game appearance (approximate using conference standings logic), rough College Football Playoff odds.

Done when: the simulator runs in under a minute and results show in the dashboard.

## Phase 5 — Dashboard (Streamlit)

Pages:

1. Season overview — record, schedule with results and predicted lines, key metrics vs. Big Ten and national averages.
2. Efficiency trends — EPA/play and success rate by season (2022–present) and by week for the current season.
3. Game explorer — pick a game: box score, drive chart, win probability chart, top plays by EPA.
4. Players — QB, RB, and WR efficiency tables, with filters.
5. Predictions — upcoming game predictions and season simulation results.

Done when: `streamlit run app/streamlit_app.py` loads every page from DuckDB with no API calls.

## Phase 6 (optional) — Extensions

* 4th-down decision model: go-for-it vs. punt vs. field goal by expected win probability.
* Recruiting-to-production analysis.
* Scheduled weekly refresh during the season.

## Working instructions for Claude Code

* Build phase by phase, and stop for review at the end of each phase.
* Before writing API code, check the current CFBD docs and `cfbd` package version for exact endpoint and method names.
* Use a small sample (one season) while developing, then scale up.
* Write a short README section for each phase covering how to run it.
