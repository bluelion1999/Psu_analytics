# PSU Analytics

[![CI](https://github.com/bluelion1999/Psu_analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/bluelion1999/Psu_analytics/actions/workflows/ci.yml)

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
- **Garbage time:** a margin above 38 in Q2, above 28 in Q3, or above 22 in Q4. Margins and score
  states use the score before the play (the drive's starting score).
- **Opponent adjustment:** adjusted values are centred on the league's play-weighted average, so
  raw and adjusted numbers are directly comparable.
- **Red zone:** a trip is a drive that reaches the 20. Points come from the drive's result: a TD
  counts 6–8 and a FG counts 3.

**Caveat for modelling:** these are full-season numbers, postseason included. Phase 3 builds
week-by-week versions for pregame predictions so no game uses information from its own future.

**How this compares with CFBD:** CFBD's advanced stats count every play that has a PPA value,
including End Period rows and punt-return or blocked-kick touchdowns charged to the defense.
Counted that way, our data reproduces CFBD's play counts and EPA/play exactly for Penn State
2022–2025. Our metrics use scrimmage plays only, so they differ slightly: up to about 2% of plays
and about 0.025 EPA. CFBD doesn't publish its explosiveness formula; ours differs by up to about
0.11 on defense.

## Phase 3: Game prediction model

```powershell
.venv\Scripts\psu train                               # ~1 min, no API calls
.venv\Scripts\psu train --alpha 20 --shrink-plays 75  # the (tuned) defaults, spelled out
```

`psu train` predicts the home team's margin for every FBS-vs-FBS game, converts it to a win
probability, compares itself with the closing Vegas line, and writes:

- `game_predictions` (DuckDB): `pred_margin` and `home_win_prob` for every game, including
  upcoming ones, next to `vegas_margin` and the actual `margin`.
- `data/models/game_model.joblib`: the fitted model.
- `data/reports/game_model.md` and `.json`: the backtest report.

**Features** (home minus away): opponent-adjusted offensive and defensive EPA/play and success
rate as of the game's week, last season's SP+ rating, the talent composite, rest days, a
home-field flag (0 at neutral sites), and `d_returning` (the difference in returning offensive
production, from CFBD's returning production endpoint).

**No leakage:**
- A game's team ratings are fit only on plays from earlier weeks of the same season. Early in the
  season they are blended with last season's final ratings. `--shrink-plays` sets how many plays it
  takes for the current season to dominate.
- SP+ comes from the previous season, because CFBD's same-season SP+ reflects the whole season.
- Preseason priors now regress last season's ratings using CFBD returning production (one extra
  API call per season, refreshed weekly).
- Evaluation is strictly time-based. The first season in the data (2022) serves only as a prior.

**Evaluation:**
- Validation trains on 2023 and evaluates on 2024; this picks the model (linear beat XGBoost) and
  the tuning.
- Test trains on 2023–2024 and evaluates on 2025, which was never used for any choice.
- The final model is refit on every completed game from 2023 onward.

Win probability is `NormalCDF(margin / sigma)`, with sigma taken from out-of-fold residuals.
Win probabilities use separate sigmas for weeks 1–4, week 5 on, and the postseason.
`data/reports/game_model.md` shows reliability and ECE.

2025 test season:

| Games | N | Model MAE | Vegas MAE | Model Brier | Vegas Brier |
|---|---|---|---|---|---|
| All FBS vs FBS | 808 | 12.52 | 11.82 | 0.185 | 0.175 |
| Penn State | 12 | 7.84 | 12.69 | 0.194 | 0.239 |

The closing line is still about 0.7 points more accurate overall, as expected against a
market-efficient baseline. The Penn State edge is 12 games, which is too few to read into. On the
2025 test season, the model's predictions correlate 0.91 with Vegas's. Its average home-win
probability is 0.584, against an actual home-win rate of 0.595.

The `split` column in `game_predictions` says which rows are genuine pregame predictions:
- `upcoming`: not played yet. This is a true forecast.
- `in_sample`: already played, and the final model was trained on it, so the prediction flatters the
  model.
- `no_prior`: the first season, with no prior-season ratings. Never trained on.

For honest pregame numbers on past games, use the backtest report.

## Phase 4: Season simulator

```
.venv\Scripts\psu simulate                 # 10,000 seasons, seed 0, tau 5
.venv\Scripts\psu simulate --sims 50000 --seed 1 --tau 4
.venv\Scripts\psu simulate --backfill      # after the normal run, replay each finished week into sim_history
```

`psu simulate --backfill`: after the normal run, replays each finished week of the current season into
`sim_history` for the dashboard's "Odds over time" chart. It makes no API calls. Each replayed week uses
team ratings as of that week, but the model's coefficients are fitted with this season's results already
in view, so replays are close to, but not exactly, what the model would have said at the time.

Run `psu train` first. The simulator reads `game_predictions` and per-phase sigmas from
`data/reports/game_model.json`. It makes no API calls, and a full run takes about 3 seconds.

How it works:
- Games already played use their actual results. Every remaining Big Ten game, and every remaining
  Penn State game, is simulated from the model's predicted margin.
- In each simulated season, every team gets one season-long strength draw (spread `--tau` points).
  A team that runs hot does so in all its games. Game noise is sized so that each game still has the
  model's win probability.
- Big Ten standings: conference win %, then head-to-head among the tied teams (only if they all
  played each other), then a coin flip. The top two meet at a neutral site, rated from power ratings
  fitted to the model's predictions.
- The conference title game is identified by its `notes` field (containing "Big Ten Championship") and
  excluded from the regular season and standings; if it's already been played, its actual teams and
  winner are used, otherwise its two teams are fixed and the winner is simulated.
- CFP (rough): in if Big Ten champion, or 2 or fewer losses, counting a title-game loss.
- Win totals are regular season only. Ratings for the title game are fitted on every prediction of the
  season, not just the games left to play, so they stay sharp late in the season.

Outputs (replaced on each run): `sim_team_summary`, `sim_win_totals` and `sim_conference`, plus
`data/reports/season_sim.md`.

Results through 2026-09-20 (Penn State 3-0):

| Mean wins | P(10+ wins) | P(title game) | P(Big Ten champ) | P(CFP) |
|---|---|---|---|---|
| 9.47 | 51.7% | 29.2% | 10.7% | 48.0% |

The most likely finishes are 10-2 (24.7%) and 9-3 (23.1%). Ohio State is the Big Ten favourite
(34.2%), ahead of Oregon (18.9%), Indiana (12.7%) and Penn State (10.7%).

Choosing tau: team-level residuals from 2024 and 2025 imply a tau of about 3.4 to 3.9. Those
predictions are in-sample, so that understates it, and the default is 5. The headline odds barely
move with tau: from tau 3 to 7, P(10+ wins) goes from 51% to 53% and P(CFP) from 46% to 51%.

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
