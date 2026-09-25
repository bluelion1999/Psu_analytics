# Model upgrade: preseason priors, calibration, per-week sim history

Sub-project 3 of the enhancement roadmap. It builds on the engineering foundation (the `feat/foundation` branch, PR #7).

## Goal and success criteria

The goal is better early-season predictions, win probabilities you can trust, and a record of how the season outlook changes week to week. This serves both the in-season tool and portfolio depth.

Success means:
- The 2025 test MAE and Brier score improve on the current baseline (model 12.52 / 0.185; Vegas 11.82 / 0.175), both overall and for weeks 1–4. The report shows before and after.
- The report shows a reliability table in which predicted and actual win rates agree within noise, and gives the expected calibration error (ECE) for both the model and Vegas.
- The dashboard shows an "odds over time" chart for the current season.

Constraints: seasons from 2022 on; a tight CFBD free-tier quota, so no live calls in tests and at most one new call per season; refresh stays manual; never import `sklearn.impute` or `sklearn.neighbors` (Smart App Control).

Out of scope: new model families, and priors for 2022, which keeps the `no_prior` split.

## 1. Preseason priors with returning production

### Data
- A new CFBD endpoint, `returning` (`PlayersApi.get_returning_production`, `/player/returning`), is added to `cfbd_api.py` and requested once per season in `ingest.py`. Like talent and recruiting, it uses `SLOW_REFRESH`.
- New table `returning_production`, key `(season, team)` (CFBD returns `season` for this endpoint, not `year`), with columns `percent_ppa`, `percent_passing_ppa`, `percent_receiving_ppa`, `percent_rushing_ppa`, `usage` (all DOUBLE) and `conference` (VARCHAR).
- Backfilling 2022–2026 costs 5 API calls.

### Projection
A new module, `src/psu/priors.py`:

- `projection_pairs(finals, returning)` builds one row per (team, season S) with `last = final rating in S-1`, `next = final rating in S`, and `returning_pct = percent_ppa for S`. It does this for each rating column in `RATING_COLUMNS`.
- `fit_projection(pairs, column)` returns `(b0, b1)`:
  - Offense columns (`off_*`) are fitted by least squares on `next − mean_S = (b0 + b1·ret)·(last − mean_{S-1})`.
  - Defense columns fit `b0` alone, with `b1 = 0`, because CFBD returning production covers offense only.
- `project_prior(last_final, means, returning_pct, coefs)` returns
  `prior = mean + (b0 + b1·ret)·(last − mean)`, where `ret` falls back to the league-median `percent_ppa` when a team has no row.
- **No leakage:** coefficients for season S are fitted only on pairs whose `next` season is before S.
  - With fewer than 20 pairs (for example the 2023 prior, since 2022 is the first season), the fallback is `b0 = 0.6`, `b1 = 0`.
  - The factor `b0 + b1·ret` is clipped to [0, 1] when it is applied.

`features.rolling_ratings` uses the projected prior instead of the raw prior-season ratings. The blend with current-season plays (`shrink_plays`) does not change.

### Feature
`d_returning` (home − away `percent_ppa`) joins `FEATURES`. The median imputer handles missing values, as for the other features.

## 2. Calibrated win probabilities

- Win probability stays `NormalCDF(margin / sigma)`, because the simulator samples margins from the same normal model. What changes is that **sigma depends on the phase**:
  - `early` = regular-season slates 1–4
  - `mid` = regular-season slate 5 and later
  - `post` = postseason
- The phase sigmas are estimated from **out-of-sample** residuals of the walk-forward backtest:
  - For each season from 2024 on, fit on earlier seasons and predict that season.
  - Pool the residuals by phase; the sigma is their RMS.
  - A phase with fewer than 30 residuals falls back to the pooled sigma.
- `GameModel` stores `sigma_by_phase: dict[str, float]`, and `win_prob` looks up each row's phase. `report["sigma"]` stays as the pooled value for backward compatibility, and `report["sigma_by_phase"]` is added.
- New in `game_predict.py`:
  - `reliability(prob, won, bins=10)` returns a DataFrame with columns bin, n, mean_pred and actual.
  - `ece(...)` returns the n-weighted mean of |mean_pred − actual|.
- The report (`game_model.json` / `.md`) gains:
  - A reliability table for the model and for Vegas on the 2025 test.
  - ECE for both.
  - An **early-season (slates 1–4)** row in the MAE/Brier table.
  - A "baseline" block recording the numbers from before this change (hard-coded constants from the current report) so improvement is visible.
- `simulate.load_sigma` becomes `load_sigmas`, which returns the phase dict. When `sigma_by_phase` is absent from an older report, it falls back to `{"early": s, "mid": s, "post": s}`. `run_simulation` looks up each game's phase.

## 3. Per-week simulation history

### Table
The table is `sim_history`, key `(season, as_of_slate, team)`:

| Column | Meaning |
|---|---|
| `as_of_slate` | First slate not yet played |
| `run_at` | Timestamp of the run |
| `mean_wins` | Mean simulated regular-season wins |
| `p_10_plus` | Chance of 10 or more wins |
| `p_title_game` | Chance of reaching the conference title game |
| `p_conf_champ` | Chance of winning the conference |
| `p_cfp` | Chance of making the College Football Playoff |
| `backfilled` | BOOLEAN; true when the row came from `--backfill` |

The probability columns are the same as in `sim_team_summary`.

### Forward runs
Each `psu simulate` run writes its summary to `sim_history`. It first deletes any rows with the same `(season, as_of_slate)`, which makes the write an upsert. The existing `sim_*` tables keep their meaning as "latest run".

### Backfill
`psu simulate --backfill` rebuilds the history for every past slate N of the current season:

1. Games whose slate is before N use actual results.
2. Every game at or after slate N gets features from **ratings frozen at slate N**:
   - `features.frozen_ratings(ratings, season, as_of_slate)` returns each team's row from the latest slate at or before N.
   - A team with no row at or before N uses its first row of the season. That row holds only its preseason prior, because the team had played no games yet.
   - A team that had a bye just before N carries a rating that is one week stale. That is acceptable: it never uses information from after N.
   - Rest and Vegas features are unchanged.
3. The saved model predicts margins from those features; phase sigmas come from the report.
4. The existing simulation code runs on those margins, and the rows are written with `backfilled = true`.

A caveat, noted in the report and the dashboard caption: the model's coefficients were fitted with this season's completed games in view. The ratings are honest as of week N; the coefficients are not.

### Dashboard
- The Predictions page gains an "Odds over time" line chart for the simulated team (Penn State): `p_cfp`, `p_conf_champ` and `p_title_game` by `as_of_slate`.
- There is no team picker, because the simulator reports on one team per run.
- When `sim_history` is empty or missing, the page shows a note instead of an error.

## Error handling
- A missing `returning_production` table or rows means projection with the fallback median `ret`. It never raises.
- An older model report without phase sigmas falls back to the pooled sigma.
- `--backfill` before any games are completed writes nothing and prints a message.
- CLI errors go through the existing handler and DB-error reporting in `cli.py`.

## Testing
All tests use synthetic fixtures; none call the API.
- `priors`: pairs are built correctly; the fit recovers known `b0`/`b1` from synthetic data; the fallback is used with no pairs; clipping works; no season's coefficients use its own or a later season.
- `features`: `d_returning` is present; `frozen_ratings` never returns ratings from a slate later than the as-of slate.
- `game_predict`: phase assignment; phase sigmas come from out-of-sample residuals with the fallback below 30; reliability bins and ECE on hand-computed cases.
- `simulate`: `load_sigmas` handles old and new reports; the history upsert replaces the same `(season, as_of_slate)` and keeps the others; backfill fixes games before N to actual results.
- `ingest`: the returning endpoint is scheduled with `SLOW_REFRESH` (mocked client).
- Dashboard: the history loader returns an empty frame when the table is missing.

## Outcome (2026-09-25)
On the real data, the 2025 test season scored (model MAE, early-season model MAE):
- Old raw prior, no `d_returning`: 12.517, 12.805
- Projected prior + `d_returning`: 12.586, 13.096
- Projected prior alone: 12.562, 13.032
- `d_returning` alone: 12.541, 12.877

Both the projected preseason prior (section 1, `psu/priors.py`) and the `d_returning` feature made
predictions worse on every combination tried, so they were reverted: `rolling_ratings` goes back to
using last season's final ratings directly as the prior, and `d_returning` was dropped from
`FEATURES`. The returning-production ingest was kept as-is (data collected weekly, not yet used by
the model), along with the phase sigmas, calibration report, `sim_history`, backfill, and
`frozen_ratings` from the rest of this design.
