# Phase 4: Season simulator (design)

Date: 2026-09-23. Status: approved in chat; awaiting review of this written spec.

## Goal

Simulate the rest of Penn State's current season many times (default 10,000) using the Phase 3 game model. Report:

- the distribution of regular-season win totals and the mean,
- P(10+ regular-season wins),
- P(Big Ten title game appearance) and P(Big Ten champion),
- a rough P(College Football Playoff).

Constraints: no API calls; full run under one minute; results stored in DuckDB for the Phase 5 dashboard.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Engine | Power-rating simulation (approach A): per-game `pred_margin`, a shared per-team strength draw, game noise. |
| Scope | Simulate the whole Big Ten regular season (title-game odds depend on every team). |
| Tiebreakers | Simplified: conference win %, then head-to-head, then a seeded coin flip. |
| CFP | Record threshold: in if Big Ten champion, or ≤ 2 total losses (title-game loss counts). Both are settings. |
| Win totals | Regular season only. The title game counts only toward the champion and CFP outcomes. Bowls and CFP games are not simulated. |

## Inputs (all from DuckDB, read-only)

- `game_predictions`: every row for the current season, played or not, with `pred_margin` and `neutral_site`. The ratings used for the title game are fitted on all of them, not just the remaining games, so the fit stays sharp late in the season.
- `games`: current-season results (`completed`, points), `conference_game`, `home_conference`, `away_conference`, `season_type`, `notes`.
- `data/reports/game_model.json`: `sigma` (the normal spread of the model's margin errors, ~16.3).

**Big Ten membership** is taken from the current season's `games` (teams whose `home_conference` or `away_conference` is `"Big Ten"`). A **conference game** is a regular-season game with both teams in the Big Ten, excluding the conference title game itself. The **title game** is identified as a regular-season game between two conference members whose `notes` contains `"<conference> Championship"` (case-insensitive); it is removed before standings, win totals, and head-to-head are computed. If it's present and completed, its actual teams and winner are used every simulation; if present and not yet played, its two teams are fixed and the winner is simulated; if absent, the simulated top two teams meet instead.

## Components

### `src/psu/sim/ratings.py`: power ratings

`fit_ratings(upcoming: DataFrame, ridge: float = 1.0) -> Ratings`

- Least squares on `pred_margin ≈ hfa·(not neutral) + r[home] − r[away]` over all upcoming current-season predictions, with a small ridge penalty on team ratings, ratings centred at mean 0.
- Returns a `Ratings` dataclass: `hfa: float`, `rating: dict[str, float]`, and `neutral_margin(a, b) -> float` = `rating[a] − rating[b]`.
- Used only for games without a `pred_margin`, meaning the neutral-site title game. Regular-season games keep their own `pred_margin`, which carries per-game features (rest, talent, and so on).

### `src/psu/sim/season.py`: vectorised game draws

`draw_margins(pred_margin: ndarray[G], home_idx, away_idx, n_teams, *, n_sims, sigma, tau, rng) -> ndarray[n_sims, G]`

- Per simulation, each team gets a strength draw `u ~ N(0, tau)` shared across all its games.
- Game margin = `pred_margin + u[home] − u[away] + ε`, with `ε ~ N(0, s)` and `s = sqrt(sigma² − 2·tau²)`, so the marginal spread of each game matches the model's `sigma`.
- Raises `ValueError` if `2·tau² ≥ sigma²`.
- The home team wins when margin > 0 (a margin of exactly 0 has probability zero with continuous draws).

The same function is reused for the title game, with `pred_margin = neutral_margin(a, b)` and the same per-simulation `u` (pass the draws in, so a team's strength is consistent between its regular season and the title game).

### `src/psu/sim/standings.py`: title game participants

`title_game_teams(conf_wins, conf_games, h2h, rng) -> (team_a, team_b)` for one simulation (a Python loop over 10,000 simulations is fast enough).

Ranking:

1. Conference win % (wins / conference games played).
2. Within a group tied on win %: head-to-head win % in games among the tied teams. It applies only if every team in the group has played every other team in the group. Otherwise skip to step 3.
3. A seeded random order (coin flip).

Tiebreaks are applied only where they decide the top two: resolve the group containing 1st place, then (if needed) the group containing 2nd place. After a head-to-head step, teams that are still level fall to the coin flip, without re-applying head-to-head to a smaller subgroup.

### `src/psu/simulate.py`: orchestration

`simulate_season(con, *, season, team="Penn State", n_sims=10_000, seed=0, tau=5.0, cfp_max_losses=2, sigma) -> SimResult`

1. Load inputs. Completed games use actual results; upcoming games use predictions.
2. Fit ratings. Draw `u` and all remaining game margins for every Big Ten conference game and every remaining regular-season game of `team`.
3. Per simulation: Big Ten standings → title game teams → title game margin via `neutral_margin` + `u` + noise → champion.
4. Tally the `team` outcomes and every Big Ten team's title game and champion rates.

If a remaining regular-season game of `team` has no prediction (an FCS opponent), it is treated as a win for `team` with probability `unrated_win_prob = 0.95` (a setting) and a warning is logged. There are none for Penn State in 2026.

`write_results(con, result)` replaces three tables:

- `sim_team_summary`: one row with `season, team, n_sims, seed, tau, as_of, mean_wins, p_10_plus, p_title_game, p_conf_champ, p_cfp`. `as_of` = the latest `start_date` among completed current-season games.
- `sim_win_totals`: `season, team, wins, prob` for every possible regular-season win total (0 to the number of regular-season games).
- `sim_conference`: `season, team, mean_conf_wins, p_title_game, p_conf_champ` for each Big Ten team.

It also writes `data/reports/season_sim.md` (a short summary table).

### CLI

`psu simulate [--sims 10000] [--seed 0] [--tau 5]`

- It prints the summary and the win-total table.
- If `game_predictions` or `game_model.json` is missing, it prints `error: run \`psu train\` first` and exits with code 2.
- It makes no API calls.

## Testing (test-first, synthetic data)

- `fit_ratings` recovers known ratings and the home edge from noiseless synthetic margins.
- `draw_margins` with `tau = 0`, a single game and a large `n_sims` gives a home win rate within tolerance of `Φ(pred_margin / sigma)`. With `tau > 0`, the per-game spread still matches `sigma`, and two games sharing a team have correlated outcomes.
- `draw_margins` raises on `2·tau² ≥ sigma²`.
- `title_game_teams`: a clear top two; a two-way tie broken by head-to-head; a three-way tie with an incomplete round robin that goes to the coin flip; coin-flip results that are reproducible with a fixed seed.
- The CFP rule at its boundaries: champion with 3 losses is in; 2 losses non-champion is in; 3 losses non-champion is out; a title-game loss counts toward losses.
- `simulate_season` on a small synthetic league: probabilities are in [0, 1], `sim_win_totals` sums to 1, completed games are fixed (a team with a completed loss never goes undefeated), and the results are reproducible with the same seed.
- The CLI writes all three tables. A timing check on the real database (not in CI) confirms the full run is under one minute.

## Out of scope

Official Big Ten tiebreaker rules, other conferences' title games, a national ranking or committee model, bowl and CFP game simulation, and dashboard pages (Phase 5).
