# Phase 4: Season Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **In this repo:** implementers and reviewers follow `.claude/agents/psu-implementer.md` and `.claude/agents/psu-reviewer.md`. Give each agent only its own task, plus **Global Constraints** and **Review Focus**.

**Goal:** `psu simulate` runs a Monte Carlo of the rest of the current season (default 10,000 runs). It writes the following to DuckDB and to a short markdown report:

- Penn State's regular-season win distribution
- P(10+ wins)
- P(Big Ten title game) and P(Big Ten champion)
- a rough P(CFP)
- every Big Ten team's title odds

**Architecture:** A small `psu.sim` package does the pure numerics:

- `ratings.py` fits power ratings to the model's predicted margins, for the neutral-site title game.
- `season.py` does vectorised game draws, with a shared per-team strength draw plus game noise.
- `standings.py` picks the title game pair using simplified tiebreakers.

`simulate.py` combines them over DataFrames (`run_simulation`) and handles the DuckDB and file I/O. `psu simulate` wraps it.

**Tech Stack:** numpy (`default_rng`, vectorised draws), pandas, DuckDB, scipy (`norm`, tests only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-phase4-season-simulator-design.md`

## Global Constraints

- Branch `feat/phase4-season-sim`. Make one Conventional Commit per task (the message is given in the task), staging only that task's files by explicit path. End each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No CFBD API calls.** Never run `psu ingest`. Unit tests use synthetic frames or in-memory DuckDB.
- Run tests with `.venv/Scripts/python -m pytest` (Windows). Never import `sklearn.impute` or `sklearn.neighbors`, because Smart App Control blocks them.
- **Game draws:** each team gets `u ~ N(0, tau)` per simulation, shared across all its games. A game's home margin is `pred_margin + u[home] - u[away] + eps`, with `eps ~ N(0, s)` and `s = sqrt(sigma^2 - 2*tau^2)`. `sigma` is read from `data/reports/game_model.json` (`"sigma"`). The default `tau` is 5.0. Raise `ValueError` if `tau < 0` or `2*tau^2 >= sigma^2`. The home team wins when margin > 0.
- **Completed games** always use their actual result (`home_points > away_points`). They are never simulated.
- **Big Ten** membership is the current season's regular-season teams with `home_conference` or `away_conference == "Big Ten"`. A conference game is a regular-season game between two members.
- **Standings:** order by conference win %. Within a tie, use head-to-head win % among the tied teams, but only if every pair in the group has played. Otherwise, and for anyone still level, use a seeded random order. Tiebreaks resolve only the 1st-place group, then the 2nd-place group.
- **Title game:** 1st vs 2nd at a neutral site. Margin = `rating[a] - rating[b] + u[a] - u[b] + eps`, using the same per-simulation `u`. A team missing from the fitted ratings gets 0.0.
- **Win totals** are regular season only. The title game counts only toward champion and CFP. **CFP rule:** champion, or total losses (regular season plus a title-game loss) `<= cfp_max_losses` (default 2).
- **A remaining game of `team` with no prediction** (FCS opponent): `team` wins with probability `unrated_win_prob = 0.95`, and a warning is logged. **A remaining conference game with no prediction** raises `MissingModel` (the message mentions `psu train`).
- All randomness comes from one `np.random.default_rng(seed)`, and games are processed in `id` order, so the same seed gives identical results whether the input came from DuckDB or a DataFrame.
- Output tables are replaced on every run: `sim_team_summary`, `sim_win_totals`, `sim_conference`. The report goes to `data/reports/season_sim.md`, which is gitignored.
- Console output must be ASCII only (the Windows console is cp1252).

## Review Focus

1. **Completed games re-simulated by mistake:** a team that already lost can never finish unbeaten, and a finished season gives a single certain win total. Tested in Task 4 (`test_completed_games_are_fixed`, `test_finished_season_uses_actual_results`).
2. **Games with no prediction:** Penn State's FCS game counts as a likely win with a warning. A conference game with no prediction stops with a `psu train` message instead of silently dropping out of the standings. Tested in Task 4 (`test_unrated_team_game_is_a_likely_win_with_warning`, `test_conference_game_without_prediction_is_an_error`).
3. **Phase 3 outputs missing** (no `game_predictions` table or no `game_model.json`): the command exits with code 2 and says to run `psu train`, and no traceback is shown. Tested in Task 5 (`test_load_inputs_without_predictions_is_missing_model`, `test_load_sigma_missing_file`) and Task 6 (`test_simulate_requires_trained_model`).
4. **A `--tau` too large for sigma:** the command gives a clean usage error, not a NaN from `sqrt` of a negative. Tested in Task 2 (`test_noise_sd_rejects_tau_too_large_for_sigma`) and Task 6 (`test_simulate_rejects_too_large_tau`).
5. **Late or finished season** (no upcoming predictions left): the rating fit handles an empty frame and the simulation still runs. Tested in Task 1 (`test_fit_ratings_on_no_games_is_empty`) and Task 4 (`test_finished_season_uses_actual_results`).

## File Map

| File | Responsibility |
|---|---|
| `src/psu/sim/__init__.py` | package marker |
| `src/psu/sim/ratings.py` | `Ratings`, `fit_ratings` |
| `src/psu/sim/season.py` | `game_noise_sd`, `draw_strengths`, `draw_margins`, `draw_matchups` |
| `src/psu/sim/standings.py` | `top_two` (and private `_break_tie`) |
| `src/psu/simulate.py` | `MissingModel`, `SimResult`, `makes_cfp`, `run_simulation`, `load_inputs`, `load_sigma`, `simulate_season`, `write_results`, `report_markdown` |
| `src/psu/cli.py` | adds `psu simulate` |
| `tests/conftest.py` | adds `synthetic_league()`, `seed_league_db(con)` |
| `tests/test_sim_ratings.py`, `tests/test_sim_season.py`, `tests/test_sim_standings.py`, `tests/test_simulate.py`, `tests/test_cli.py` | tests |

---

### Task 1: Power ratings (`sim/ratings.py`)

**Files:**
- Create: `src/psu/sim/__init__.py`
- Create: `src/psu/sim/ratings.py`
- Test: `tests/test_sim_ratings.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. The input frame has the columns `home_team, away_team, neutral_site, pred_margin` (upcoming `game_predictions` rows).
- Produces:
  - `Ratings(hfa: float, rating: dict[str, float])`, a frozen dataclass with `neutral_margin(a: str, b: str) -> float` = `rating[a] - rating[b]`
  - `fit_ratings(upcoming: pd.DataFrame, ridge: float = 1.0) -> Ratings`. An empty frame returns `Ratings(0.0, {})`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sim_ratings.py`:

```python
import pandas as pd
import pytest

from psu.sim.ratings import Ratings, fit_ratings

TRUE = {"A": 10.0, "B": 4.0, "C": -3.0, "D": -11.0}  # sums to 0


def round_robin(hfa=3.0):
    rows = [
        {"home_team": h, "away_team": a, "neutral_site": False, "pred_margin": hfa + TRUE[h] - TRUE[a]}
        for h in TRUE for a in TRUE if h != a
    ]
    rows.append({"home_team": "A", "away_team": "D", "neutral_site": True, "pred_margin": TRUE["A"] - TRUE["D"]})
    return pd.DataFrame(rows)


def test_fit_ratings_recovers_known_ratings_and_home_edge():
    r = fit_ratings(round_robin(), ridge=1e-6)
    assert r.hfa == pytest.approx(3.0, abs=1e-3)
    for team, value in TRUE.items():
        assert r.rating[team] == pytest.approx(value, abs=1e-3)
    assert r.neutral_margin("B", "C") == pytest.approx(7.0, abs=1e-3)


def test_ridge_shrinks_but_keeps_order_and_centres_ratings():
    r = fit_ratings(round_robin(), ridge=5.0)
    values = [r.rating[t] for t in ("A", "B", "C", "D")]
    assert values == sorted(values, reverse=True)
    assert abs(r.rating["A"]) < TRUE["A"]
    assert sum(values) == pytest.approx(0.0, abs=1e-9)


def test_missing_neutral_flag_counts_as_home_game():
    games = round_robin()
    games["neutral_site"] = games["neutral_site"].astype(object)
    games.loc[0, "neutral_site"] = None
    r = fit_ratings(games, ridge=1e-6)
    assert r.hfa == pytest.approx(3.0, abs=1e-3)


def test_fit_ratings_on_no_games_is_empty():
    empty = pd.DataFrame(columns=["home_team", "away_team", "neutral_site", "pred_margin"])
    assert fit_ratings(empty) == Ratings(0.0, {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_sim_ratings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.sim'`.

- [ ] **Step 3: Write the implementation**

`src/psu/sim/__init__.py`:

```python
"""Season simulation building blocks: power ratings, game draws, and standings."""
```

`src/psu/sim/ratings.py`:

```python
"""Power ratings fitted to the game model's own predicted margins (used for games with no prediction)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ratings:
    hfa: float
    rating: dict[str, float]

    def neutral_margin(self, a: str, b: str) -> float:
        return self.rating[a] - self.rating[b]


def fit_ratings(upcoming: pd.DataFrame, ridge: float = 1.0) -> Ratings:
    """Least squares on pred_margin ~ hfa * (not neutral) + r[home] - r[away], ridge on r, ratings centred at 0."""
    if upcoming.empty:
        return Ratings(0.0, {})
    teams = sorted(set(upcoming["home_team"]) | set(upcoming["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    rows = np.arange(len(upcoming))
    x = np.zeros((len(upcoming), k + 1))
    x[rows, upcoming["home_team"].map(index).to_numpy()] += 1.0
    x[rows, upcoming["away_team"].map(index).to_numpy()] -= 1.0
    x[:, k] = (~upcoming["neutral_site"].fillna(False).astype(bool)).astype(float).to_numpy()
    y = upcoming["pred_margin"].to_numpy(float)
    penalty = np.full(k + 1, float(ridge))
    penalty[k] = 1e-9  # hfa is unpenalised; the tiny term keeps an all-neutral slate solvable
    coef = np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)
    centred = coef[:k] - coef[:k].mean()
    return Ratings(float(coef[k]), {t: float(v) for t, v in zip(teams, centred)})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_sim_ratings.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/sim/__init__.py src/psu/sim/ratings.py tests/test_sim_ratings.py
git commit -m "feat(sim): fit power ratings to predicted margins"
```

---

### Task 2: Vectorised game draws (`sim/season.py`)

**Files:**
- Create: `src/psu/sim/season.py`
- Test: `tests/test_sim_season.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `game_noise_sd(sigma: float, tau: float) -> float`. It raises `ValueError` (the message contains `tau`) if `tau < 0` or `2*tau**2 >= sigma**2`.
  - `draw_strengths(n_sims: int, n_teams: int, tau: float, rng: np.random.Generator) -> np.ndarray` with shape `(n_sims, n_teams)`
  - `draw_margins(pred_margin, home_idx, away_idx, strengths, *, sigma, tau, rng) -> np.ndarray` with shape `(n_sims, n_games)`. Every simulation plays every game.
  - `draw_matchups(pred_margin, a_idx, b_idx, strengths, *, sigma, tau, rng) -> np.ndarray` with shape `(n_sims,)`. There's one game per simulation, and each simulation has its own teams (`a_idx[i]`, `b_idx[i]`, `pred_margin[i]`). The result is the a-minus-b margin.

- [ ] **Step 1: Write the failing tests**

`tests/test_sim_season.py`:

```python
import numpy as np
import pytest
from scipy.stats import norm

from psu.sim.season import draw_margins, draw_matchups, draw_strengths, game_noise_sd

SIGMA = 16.0


def test_noise_sd_keeps_total_spread_at_sigma():
    assert game_noise_sd(SIGMA, 0.0) == pytest.approx(SIGMA)
    assert game_noise_sd(SIGMA, 5.0) == pytest.approx(np.sqrt(SIGMA**2 - 50.0))


def test_noise_sd_rejects_tau_too_large_for_sigma():
    with pytest.raises(ValueError, match="tau"):
        game_noise_sd(SIGMA, 12.0)  # 2 * 144 > 256
    with pytest.raises(ValueError, match="tau"):
        game_noise_sd(SIGMA, -1.0)


def test_single_game_win_rate_matches_normal_cdf_without_team_draws():
    rng = np.random.default_rng(0)
    strengths = draw_strengths(200_000, 2, 0.0, rng)
    margins = draw_margins(np.array([5.0]), np.array([0]), np.array([1]), strengths, sigma=SIGMA, tau=0.0, rng=rng)
    assert margins.shape == (200_000, 1)
    assert (margins[:, 0] > 0).mean() == pytest.approx(norm.cdf(5.0 / SIGMA), abs=0.005)


def test_team_draws_keep_spread_and_correlate_games_sharing_a_team():
    rng = np.random.default_rng(1)
    strengths = draw_strengths(100_000, 3, 8.0, rng)
    margins = draw_margins(
        np.array([0.0, 0.0]), np.array([0, 0]), np.array([1, 2]), strengths, sigma=SIGMA, tau=8.0, rng=rng
    )
    assert margins.std(axis=0) == pytest.approx([SIGMA, SIGMA], rel=0.02)
    corr = np.corrcoef(margins[:, 0], margins[:, 1])[0, 1]
    assert 0.2 < corr < 0.3  # tau^2 / sigma^2 = 64 / 256 = 0.25


def test_games_are_independent_without_team_draws():
    rng = np.random.default_rng(2)
    strengths = draw_strengths(100_000, 3, 0.0, rng)
    margins = draw_margins(
        np.array([0.0, 0.0]), np.array([0, 0]), np.array([1, 2]), strengths, sigma=SIGMA, tau=0.0, rng=rng
    )
    assert abs(np.corrcoef(margins[:, 0], margins[:, 1])[0, 1]) < 0.02


def test_matchups_use_each_simulations_own_teams_and_strengths():
    rng = np.random.default_rng(3)
    n = 50_000
    strengths = np.zeros((n, 3))
    strengths[:, 2] = 100.0  # team 2 is overwhelming in every simulation
    a = np.where(np.arange(n) % 2 == 0, 2, 0)  # even simulations: team 2 vs team 1
    b = np.ones(n, dtype=int)
    margins = draw_matchups(np.zeros(n), a, b, strengths, sigma=SIGMA, tau=0.0, rng=rng)
    assert margins.shape == (n,)
    assert (margins[a == 2] > 0).all()
    assert (margins[a == 0] > 0).mean() == pytest.approx(0.5, abs=0.02)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_sim_season.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.sim.season'`.

- [ ] **Step 3: Write the implementation**

`src/psu/sim/season.py`:

```python
"""Vectorised game draws: a season-long strength per team per simulation, plus independent game noise."""
from __future__ import annotations

import math

import numpy as np


def game_noise_sd(sigma: float, tau: float) -> float:
    """Game-level noise so each game's total spread (two team draws plus noise) is still sigma."""
    if tau < 0 or 2 * tau**2 >= sigma**2:
        raise ValueError(f"tau must satisfy 0 <= tau < sigma/sqrt(2) (sigma={sigma:.1f}, tau={tau:g})")
    return math.sqrt(sigma**2 - 2 * tau**2)


def draw_strengths(n_sims: int, n_teams: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, tau, size=(n_sims, n_teams))


def draw_margins(
    pred_margin, home_idx, away_idx, strengths: np.ndarray, *, sigma: float, tau: float, rng: np.random.Generator
) -> np.ndarray:
    """Home-minus-away margins, shape (n_sims, n_games): every simulation plays every game."""
    s = game_noise_sd(sigma, tau)
    pred = np.asarray(pred_margin, dtype=float)
    noise = rng.normal(0.0, s, size=(strengths.shape[0], pred.size))
    return pred + strengths[:, home_idx] - strengths[:, away_idx] + noise


def draw_matchups(
    pred_margin, a_idx, b_idx, strengths: np.ndarray, *, sigma: float, tau: float, rng: np.random.Generator
) -> np.ndarray:
    """One game per simulation with its own teams (e.g. the title game): a-minus-b margins, shape (n_sims,)."""
    s = game_noise_sd(sigma, tau)
    rows = np.arange(strengths.shape[0])
    noise = rng.normal(0.0, s, size=rows.size)
    return np.asarray(pred_margin, dtype=float) + strengths[rows, a_idx] - strengths[rows, b_idx] + noise
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_sim_season.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/sim/season.py tests/test_sim_season.py
git commit -m "feat(sim): draw correlated game margins with shared team strengths"
```

---

### Task 3: Title game standings (`sim/standings.py`)

**Files:**
- Create: `src/psu/sim/standings.py`
- Test: `tests/test_sim_standings.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `top_two(conf_wins, conf_games, h2h_wins, h2h_games, rng) -> tuple[int, int]`, the member indices of 1st and 2nd.
  - `conf_wins` and `conf_games` are 1-D arrays with one entry per member.
  - `h2h_wins[i, j]` is i's wins over j. `h2h_games[i, j]` is the number of games between i and j (a symmetric integer matrix).
  - It needs at least two members.

- [ ] **Step 1: Write the failing tests**

`tests/test_sim_standings.py`:

```python
import numpy as np

from psu.sim.standings import top_two

FULL = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)  # every pair played once


def h2h(*results):
    """results: (winner, loser) pairs -> wins matrix."""
    wins = np.zeros((4, 4), dtype=int)
    for w, l in results:
        wins[w, l] += 1
    return wins


def rng(seed=0):
    return np.random.default_rng(seed)


def test_clear_top_two():
    wins = h2h((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert top_two(np.array([3, 2, 1, 0]), np.array([3, 3, 3, 3]), wins, FULL, rng()) == (0, 1)


def test_win_percentage_not_raw_wins():
    games = np.array([6, 4, 4, 4])
    assert top_two(np.array([5, 4, 1, 0]), games, np.zeros((4, 4), int), FULL, rng())[0] == 1


def test_two_way_tie_for_first_goes_to_head_to_head():
    wins = h2h((1, 0), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3))  # 0 and 1 both 2-1; 1 beat 0
    for seed in range(20):
        assert top_two(np.array([2, 2, 1, 1]), np.array([3, 3, 3, 3]), wins, FULL, rng(seed)) == (1, 0)


def test_three_way_tie_for_second_uses_complete_round_robin():
    # 0 is clear first. 1, 2 and 3 tie on conference win % (conf_wins is passed directly); among them
    # 2 beat 1 and 3, and 3 beat 1, so head-to-head (2: 2-0, 3: 1-1, 1: 0-2) puts 2 second every time.
    wins = h2h((0, 1), (0, 2), (0, 3), (2, 1), (2, 3), (3, 1))
    conf_wins = np.array([3, 1, 1, 1])
    for seed in range(20):
        assert top_two(conf_wins, np.array([3, 3, 3, 3]), wins, FULL, rng(seed)) == (0, 2)


def test_incomplete_round_robin_falls_to_a_seeded_coin_flip():
    games = FULL.copy()
    games[1, 3] = games[3, 1] = 0  # 1 and 3 never met
    wins = h2h((0, 1), (0, 2), (0, 3), (2, 1), (2, 3))
    conf_wins = np.array([3, 1, 1, 1])
    seconds = {top_two(conf_wins, np.array([3, 3, 3, 3]), wins, games, rng(seed))[1] for seed in range(50)}
    assert seconds == {1, 2, 3}  # head-to-head skipped although 2 beat both others
    assert top_two(conf_wins, np.array([3, 3, 3, 3]), wins, games, rng(7)) == top_two(
        conf_wins, np.array([3, 3, 3, 3]), wins, games, rng(7)
    )


def test_even_head_to_head_is_a_coin_flip():
    wins = h2h((0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1))  # 1, 2, 3 each 1-1 among themselves
    firsts_seconds = {top_two(np.array([3, 1, 1, 1]), np.array([3, 3, 3, 3]), wins, FULL, rng(s)) for s in range(50)}
    assert {s for _, s in firsts_seconds} == {1, 2, 3}
    assert all(f == 0 for f, _ in firsts_seconds)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_sim_standings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.sim.standings'`.

- [ ] **Step 3: Write the implementation**

`src/psu/sim/standings.py`:

```python
"""Pick the two conference title game teams for one simulated season (simplified Big Ten tiebreakers)."""
from __future__ import annotations

import numpy as np


def _break_tie(group: np.ndarray, h2h_wins: np.ndarray, h2h_games: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Order teams tied on conference win %: head-to-head win % (only if every pair played), then a coin flip."""
    if len(group) == 1:
        return [int(group[0])]
    coin = rng.random(len(group))
    games = h2h_games[np.ix_(group, group)]
    every_pair_played = (games + np.eye(len(group), dtype=int) > 0).all()
    if every_pair_played:
        h2h = h2h_wins[np.ix_(group, group)].sum(axis=1) / games.sum(axis=1)
    else:
        h2h = np.zeros(len(group))
    order = np.lexsort((coin, -h2h))  # primary key: head-to-head (desc); then the coin
    return [int(t) for t in group[order]]


def top_two(
    conf_wins, conf_games, h2h_wins: np.ndarray, h2h_games: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    """Indices of the 1st and 2nd placed teams by conference win %, resolving only the ties that decide them."""
    pct = np.asarray(conf_wins, dtype=float) / np.maximum(np.asarray(conf_games, dtype=float), 1.0)
    remaining = np.arange(len(pct))
    order: list[int] = []
    while len(order) < 2:
        tied = np.isclose(pct[remaining], pct[remaining].max())
        order += _break_tie(remaining[tied], h2h_wins, h2h_games, rng)[: 2 - len(order)]
        remaining = remaining[~tied]
    return order[0], order[1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_sim_standings.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/psu/sim/standings.py tests/test_sim_standings.py
git commit -m "feat(sim): pick title game teams with simplified tiebreakers"
```

---

### Task 4: Season simulation over DataFrames (`simulate.py`, part 1)

**Files:**
- Create: `src/psu/simulate.py`
- Modify: `tests/conftest.py` (add `synthetic_league()` and `seed_league_db(con)` at the end of the file)
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes:
  - `fit_ratings(upcoming) -> Ratings` (Task 1)
  - `draw_strengths`, `draw_margins`, `draw_matchups` (Task 2)
  - `top_two` (Task 3)
  - `psu.config.TEAM` (`"Penn State"`)
- Produces:
  - `CONFERENCE = "Big Ten"`
  - `class MissingModel(RuntimeError)`
  - `SimResult`, a frozen dataclass with these fields:
    - `season: int`, `team: str`, `n_sims: int`, `seed: int`, `tau: float`
    - `as_of: pd.Timestamp | None`
    - `mean_wins: float`, `p_10_plus: float`, `p_title_game: float`, `p_conf_champ: float`, `p_cfp: float`
    - `win_totals: pd.DataFrame[wins, prob]`
    - `conference: pd.DataFrame[team, mean_conf_wins, p_title_game, p_conf_champ]`
  - `makes_cfp(losses, champion, max_losses=2) -> np.ndarray[bool]`
  - `run_simulation(games, upcoming, *, season, sigma, team=TEAM, n_sims=10_000, seed=0, tau=5.0, cfp_max_losses=2, unrated_win_prob=0.95, conference=CONFERENCE) -> SimResult`
    - `games` has these columns: `id, season, week, season_type, start_date, completed, home_team, away_team, home_conference, away_conference, home_points, away_points`
    - `upcoming` has these columns: `game_id, home_team, away_team, neutral_site, pred_margin`
  - conftest `synthetic_league() -> (games, upcoming)` and `seed_league_db(con)`

- [ ] **Step 1: Add the synthetic league to `tests/conftest.py`**

Append to the end of `tests/conftest.py` (it already imports `pandas as pd`; add `import pandas as pd` at the top if it doesn't):

```python
def synthetic_league():
    """A 2026 four-team 'Big Ten' (A-D), an outsider X ('Other') and an FCS team F.

    Played: A beat B 30-10 (conference) and X beat C 24-14 (non-conference), both on 2026-09-01. Still to play:
    the other five conference games, A-X, and A-F. Predictions cover every remaining game except A-F (FCS).
    Returns (games, upcoming) shaped like the `games` table and upcoming `game_predictions` rows.
    """
    rows = [
        # id, week, home, away, home_conf, away_conf, completed, home_pts, away_pts, pred_margin
        (1, 1, "A", "B", "Big Ten", "Big Ten", True, 30, 10, None),
        (2, 1, "X", "C", "Other", "Big Ten", True, 24, 14, None),
        (3, 2, "C", "D", "Big Ten", "Big Ten", False, None, None, 0.0),
        (4, 2, "A", "X", "Big Ten", "Other", False, None, None, 20.0),
        (5, 3, "B", "C", "Big Ten", "Big Ten", False, None, None, 2.0),
        (6, 3, "D", "A", "Big Ten", "Big Ten", False, None, None, -20.0),
        (7, 4, "A", "C", "Big Ten", "Big Ten", False, None, None, 20.0),
        (8, 4, "B", "D", "Big Ten", "Big Ten", False, None, None, -1.0),
        (9, 5, "A", "F", "Big Ten", "FCS", False, None, None, None),
    ]
    games = pd.DataFrame([
        {
            "id": gid, "season": 2026, "week": week, "season_type": "regular",
            "start_date": pd.Timestamp(2026, 9, 1) + pd.Timedelta(days=7 * (week - 1)),
            "completed": done, "home_team": home, "away_team": away,
            "home_conference": home_conf, "away_conference": away_conf,
            "home_points": home_pts, "away_points": away_pts,
        }
        for gid, week, home, away, home_conf, away_conf, done, home_pts, away_pts, _ in rows
    ])
    games[["home_points", "away_points"]] = games[["home_points", "away_points"]].astype("Int64")
    upcoming = pd.DataFrame([
        {"game_id": gid, "home_team": home, "away_team": away, "neutral_site": False, "pred_margin": pred}
        for gid, _, home, away, _, _, done, _, _, pred in rows
        if not done and pred is not None
    ])
    return games, upcoming


def seed_league_db(con):
    """Load synthetic_league() into a psu.db connection: rows into `games`, plus a `game_predictions` table."""
    from psu.db import SPECS, upsert

    games, upcoming = synthetic_league()
    upsert(con, SPECS["games"], games)  # creates the declared `games` table (connect() does not)
    con.register("_preds", upcoming.assign(season=2026, split="upcoming"))
    try:
        con.execute("CREATE OR REPLACE TABLE game_predictions AS SELECT * FROM _preds")
    finally:
        con.unregister("_preds")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_simulate.py`:

```python
import logging

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_league
from psu.simulate import MissingModel, makes_cfp, run_simulation

SIGMA = 16.0


def run(team="A", games=None, upcoming=None, **kw):
    g, u = synthetic_league()
    kw = {"season": 2026, "sigma": SIGMA, "team": team, "n_sims": 2000, "seed": 0, **kw}
    return run_simulation(g if games is None else games, u if upcoming is None else upcoming, **kw)


def test_probabilities_are_valid_and_win_totals_sum_to_one():
    r = run()
    assert list(r.win_totals.columns) == ["wins", "prob"]
    assert list(r.win_totals["wins"]) == [0, 1, 2, 3, 4, 5]  # A plays 5 regular-season games
    assert r.win_totals["prob"].sum() == pytest.approx(1.0)
    for p in (r.p_10_plus, r.p_title_game, r.p_conf_champ, r.p_cfp):
        assert 0.0 <= p <= 1.0
    assert r.mean_wins == pytest.approx((r.win_totals["wins"] * r.win_totals["prob"]).sum())
    assert r.p_10_plus == 0.0  # only 5 games


def test_completed_games_are_fixed():
    r = run(team="B")  # B already lost to A
    assert list(r.win_totals["wins"]) == [0, 1, 2, 3]
    assert r.win_totals.loc[r.win_totals["wins"] == 3, "prob"].item() == 0.0


def test_same_seed_gives_identical_results_and_other_seeds_differ():
    a, b, c = run(seed=4), run(seed=4), run(seed=5)
    pd.testing.assert_frame_equal(a.win_totals, b.win_totals)
    pd.testing.assert_frame_equal(a.conference, b.conference)
    assert a.p_conf_champ == b.p_conf_champ
    assert not a.conference.equals(c.conference)


def test_conference_odds_add_up():
    r = run()
    assert list(r.conference.columns) == ["team", "mean_conf_wins", "p_title_game", "p_conf_champ"]
    assert list(r.conference["team"]) == ["A", "B", "C", "D"]
    assert r.conference["p_title_game"].sum() == pytest.approx(2.0)
    assert r.conference["p_conf_champ"].sum() == pytest.approx(1.0)


def test_dominant_team_wins_the_conference():
    r = run(sigma=1.0, tau=0.1)
    a = r.conference.set_index("team").loc["A"]
    assert a["mean_conf_wins"] == pytest.approx(3.0)
    assert a["p_title_game"] == pytest.approx(1.0)
    assert r.p_conf_champ > 0.99 and r.p_cfp > 0.99


def test_unrated_team_game_is_a_likely_win_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="psu.simulate"):
        r = run(sigma=1.0, tau=0.1, n_sims=4000)
    assert "no prediction" in caplog.text
    assert r.win_totals.loc[r.win_totals["wins"] == 5, "prob"].item() == pytest.approx(0.95, abs=0.02)


def test_conference_game_without_prediction_is_an_error():
    _, upcoming = synthetic_league()
    with pytest.raises(MissingModel, match="psu train"):
        run(upcoming=upcoming[upcoming["game_id"] != 5])


def test_finished_season_uses_actual_results():
    games, upcoming = synthetic_league()
    games["completed"] = True
    todo = games["home_points"].isna()
    games.loc[todo, "home_points"] = 21  # every remaining game: home team wins 21-14
    games.loc[todo, "away_points"] = 14
    r = run(games=games, upcoming=upcoming.iloc[0:0])
    # A: beat B, beat X, lost at D, beat C, beat F -> 4 wins for certain
    assert r.win_totals.loc[r.win_totals["wins"] == 4, "prob"].item() == 1.0
    # A and B finish 2-1; A won head-to-head, so the title game is A vs B every time
    conf = r.conference.set_index("team")
    assert conf.loc["A", "p_title_game"] == 1.0 and conf.loc["B", "p_title_game"] == 1.0
    assert r.as_of == games["start_date"].max()


def test_as_of_is_latest_completed_game():
    assert run().as_of == pd.Timestamp(2026, 9, 1)


def test_makes_cfp_boundaries():
    losses = np.array([3, 2, 3, 1])
    champion = np.array([True, False, False, False])
    assert makes_cfp(losses, champion).tolist() == [True, True, False, True]
    assert makes_cfp(np.array([2]), np.array([False]), max_losses=1).tolist() == [False]


def test_title_game_loss_counts_toward_cfp_losses():
    r = run(team="B", sigma=1.0, tau=0.1, cfp_max_losses=1)
    # B already lost to A. Whenever B reaches the title game it loses to A (far stronger), giving B at least
    # 2 losses, so with max_losses=1 no season where B plays in the title game can reach the playoff.
    assert r.p_title_game > 0
    assert r.p_cfp <= 1.0 - r.p_title_game + 1e-9
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.simulate'`.

- [ ] **Step 4: Write the implementation**

`src/psu/simulate.py`:

```python
"""Season simulator: Monte Carlo over the rest of the current season using the Phase 3 game model."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from psu.config import TEAM
from psu.sim.ratings import fit_ratings
from psu.sim.season import draw_margins, draw_matchups, draw_strengths
from psu.sim.standings import top_two

log = logging.getLogger(__name__)

CONFERENCE = "Big Ten"


class MissingModel(RuntimeError):
    """The Phase 3 outputs the simulator needs are missing; run `psu train` first."""


@dataclass(frozen=True)
class SimResult:
    season: int
    team: str
    n_sims: int
    seed: int
    tau: float
    as_of: pd.Timestamp | None
    mean_wins: float
    p_10_plus: float
    p_title_game: float
    p_conf_champ: float
    p_cfp: float
    win_totals: pd.DataFrame
    conference: pd.DataFrame


def makes_cfp(losses, champion, max_losses: int = 2) -> np.ndarray:
    return np.asarray(champion, dtype=bool) | (np.asarray(losses) <= max_losses)


def run_simulation(
    games: pd.DataFrame,
    upcoming: pd.DataFrame,
    *,
    season: int,
    sigma: float,
    team: str = TEAM,
    n_sims: int = 10_000,
    seed: int = 0,
    tau: float = 5.0,
    cfp_max_losses: int = 2,
    unrated_win_prob: float = 0.95,
    conference: str = CONFERENCE,
) -> SimResult:
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    rng = np.random.default_rng(seed)
    season_games = games[games["season"] == season]
    regular = season_games[season_games["season_type"] == "regular"]
    members = sorted(
        set(regular.loc[regular["home_conference"] == conference, "home_team"])
        | set(regular.loc[regular["away_conference"] == conference, "away_team"])
    )
    if len(members) < 2:
        raise ValueError(f"fewer than two {conference} teams in {season}")

    is_conf = regular["home_team"].isin(members) & regular["away_team"].isin(members)
    is_team = (regular["home_team"] == team) | (regular["away_team"] == team)
    preds = upcoming[["game_id", "pred_margin"]].rename(columns={"game_id": "id"})
    relevant = regular[is_conf | is_team].drop_duplicates("id").merge(preds, on="id", how="left")
    relevant = relevant.sort_values("id").reset_index(drop=True)  # fixed column order: same seed, same draws

    completed = relevant["completed"].fillna(False).astype(bool).to_numpy()
    predicted = ~completed & relevant["pred_margin"].notna().to_numpy()
    unrated = ~completed & ~predicted
    conf_col = (relevant["home_team"].isin(members) & relevant["away_team"].isin(members)).to_numpy()
    if (unrated & conf_col).any():
        missing = relevant.loc[unrated & conf_col, "id"].tolist()
        raise MissingModel(f"no prediction for {conference} games {missing}; run `psu train` first")

    teams = sorted(set(relevant["home_team"]) | set(relevant["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    home = relevant["home_team"].map(index).to_numpy()
    away = relevant["away_team"].map(index).to_numpy()
    strengths = draw_strengths(n_sims, len(teams), tau, rng)

    home_win = np.zeros((n_sims, len(relevant)), dtype=bool)
    done = relevant.loc[completed]
    home_win[:, completed] = done["home_points"].to_numpy(float) > done["away_points"].to_numpy(float)
    if predicted.any():
        margins = draw_margins(
            relevant.loc[predicted, "pred_margin"].to_numpy(float), home[predicted], away[predicted],
            strengths, sigma=sigma, tau=tau, rng=rng,
        )
        home_win[:, predicted] = margins > 0
    if unrated.any():
        log.warning(
            "%d %s game(s) have no prediction; counting each as a win with probability %.2f",
            int(unrated.sum()), team, unrated_win_prob,
        )
        team_won_unrated = rng.random((n_sims, int(unrated.sum()))) < unrated_win_prob
        team_is_home = (relevant.loc[unrated, "home_team"] == team).to_numpy()
        home_win[:, unrated] = np.where(team_is_home, team_won_unrated, ~team_won_unrated)

    team_col = ((relevant["home_team"] == team) | (relevant["away_team"] == team)).to_numpy()
    team_home = (relevant["home_team"] == team).to_numpy()[team_col]
    wins = np.where(team_home, home_win[:, team_col], ~home_win[:, team_col]).sum(axis=1)
    n_games = int(team_col.sum())

    m_index = {t: i for i, t in enumerate(members)}
    ch = relevant.loc[conf_col, "home_team"].map(m_index).to_numpy()
    ca = relevant.loc[conf_col, "away_team"].map(m_index).to_numpy()
    conf_home_win = home_win[:, conf_col]
    n_members = len(members)
    conf_wins = np.zeros((n_sims, n_members), dtype=int)
    for j in range(len(ch)):
        conf_wins[:, ch[j]] += conf_home_win[:, j]
        conf_wins[:, ca[j]] += ~conf_home_win[:, j]
    conf_games = np.bincount(np.concatenate([ch, ca]), minlength=n_members)
    h2h_games = np.zeros((n_members, n_members), dtype=int)
    np.add.at(h2h_games, (ch, ca), 1)
    np.add.at(h2h_games, (ca, ch), 1)

    first = np.empty(n_sims, dtype=int)
    second = np.empty(n_sims, dtype=int)
    for i in range(n_sims):
        hw = conf_home_win[i]
        h2h_wins = np.zeros((n_members, n_members), dtype=int)
        np.add.at(h2h_wins, (ch[hw], ca[hw]), 1)
        np.add.at(h2h_wins, (ca[~hw], ch[~hw]), 1)
        first[i], second[i] = top_two(conf_wins[i], conf_games, h2h_wins, h2h_games, rng)

    ratings = fit_ratings(upcoming)
    member_rating = np.array([ratings.rating.get(t, 0.0) for t in members])
    member_team = np.array([index[t] for t in members])
    title_margin = draw_matchups(
        member_rating[first] - member_rating[second], member_team[first], member_team[second],
        strengths, sigma=sigma, tau=tau, rng=rng,
    )
    champion = np.where(title_margin > 0, first, second)

    t = m_index.get(team)
    in_title = np.zeros(n_sims, dtype=bool) if t is None else (first == t) | (second == t)
    champ = np.zeros(n_sims, dtype=bool) if t is None else champion == t
    losses = n_games - wins + (in_title & ~champ)
    cfp = makes_cfp(losses, champ, cfp_max_losses)

    completed_dates = season_games.loc[season_games["completed"].fillna(False).astype(bool), "start_date"]
    return SimResult(
        season=season,
        team=team,
        n_sims=n_sims,
        seed=seed,
        tau=tau,
        as_of=pd.Timestamp(completed_dates.max()) if len(completed_dates) else None,
        mean_wins=float(wins.mean()),
        p_10_plus=float((wins >= 10).mean()),
        p_title_game=float(in_title.mean()),
        p_conf_champ=float(champ.mean()),
        p_cfp=float(cfp.mean()),
        win_totals=pd.DataFrame({
            "wins": np.arange(n_games + 1),
            "prob": np.bincount(wins, minlength=n_games + 1) / n_sims,
        }),
        conference=pd.DataFrame({
            "team": members,
            "mean_conf_wins": conf_wins.mean(axis=0),
            "p_title_game": [float(((first == m) | (second == m)).mean()) for m in range(n_members)],
            "p_conf_champ": np.bincount(champion, minlength=n_members) / n_sims,
        }),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py -v`
Expected: 11 passed.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python -m pytest`
Expected: everything passes, including the earlier tests, since conftest only gained new helpers.

- [ ] **Step 7: Commit**

```bash
git add src/psu/simulate.py tests/conftest.py tests/test_simulate.py
git commit -m "feat(simulate): Monte Carlo the season, standings, title game and CFP odds"
```

---

### Task 5: DuckDB and report I/O (`simulate.py`, part 2)

**Files:**
- Modify: `src/psu/simulate.py` (add the I/O functions; the imports go at the top)
- Test: `tests/test_simulate_io.py`

**Interfaces:**
- Consumes:
  - `run_simulation`, `SimResult`, `MissingModel` (Task 4)
  - `psu.build._write(con, name, df)`, which does `CREATE OR REPLACE TABLE` from a DataFrame
  - conftest `synthetic_league`, `seed_league_db`
  - `psu.db.connect(":memory:")`, which does **not** create the declared tables. `seed_league_db` creates `games` through `psu.db.upsert`.
- Produces:
  - `load_inputs(con, season) -> (games, upcoming)`. It raises `MissingModel` if the `game_predictions` table doesn't exist.
  - `load_sigma(out_dir: Path) -> float`. It reads `out_dir/reports/game_model.json`, and raises `MissingModel` if the file is missing.
  - `simulate_season(con, *, season, sigma, team=TEAM, n_sims=10_000, seed=0, tau=5.0, cfp_max_losses=2) -> SimResult`
  - `write_results(con, result, out_dir: Path) -> None`. It replaces `sim_team_summary`, `sim_win_totals` and `sim_conference`, and writes `out_dir/reports/season_sim.md`.
  - `report_markdown(result) -> str` (ASCII only)
  - `SUMMARY_COLUMNS = ["season", "team", "n_sims", "seed", "tau", "as_of", "mean_wins", "p_10_plus", "p_title_game", "p_conf_champ", "p_cfp"]`

- [ ] **Step 1: Write the failing tests**

`tests/test_simulate_io.py`:

```python
import json

import pytest

from conftest import seed_league_db, synthetic_league
from psu.db import connect
from psu.simulate import (
    SUMMARY_COLUMNS, MissingModel, load_inputs, load_sigma, report_markdown, run_simulation, simulate_season,
    write_results,
)


def test_load_inputs_without_predictions_is_missing_model():
    con = connect(":memory:")
    with pytest.raises(MissingModel, match="psu train"):
        load_inputs(con, 2026)


def test_load_sigma_missing_file(tmp_path):
    with pytest.raises(MissingModel, match="psu train"):
        load_sigma(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "game_model.json").write_text(json.dumps({"sigma": 16.25}), encoding="utf-8")
    assert load_sigma(tmp_path) == 16.25


def test_simulate_season_from_duckdb_matches_dataframe_run():
    con = connect(":memory:")
    seed_league_db(con)
    from_db = simulate_season(con, season=2026, sigma=16.0, team="A", n_sims=500, seed=3)
    games, upcoming = synthetic_league()
    direct = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=500, seed=3)
    assert from_db.win_totals.equals(direct.win_totals)
    assert from_db.p_conf_champ == direct.p_conf_champ
    assert from_db.as_of == direct.as_of


def test_only_upcoming_rows_of_the_season_are_used():
    con = connect(":memory:")
    seed_league_db(con)
    con.execute("UPDATE game_predictions SET split = 'in_sample' WHERE game_id = 5")
    with pytest.raises(MissingModel):
        simulate_season(con, season=2026, sigma=16.0, team="A", n_sims=10)


def test_write_results_replaces_tables_and_writes_report(tmp_path):
    con = connect(":memory:")
    games, upcoming = synthetic_league()
    result = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=300)
    write_results(con, result, tmp_path)
    write_results(con, result, tmp_path)  # a second run replaces, never appends

    summary = con.execute("SELECT * FROM sim_team_summary").df()
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert len(summary) == 1 and summary.loc[0, "team"] == "A" and summary.loc[0, "n_sims"] == 300

    totals = con.execute("SELECT * FROM sim_win_totals").df()
    assert list(totals.columns) == ["season", "team", "wins", "prob"]
    assert len(totals) == 6 and totals["prob"].sum() == pytest.approx(1.0)

    conf = con.execute("SELECT * FROM sim_conference").df()
    assert list(conf.columns) == ["season", "team", "mean_conf_wins", "p_title_game", "p_conf_champ"]
    assert len(conf) == 4

    text = (tmp_path / "reports" / "season_sim.md").read_text(encoding="utf-8")
    assert text == report_markdown(result)


def test_report_is_ascii_and_names_the_key_odds():
    games, upcoming = synthetic_league()
    text = report_markdown(run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=200))
    text.encode("ascii")
    for label in ("Mean wins", "P(10+ wins)", "P(title game)", "P(Big Ten champ)", "P(CFP)", "2026-09-01"):
        assert label in text


def test_report_before_any_game_is_played():
    games, upcoming = synthetic_league()
    games = games[~games["completed"]]
    text = report_markdown(run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=50))
    assert "no games played yet" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_simulate_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'SUMMARY_COLUMNS'`.

- [ ] **Step 3: Write the implementation**

In `src/psu/simulate.py`, add these to the imports at the top:

```python
import json
from pathlib import Path

import duckdb

from psu.build import _write
```

Then append to the end of `src/psu/simulate.py`:

```python
GAME_COLUMNS = (
    "id, season, week, season_type, start_date, completed, home_team, away_team, "
    "home_conference, away_conference, home_points, away_points"
)
SUMMARY_COLUMNS = [
    "season", "team", "n_sims", "seed", "tau", "as_of",
    "mean_wins", "p_10_plus", "p_title_game", "p_conf_champ", "p_cfp",
]


def load_inputs(con: duckdb.DuckDBPyConnection, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = set(con.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
    if "game_predictions" not in tables:
        raise MissingModel("game_predictions table not found; run `psu train` first")
    games = con.execute(f"SELECT {GAME_COLUMNS} FROM games WHERE season = ?", [season]).df()
    upcoming = con.execute(
        "SELECT game_id, home_team, away_team, neutral_site, pred_margin FROM game_predictions "
        "WHERE season = ? AND split = 'upcoming'",
        [season],
    ).df()
    return games, upcoming


def load_sigma(out_dir: Path) -> float:
    path = Path(out_dir) / "reports" / "game_model.json"
    if not path.exists():
        raise MissingModel(f"{path} not found; run `psu train` first")
    return float(json.loads(path.read_text(encoding="utf-8"))["sigma"])


def simulate_season(
    con: duckdb.DuckDBPyConnection,
    *,
    season: int,
    sigma: float,
    team: str = TEAM,
    n_sims: int = 10_000,
    seed: int = 0,
    tau: float = 5.0,
    cfp_max_losses: int = 2,
) -> SimResult:
    games, upcoming = load_inputs(con, season)
    return run_simulation(
        games, upcoming, season=season, sigma=sigma, team=team, n_sims=n_sims, seed=seed, tau=tau,
        cfp_max_losses=cfp_max_losses,
    )


def report_markdown(result: SimResult) -> str:
    as_of = "no games played yet" if result.as_of is None else f"results through {result.as_of:%Y-%m-%d}"
    lines = [
        f"# Season simulation: {result.team} {result.season}",
        "",
        f"{result.n_sims:,} simulated seasons (seed {result.seed}, tau {result.tau:g}), {as_of}.",
        "",
        "| Mean wins | P(10+ wins) | P(title game) | P(Big Ten champ) | P(CFP) |",
        "|---|---|---|---|---|",
        f"| {result.mean_wins:.2f} | {result.p_10_plus:.1%} | {result.p_title_game:.1%} | "
        f"{result.p_conf_champ:.1%} | {result.p_cfp:.1%} |",
        "",
        "## Regular-season wins",
        "",
        "| Wins | Probability |",
        "|---|---|",
    ]
    lines += [f"| {w} | {p:.1%} |" for w, p in zip(result.win_totals["wins"], result.win_totals["prob"])]
    lines += [
        "",
        "## Big Ten title race",
        "",
        "| Team | Mean conf wins | P(title game) | P(champ) |",
        "|---|---|---|---|",
    ]
    ranked = result.conference.sort_values(["p_conf_champ", "p_title_game"], ascending=False)
    lines += [
        f"| {r.team} | {r.mean_conf_wins:.2f} | {r.p_title_game:.1%} | {r.p_conf_champ:.1%} |"
        for r in ranked.itertuples()
    ]
    return "\n".join(lines) + "\n"


def write_results(con: duckdb.DuckDBPyConnection, result: SimResult, out_dir: Path) -> None:
    summary = pd.DataFrame([{
        "season": result.season, "team": result.team, "n_sims": result.n_sims, "seed": result.seed,
        "tau": result.tau, "as_of": pd.NaT if result.as_of is None else result.as_of, "mean_wins": result.mean_wins,
        "p_10_plus": result.p_10_plus, "p_title_game": result.p_title_game,
        "p_conf_champ": result.p_conf_champ, "p_cfp": result.p_cfp,
    }])[SUMMARY_COLUMNS]
    _write(con, "sim_team_summary", summary)
    _write(
        con, "sim_win_totals",
        result.win_totals.assign(season=result.season, team=result.team)[["season", "team", "wins", "prob"]],
    )
    _write(
        con, "sim_conference",
        result.conference.assign(season=result.season)[
            ["season", "team", "mean_conf_wins", "p_title_game", "p_conf_champ"]
        ],
    )
    reports = Path(out_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "season_sim.md").write_text(report_markdown(result), encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_simulate_io.py tests/test_simulate.py -v`
Expected: all pass (7 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/psu/simulate.py tests/test_simulate_io.py
git commit -m "feat(simulate): load inputs from DuckDB, write sim tables and report"
```

---

### Task 6: `psu simulate` command

**Files:**
- Modify: `src/psu/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `load_sigma`, `simulate_season`, `write_results`, `report_markdown` and `MissingModel` from `psu.simulate` (Task 5); `config.TEAM`; conftest `seed_league_db`.
- Produces: `psu simulate [--sims N] [--seed N] [--tau X] [--team NAME]`. It exits with code 0 on success. It exits with code 2 on `MissingModel` or `ValueError`, printing `error: ...` to stderr.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def _trained(settings):
    from conftest import seed_league_db
    from psu import db

    reports = settings.db_path.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "game_model.json").write_text('{"sigma": 16.0}', encoding="utf-8")
    con = db.connect(settings.db_path)
    try:
        seed_league_db(con)
    finally:
        con.close()


def test_simulate_requires_trained_model(settings, capsys):
    assert cli.main(["simulate"]) == 2
    assert "psu train" in capsys.readouterr().err
    assert not settings.db_path.exists()


def test_simulate_writes_tables_and_prints_summary(settings, capsys):
    import duckdb

    _trained(settings)
    assert cli.main(["simulate", "--sims", "200", "--team", "A"]) == 0
    out = capsys.readouterr().out
    assert "P(10+ wins)" in out and "Season simulation: A 2026" in out
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        assert con.execute("SELECT n_sims FROM sim_team_summary").fetchone()[0] == 200
        assert con.execute("SELECT count(*) FROM sim_conference").fetchone()[0] == 4
    finally:
        con.close()
    assert (settings.db_path.parent / "reports" / "season_sim.md").exists()


def test_simulate_rejects_too_large_tau(settings, capsys):
    _trained(settings)
    assert cli.main(["simulate", "--tau", "20", "--team", "A"]) == 2
    assert "tau" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v -k simulate`
Expected: FAIL. argparse exits with `invalid choice: 'simulate'` (`SystemExit: 2`) and the other assertions fail.

- [ ] **Step 3: Write the implementation**

In `src/psu/cli.py`, change the docstring on line 1 to:

```python
"""Command-line entry point: `psu ingest`, `psu status`, `psu build`, `psu train` and `psu simulate`."""
```

Add this import next to the other `psu.*` imports:

```python
from psu.simulate import MissingModel, load_sigma, report_markdown as sim_report, simulate_season, write_results
```

After the `trn` parser lines (after the `--shrink-plays` argument), add:

```python
    sim = sub.add_parser("simulate", help="Simulate the rest of the season (no API calls)")
    sim.add_argument("--sims", type=int, default=10_000, help="Number of simulated seasons")
    sim.add_argument("--seed", type=int, default=0, help="Random seed (same seed, same results)")
    sim.add_argument("--tau", type=float, default=5.0, help="Spread (points) of each team's season-long strength draw")
    sim.add_argument("--team", default=config.TEAM, help="Team to report on")
```

Before the `try: seasons = config.parse_seasons(...)` block (the ingest path), add:

```python
    if args.command == "simulate":
        try:
            sigma = load_sigma(settings.db_path.parent)
            con = db.connect(settings.db_path)
            try:
                result = simulate_season(
                    con, season=settings.current_season, sigma=sigma, team=args.team,
                    n_sims=args.sims, seed=args.seed, tau=args.tau,
                )
                write_results(con, result, settings.db_path.parent)
            finally:
                con.close()
        except (MissingModel, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(sim_report(result))
        return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: all CLI tests pass (3 new).

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest`
Expected: everything passes.

- [ ] **Step 6: Commit**

```bash
git add src/psu/cli.py tests/test_cli.py
git commit -m "feat(cli): add psu simulate command"
```

---

### Task 7: Real run, tau check, README, PR (orchestrator)

**Files:**
- Modify: `README.md` (add a "Phase 4: season simulator" section after the Phase 3 section)

- [ ] **Step 1: Time the full run on the real database**

```bash
.venv/Scripts/python -c "import time, subprocess; t=time.time(); r=subprocess.run(['.venv/Scripts/psu','simulate']); print('seconds', round(time.time()-t,1), 'exit', r.returncode)"
```

Expected: exit code 0, well under 60 seconds. It should print Penn State's table: 3-0 so far, 12 regular-season games, so win totals run from 3 to 12. Record the headline numbers for the README.

- [ ] **Step 2: Sanity-check tau against 2025 residuals**

This is rough: 2025 predictions are in-sample for the final model, so the implied tau is understated.

```bash
.venv/Scripts/python - <<'EOF'
import duckdb, json
sigma = json.load(open("data/reports/game_model.json"))["sigma"]
con = duckdb.connect("data/psu.duckdb", read_only=True)
df = con.execute("""
  WITH r AS (
    SELECT home_team AS team, margin - pred_margin AS resid FROM game_predictions WHERE season = 2025 AND margin IS NOT NULL
    UNION ALL
    SELECT away_team, pred_margin - margin FROM game_predictions WHERE season = 2025 AND margin IS NOT NULL)
  SELECT team, avg(resid) AS m, count(*) AS n FROM r GROUP BY team HAVING n >= 8""").df()
n = df["n"].mean(); v = df["m"].var()
tau2 = (v - sigma**2 / n) / (1 - 1 / n)
print(f"sigma {sigma:.2f}  var(team mean resid) {v:.2f}  n {n:.1f}  implied tau {max(tau2, 0) ** 0.5:.2f}")
EOF
```

Then run `psu simulate --tau 3` and `psu simulate --tau 7` and note how much P(10+) and P(CFP) move. If the implied tau is far from 5 (below 2 or above 8), stop and report to the user before changing the default. Re-run `psu simulate` with the defaults at the end so the stored tables use them.

- [ ] **Step 3: README section**

Add after the Phase 3 section, using the real numbers from Step 1:

````markdown
## Phase 4: season simulator

```
.venv\Scripts\psu simulate                 # 10,000 seasons, seed 0, tau 5
.venv\Scripts\psu simulate --sims 50000 --seed 1 --tau 4
```

Run `psu train` first: the simulator reads `game_predictions` and the model's sigma from
`data/reports/game_model.json`. It makes no API calls.

How it works:
- Played games use their actual results. Every remaining Big Ten game and every remaining Penn State
  game is simulated from the model's predicted margin.
- Each simulated season gives every team a season-long strength draw (spread `--tau` points). A team
  that runs hot does so in all its games. Game noise is sized so each game still has the model's
  win probability.
- Big Ten standings: conference win %, then head-to-head among tied teams (only if they all played
  each other), then a coin flip. The top two meet at a neutral site, rated from power ratings fitted
  to the model's predictions.
- CFP (rough): in if Big Ten champion, or 2 or fewer losses including a title-game loss.

Outputs (replaced each run): `sim_team_summary`, `sim_win_totals`, `sim_conference`, plus
`data/reports/season_sim.md`.

<headline table from the real run, and the as-of date>
````

- [ ] **Step 4: Commit, push, PR**

```bash
git add README.md
git commit -m "docs: add Phase 4 season simulator guide to README"
git push -u origin feat/phase4-season-sim
```

Open a PR into `main` with `C:\Program Files\GitHub CLI\gh.exe`. The body should include a summary, the headline odds, the tau check result, and the test plan. Don't merge; stop for the user's review.
