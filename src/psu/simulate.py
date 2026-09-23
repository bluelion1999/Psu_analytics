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
