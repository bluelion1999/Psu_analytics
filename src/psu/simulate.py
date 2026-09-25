"""Season simulator: Monte Carlo over the rest of the current season using the Phase 3 game model."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from psu.build import _write
from psu.config import TEAM
from psu.models.game_predict import PHASES, phase_of
from psu.sim.ratings import fit_ratings
from psu.sim.season import draw_margins, draw_matchups, draw_strengths, game_noise_sd
from psu.sim.standings import top_two

log = logging.getLogger(__name__)

CONFERENCE = "Big Ten"
STALE_AFTER_DAYS = 3  # a regular-season game this long past kickoff without `completed` is treated as finished


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
    as_of_slate: int = 1


def makes_cfp(losses, champion, max_losses: int = 2) -> np.ndarray:
    return np.asarray(champion, dtype=bool) | (np.asarray(losses) <= max_losses)


def _naive(ts):
    """Strip tz info (assumed UTC) so tz-aware and naive timestamps can be compared safely."""
    if isinstance(ts, pd.Timestamp):
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    ts = pd.to_datetime(ts)
    return ts.dt.tz_localize(None) if ts.dt.tz is not None else ts


def current_slate(games: pd.DataFrame, season: int, *, now: pd.Timestamp | None = None) -> int:
    """First regular-season week with an unfinished game (one past the last week once all are played).

    A game counts as finished if `completed` is true, or if its kickoff was more than
    STALE_AFTER_DAYS days ago -- this keeps a cancelled game that never gets marked completed
    (e.g. 2024 week 5 App State vs Liberty) from freezing the as-of week forever.
    """
    now = _naive(pd.Timestamp.now() if now is None else pd.Timestamp(now))
    regular = games[(games["season"] == season) & (games["season_type"] == "regular")]
    completed = regular["completed"].fillna(False).astype(bool)
    stale = _naive(regular["start_date"]) < (now - pd.Timedelta(days=STALE_AFTER_DAYS))
    open_weeks = regular.loc[~(completed | stale), "week"]
    if len(open_weeks):
        return int(open_weeks.min())
    return int(regular["week"].max()) + 1 if len(regular) else 1


def run_simulation(
    games: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    season: int,
    sigma: float | Mapping[str, float],
    team: str = TEAM,
    n_sims: int = 10_000,
    seed: int = 0,
    tau: float = 5.0,
    cfp_max_losses: int = 2,
    unrated_win_prob: float = 0.95,
    conference: str = CONFERENCE,
    now: pd.Timestamp | None = None,
) -> SimResult:
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    sigmas = {p: float(sigma[p]) for p in PHASES} if isinstance(sigma, Mapping) else dict.fromkeys(PHASES, float(sigma))
    for value in sigmas.values():
        game_noise_sd(value, tau)  # fail fast on a bad tau, before any draw
    rng = np.random.default_rng(seed)
    season_games = games[games["season"] == season]
    regular_all = season_games[season_games["season_type"] == "regular"]
    members = sorted(
        set(regular_all.loc[regular_all["home_conference"] == conference, "home_team"])
        | set(regular_all.loc[regular_all["away_conference"] == conference, "away_team"])
    )
    if len(members) < 2:
        raise ValueError(f"fewer than two {conference} teams in {season}")
    if not ((regular_all["home_team"] == team) | (regular_all["away_team"] == team)).any():
        raise ValueError(f"{team} has no {season} regular-season games")

    notes = regular_all["notes"] if "notes" in regular_all.columns else pd.Series(index=regular_all.index, dtype=object)
    notes = notes.fillna("").astype(str)
    is_title = (
        regular_all["home_team"].isin(members)
        & regular_all["away_team"].isin(members)
        & notes.str.contains(f"{conference} Championship", case=False, regex=False)
    )
    title_rows = regular_all[is_title]
    title_match = title_rows.iloc[0] if len(title_rows) else None
    regular = regular_all[~is_title]

    is_conf = regular["home_team"].isin(members) & regular["away_team"].isin(members)
    is_team = (regular["home_team"] == team) | (regular["away_team"] == team)
    preds = predictions[["game_id", "pred_margin"]].rename(columns={"game_id": "id"})
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
        home_win[:, predicted] = margins > 0
    if unrated.any():
        log.warning(
            "%d %s game(s) have no prediction; counting each as a win with probability %.2f",
            int(unrated.sum()),
            team,
            unrated_win_prob,
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

    if title_match is not None:
        # A real conference title game: the pair (and, if it's already been played, the champion) is fixed.
        first = np.full(n_sims, m_index[title_match["home_team"]], dtype=int)
        second = np.full(n_sims, m_index[title_match["away_team"]], dtype=int)
    else:
        first = np.empty(n_sims, dtype=int)
        second = np.empty(n_sims, dtype=int)
        for i in range(n_sims):
            hw = conf_home_win[i]
            h2h_wins = np.zeros((n_members, n_members), dtype=int)
            np.add.at(h2h_wins, (ch[hw], ca[hw]), 1)
            np.add.at(h2h_wins, (ca[~hw], ch[~hw]), 1)
            first[i], second[i] = top_two(conf_wins[i], conf_games, h2h_wins, h2h_games, rng)

    title_completed = (
        title_match is not None
        and bool(title_match["completed"])
        and pd.notna(title_match["home_points"])
        and pd.notna(title_match["away_points"])
    )
    if title_completed:
        winner = (
            m_index[title_match["home_team"]]
            if title_match["home_points"] > title_match["away_points"]
            else m_index[title_match["away_team"]]
        )
        champion = np.full(n_sims, winner, dtype=int)
    else:
        ratings = fit_ratings(predictions)
        member_rating = np.array([ratings.rating.get(t, 0.0) for t in members])
        member_team = np.array([index[t] for t in members])
        title_margin = draw_matchups(
            member_rating[first] - member_rating[second],
            member_team[first],
            member_team[second],
            strengths,
            sigma=sigmas["mid"],
            tau=tau,
            rng=rng,
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
        win_totals=pd.DataFrame(
            {
                "wins": np.arange(n_games + 1),
                "prob": np.bincount(wins, minlength=n_games + 1) / n_sims,
            }
        ),
        conference=pd.DataFrame(
            {
                "team": members,
                "mean_conf_wins": conf_wins.mean(axis=0),
                "p_title_game": [float(((first == m) | (second == m)).mean()) for m in range(n_members)],
                "p_conf_champ": np.bincount(champion, minlength=n_members) / n_sims,
            }
        ),
        as_of_slate=current_slate(games, season, now=now),
    )


GAME_COLUMNS = (
    "id, season, week, season_type, start_date, completed, home_team, away_team, "
    "home_conference, away_conference, home_points, away_points, notes"
)
SUMMARY_COLUMNS = [
    "season",
    "team",
    "n_sims",
    "seed",
    "tau",
    "as_of",
    "mean_wins",
    "p_10_plus",
    "p_title_game",
    "p_conf_champ",
    "p_cfp",
]
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


def load_inputs(con: duckdb.DuckDBPyConnection, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = set(con.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
    if "game_predictions" not in tables:
        raise MissingModel("game_predictions table not found; run `psu train` first")
    games = con.execute(f"SELECT {GAME_COLUMNS} FROM games WHERE season = ?", [season]).df()
    predictions = con.execute(
        "SELECT game_id, home_team, away_team, neutral_site, pred_margin FROM game_predictions WHERE season = ?",
        [season],
    ).df()
    return games, predictions


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


def simulate_season(
    con: duckdb.DuckDBPyConnection,
    *,
    season: int,
    sigma: float | Mapping[str, float],
    team: str = TEAM,
    n_sims: int = 10_000,
    seed: int = 0,
    tau: float = 5.0,
    cfp_max_losses: int = 2,
) -> SimResult:
    games, predictions = load_inputs(con, season)
    return run_simulation(
        games,
        predictions,
        season=season,
        sigma=sigma,
        team=team,
        n_sims=n_sims,
        seed=seed,
        tau=tau,
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
    lines += [f"| {w} | {p:.1%} |" for w, p in zip(result.win_totals["wins"], result.win_totals["prob"], strict=True)]
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
    summary = pd.DataFrame(
        [
            {
                "season": result.season,
                "team": result.team,
                "n_sims": result.n_sims,
                "seed": result.seed,
                "tau": result.tau,
                "as_of": pd.NaT if result.as_of is None else result.as_of,
                "mean_wins": result.mean_wins,
                "p_10_plus": result.p_10_plus,
                "p_title_game": result.p_title_game,
                "p_conf_champ": result.p_conf_champ,
                "p_cfp": result.p_cfp,
            }
        ]
    )[SUMMARY_COLUMNS]
    con.execute("BEGIN TRANSACTION")
    try:
        _write(con, "sim_team_summary", summary)
        _write(
            con,
            "sim_win_totals",
            result.win_totals.assign(season=result.season, team=result.team)[["season", "team", "wins", "prob"]],
        )
        _write(
            con,
            "sim_conference",
            result.conference.assign(season=result.season)[
                ["season", "team", "mean_conf_wins", "p_title_game", "p_conf_champ"]
            ],
        )
        write_history(con, history_rows([result], backfilled=False))
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
    reports = Path(out_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "season_sim.md").write_text(report_markdown(result), encoding="utf-8")
