"""Predictions data: upcoming games with model and Vegas lines, the season simulation, and the next FBS slate."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from psu.dashboard.common import MissingData, has_table, require
from psu.dashboard.overview import schedule


def upcoming(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    games = schedule(con, season, team)
    return games[~games["completed"]].reset_index(drop=True)


def sim_summary(con: duckdb.DuckDBPyConnection, team: str) -> dict[str, Any]:
    require(con, "sim_team_summary")
    df = con.execute("SELECT * FROM sim_team_summary WHERE team = ?", [team]).df()
    if df.empty:
        raise MissingData(f"No season simulation for {team}; run `psu simulate` first")
    return df.iloc[0].to_dict()


def sim_win_totals(con: duckdb.DuckDBPyConnection, team: str) -> pd.DataFrame:
    require(con, "sim_win_totals")
    return con.execute("SELECT wins, prob FROM sim_win_totals WHERE team = ? ORDER BY wins", [team]).df()


def sim_conference(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    require(con, "sim_conference")
    return con.execute(
        "SELECT team, mean_conf_wins, p_title_game, p_conf_champ FROM sim_conference "
        "ORDER BY p_conf_champ DESC, p_title_game DESC, team"
    ).df()


HISTORY_VIEW_COLUMNS = ["as_of_slate", "p_title_game", "p_conf_champ", "p_cfp", "backfilled"]


def sim_history(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    """Season-outlook odds by week; empty (not an error) before the first `psu simulate` that records history."""
    if not has_table(con, "sim_history"):
        return pd.DataFrame(columns=HISTORY_VIEW_COLUMNS)
    return con.execute(
        f"SELECT {', '.join(HISTORY_VIEW_COLUMNS)} FROM sim_history WHERE season = ? AND team = ? ORDER BY as_of_slate",
        [season, team],
    ).df()


def next_slate(con: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
    require(con, "game_predictions")
    return con.execute(
        """
        SELECT week, start_date, away_team, home_team, neutral_site, pred_margin, home_win_prob, vegas_margin
        FROM game_predictions
        WHERE season = $season AND split = 'upcoming' AND season_type = 'regular'
          AND week = (SELECT min(week) FROM game_predictions
                      WHERE season = $season AND split = 'upcoming' AND season_type = 'regular')
        ORDER BY start_date, home_team
        """,
        {"season": season},
    ).df()
