"""Season overview data: record, schedule with results and pregame lines, and metrics against conference and FBS."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from psu.dashboard.common import benchmark, conference_members, has_table, require, team_conference, team_games

SCHEDULE_COLUMNS = [
    "game_id",
    "week",
    "season_type",
    "start_date",
    "opponent",
    "venue",
    "completed",
    "result",
    "team_points",
    "opp_points",
    "vegas_margin",
    "model_margin",
    "win_prob",
    "prediction",
]
METRICS = [  # (label, table, column, higher_is_better)
    ("Offense EPA/play", "team_offense", "epa_per_play", True),
    ("Offense success rate", "team_offense", "success_rate", True),
    ("Offense explosiveness", "team_offense", "explosiveness", True),
    ("Defense EPA/play", "team_defense", "epa_per_play", False),
    ("Defense success rate", "team_defense", "success_rate", False),
    ("Havoc rate", "team_havoc", "havoc_rate", True),
    ("Turnover margin", "team_turnovers", "margin", True),
]


def record(con: duckdb.DuckDBPyConnection, season: int, team: str) -> dict[str, int]:
    games = team_games(con, season, team)
    done = games[games["completed"]]
    conf = done[done["conference_game"]]
    return {
        "wins": int((done["result"] == "W").sum()),
        "losses": int((done["result"] == "L").sum()),
        "conf_wins": int((conf["result"] == "W").sum()),
        "conf_losses": int((conf["result"] == "L").sum()),
        "remaining": int((~games["completed"]).sum()),
    }


def schedule(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    games = team_games(con, season, team)
    if has_table(con, "game_predictions"):
        preds = con.execute(
            "SELECT game_id, pred_margin, vegas_margin, home_win_prob, split FROM game_predictions WHERE season = ?",
            [season],
        ).df()
    else:
        preds = pd.DataFrame(
            {
                "game_id": pd.Series(dtype="int64"),
                "pred_margin": pd.Series(dtype=float),
                "vegas_margin": pd.Series(dtype=float),
                "home_win_prob": pd.Series(dtype=float),
                "split": pd.Series(dtype=object),
            }
        )
    df = games.merge(preds, on="game_id", how="left")
    sign = np.where(df["is_home"], 1.0, -1.0)
    home_prob = df["home_win_prob"].astype(float)
    df["model_margin"] = sign * df["pred_margin"].astype(float)
    df["vegas_margin"] = sign * df["vegas_margin"].astype(float)
    df["win_prob"] = np.where(df["is_home"], home_prob, 1.0 - home_prob)
    df["prediction"] = df["split"]
    return df[SCHEDULE_COLUMNS]


def metric_comparison(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    require(con, *sorted({table for _, table, _, _ in METRICS}))
    conference = team_conference(con, season, team)
    members = set(conference_members(con, season, conference)) if conference else set()
    rows = [
        {"metric": label, **benchmark(con, season, team, table, column, members), "higher_is_better": higher}
        for label, table, column, higher in METRICS
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "conference_avg", "national_avg", "higher_is_better"])
