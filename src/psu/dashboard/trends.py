"""Efficiency trends data: EPA/play and success rate by season, and game by game within a season."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from psu.dashboard.common import benchmark, conference_members, require, seasons, team_conference, team_games

TREND_METRICS = [  # (label, table, column)
    ("Offense EPA/play", "team_offense", "epa_per_play"),
    ("Defense EPA/play", "team_defense", "epa_per_play"),
    ("Offense success rate", "team_offense", "success_rate"),
    ("Defense success rate", "team_defense", "success_rate"),
]
WEEKLY_COLUMNS = ["game", "start_date", "week", "opponent", "side", "epa_per_play", "success_rate", "plays"]
WEEKLY_SQL = """
SELECT game_id,
       CASE WHEN offense = $team THEN 'offense' ELSE 'defense' END AS side,
       avg(ppa) AS epa_per_play,
       avg(CAST(success AS DOUBLE)) AS success_rate,
       count(*) AS plays
FROM plays_enriched
WHERE season = $season AND (offense = $team OR defense = $team)
  AND NOT coalesce(garbage, false) AND ppa IS NOT NULL
GROUP BY ALL
"""


def season_trends(con: duckdb.DuckDBPyConnection, team: str) -> pd.DataFrame:
    require(con, "games", "team_offense", "team_defense")
    rows = []
    for season in sorted(seasons(con)):
        conference = team_conference(con, season, team)
        members = set(conference_members(con, season, conference)) if conference else set()
        for label, table, column in TREND_METRICS:
            b = benchmark(con, season, team, table, column, members)
            if np.isnan(b["national_avg"]):
                continue  # no metric rows for this season yet
            if not np.isnan(b["value"]):
                rows.append({"season": season, "metric": label, "group": team, "value": b["value"]})
            if members:
                rows.append(
                    {"season": season, "metric": label, "group": f"{conference} avg", "value": b["conference_avg"]}
                )
            rows.append({"season": season, "metric": label, "group": "FBS avg", "value": b["national_avg"]})
    return pd.DataFrame(rows, columns=["season", "metric", "group", "value"])


def weekly_trends(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    require(con, "plays_enriched", "games")
    stats = con.execute(WEEKLY_SQL, {"season": season, "team": team}).df()
    games = team_games(con, season, team)[["game_id", "week", "season_type", "start_date", "opponent"]]
    df = stats.merge(games, on="game_id").sort_values(["start_date", "side"], ascending=[True, False])
    if df.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    prefix = pd.Series(
        np.where(df["season_type"] == "postseason", "Bowl", "Wk " + df["week"].astype(str)), index=df.index
    )
    df["game"] = prefix + " " + df["opponent"]
    return df[WEEKLY_COLUMNS].reset_index(drop=True)
