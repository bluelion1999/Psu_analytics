"""Shared helpers for dashboard data: the database path, read-only access, table checks and a team's view of games."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from psu import config

SOURCES = {
    "games": "psu ingest", "plays": "psu ingest", "drives": "psu ingest", "team_game_stats": "psu ingest",
    "player_game_stats": "psu ingest",
    "plays_enriched": "psu build", "team_offense": "psu build", "team_defense": "psu build",
    "team_havoc": "psu build", "team_turnovers": "psu build",
    "game_predictions": "psu train",
    "sim_team_summary": "psu simulate", "sim_win_totals": "psu simulate", "sim_conference": "psu simulate",
}


class MissingData(RuntimeError):
    """Data the dashboard needs is missing or unreadable; the message says what to run."""


def db_path() -> Path:
    override = os.environ.get("PSU_DB_PATH")
    return Path(override) if override else config.load_settings().db_path


def read(fn: Callable[..., Any], *args: Any, path: Path | None = None) -> Any:
    """Call fn(con, *args) on a fresh read-only connection, closing it afterwards."""
    path = Path(path) if path is not None else db_path()
    if not path.exists():
        raise MissingData(f"No database at {path}; run `psu ingest` first")
    try:
        con = duckdb.connect(str(path), read_only=True)
    except duckdb.Error as e:
        raise MissingData(
            f"Could not open {path} ({type(e).__name__}); another psu command may be writing to it. "
            "Try again in a moment."
        ) from e
    try:
        return fn(con, *args)
    except duckdb.InvalidInputException as e:
        raise MissingData(f"The dashboard only reads the database: {e}") from e
    except duckdb.CatalogException as e:
        raise MissingData(f"The database schema is out of date ({e}); run `psu build` to update it") from e
    finally:
        con.close()


def has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0] > 0


def require(con: duckdb.DuckDBPyConnection, *tables: str) -> None:
    missing = [t for t in tables if not has_table(con, t)]
    if missing:
        commands = " and ".join(f"`{c}`" for c in sorted({SOURCES.get(t, "psu build") for t in missing}))
        raise MissingData(f"Missing {', '.join(missing)}; run {commands} first")


def seasons(con: duckdb.DuckDBPyConnection) -> list[int]:
    require(con, "games")
    return [int(s) for s in con.execute("SELECT DISTINCT season FROM games ORDER BY season DESC").df()["season"]]


TEAM_GAMES_SQL = """
SELECT id AS game_id, week, season_type, start_date, completed, coalesce(conference_game, false) AS conference_game,
       CASE WHEN neutral_site THEN 'neutral' WHEN home_team = $team THEN 'home' ELSE 'away' END AS venue,
       CASE WHEN home_team = $team THEN away_team ELSE home_team END AS opponent,
       CASE WHEN home_team = $team THEN home_points ELSE away_points END AS team_points,
       CASE WHEN home_team = $team THEN away_points ELSE home_points END AS opp_points,
       home_team = $team AS is_home
FROM games
WHERE season = $season AND (home_team = $team OR away_team = $team) AND season_type IN ('regular', 'postseason')
ORDER BY start_date, week
"""


def team_games(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    require(con, "games")
    df = con.execute(TEAM_GAMES_SQL, {"season": season, "team": team}).df()
    # DuckDB hands back nullable integers (pd.NA) for an all-null column; floats keep comparisons simple.
    df[["team_points", "opp_points"]] = df[["team_points", "opp_points"]].astype(float)
    df["completed"] = (
        df["completed"].fillna(False).astype(bool) & df["team_points"].notna() & df["opp_points"].notna()
    )
    df["conference_game"] = df["conference_game"].astype(bool)
    df["is_home"] = df["is_home"].astype(bool)
    df["result"] = pd.Series(  # object dtype keeps None (pandas would otherwise turn it into NaN)
        np.where(df["completed"], np.where(df["team_points"] > df["opp_points"], "W", "L"), None),
        index=df.index, dtype=object,
    )
    return df


def team_conference(con: duckdb.DuckDBPyConnection, season: int, team: str) -> str | None:
    require(con, "games")
    row = con.execute(
        """
        SELECT conf FROM (
            SELECT home_conference AS conf FROM games WHERE season = $season AND home_team = $team
            UNION ALL
            SELECT away_conference FROM games WHERE season = $season AND away_team = $team
        ) WHERE conf IS NOT NULL GROUP BY conf ORDER BY count(*) DESC, conf LIMIT 1
        """,
        {"season": season, "team": team},
    ).fetchone()
    return None if row is None else row[0]


def conference_members(con: duckdb.DuckDBPyConnection, season: int, conference: str) -> list[str]:
    require(con, "games")
    return con.execute(
        """
        SELECT DISTINCT team FROM (
            SELECT home_team AS team FROM games WHERE season = $season AND home_conference = $conf
            UNION
            SELECT away_team FROM games WHERE season = $season AND away_conference = $conf
        ) ORDER BY team
        """,
        {"season": season, "conf": conference},
    ).df()["team"].tolist()


def benchmark(
    con: duckdb.DuckDBPyConnection, season: int, team: str, table: str, column: str, members: set[str]
) -> dict[str, float]:
    """A metric for the team next to its conference average and the FBS average (the tables are FBS-only)."""
    df = con.execute(f'SELECT team, "{column}" AS value FROM "{table}" WHERE season = ?', [season]).df()
    mine = df.loc[df["team"] == team, "value"]
    return {
        "value": float(mine.iloc[0]) if len(mine) else float("nan"),
        "conference_avg": float(df.loc[df["team"].isin(members), "value"].mean()),
        "national_avg": float(df["value"].mean()),
    }
