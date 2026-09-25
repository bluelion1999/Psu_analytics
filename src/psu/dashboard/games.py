"""Game explorer data: completed games, line score, box score, drives, biggest plays and win probability."""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from psu.dashboard.common import has_table, require, team_games
from psu.dashboard.winprob import in_game_wp
from psu.models.game_predict import phase_of

BOX_STATS = [  # (team_game_stats category, label)
    ("totalYards", "Total yards"),
    ("netPassingYards", "Passing yards"),
    ("rushingYards", "Rushing yards"),
    ("yardsPerPass", "Yards per pass"),
    ("yardsPerRushAttempt", "Yards per rush"),
    ("firstDowns", "First downs"),
    ("thirdDownEff", "3rd down"),
    ("fourthDownEff", "4th down"),
    ("turnovers", "Turnovers"),
    ("totalPenaltiesYards", "Penalties-yards"),
    ("possessionTime", "Possession"),
]
REGULATION_SECONDS = 3600


def _clock(minutes: pd.Series, seconds: pd.Series) -> pd.Series:
    return minutes.fillna(0).astype(int).astype(str) + ":" + seconds.fillna(0).astype(int).astype(str).str.zfill(2)


def game_options(con: duckdb.DuckDBPyConnection, season: int, team: str) -> pd.DataFrame:
    games = team_games(con, season, team)
    done = games[games["completed"]].copy()
    if done.empty:
        return pd.DataFrame(columns=["game_id", "label"])
    week = pd.Series(
        np.where(done["season_type"] == "postseason", "Bowl", "Wk " + done["week"].astype(str)), index=done.index
    )
    where = done["venue"].map({"home": "vs", "away": "at", "neutral": "vs"})
    score = done["team_points"].astype(int).astype(str) + "-" + done["opp_points"].astype(int).astype(str)
    done["label"] = week + " " + where + " " + done["opponent"] + " (" + done["result"] + " " + score + ")"
    return done[["game_id", "label"]].reset_index(drop=True)


def _teams(con: duckdb.DuckDBPyConnection, game_id: int) -> tuple:
    row = con.execute(
        "SELECT home_team, away_team, home_points, away_points, home_line_scores, away_line_scores "
        "FROM games WHERE id = ?",
        [game_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"game {game_id} not found")
    return row


def line_scores(con: duckdb.DuckDBPyConnection, game_id: int) -> pd.DataFrame:
    require(con, "games")
    home, away, home_pts, away_pts, home_lines, away_lines = _teams(con, game_id)
    rows = []
    for team, points, raw in ((away, away_pts, away_lines), (home, home_pts, home_lines)):
        periods = json.loads(raw) if raw else []
        labels = [f"Q{i + 1}" if i < 4 else f"OT{i - 3}" for i in range(len(periods))]
        rows.append({"team": team, **dict(zip(labels, periods, strict=True)), "Total": points})
    return pd.DataFrame(rows)


def box_score(con: duckdb.DuckDBPyConnection, game_id: int) -> pd.DataFrame:
    require(con, "games", "team_game_stats")
    home, away, *_ = _teams(con, game_id)
    stats = con.execute("SELECT team, category, stat FROM team_game_stats WHERE game_id = ?", [game_id]).df()
    lookup = {(r.team, r.category): r.stat for r in stats.itertuples()}

    def cell(team: str, category: str) -> str:
        value = lookup.get((team, category))
        return "" if value is None or pd.isna(value) else str(value)

    rows = [{"stat": label, away: cell(away, category), home: cell(home, category)} for category, label in BOX_STATS]
    return pd.DataFrame(rows, columns=["stat", away, home])


def drive_chart(con: duckdb.DuckDBPyConnection, game_id: int) -> pd.DataFrame:
    require(con, "drives")
    df = con.execute(
        "SELECT drive_number, offense, start_period, start_yards_to_goal, end_yards_to_goal, plays, yards, "
        "drive_result, elapsed_minutes, elapsed_seconds FROM drives WHERE game_id = ? ORDER BY drive_number",
        [game_id],
    ).df()
    df["start_pos"] = 100 - df["start_yards_to_goal"]
    df["end_pos"] = 100 - df["end_yards_to_goal"]
    df["time"] = _clock(df["elapsed_minutes"], df["elapsed_seconds"])
    df = df.rename(columns={"start_period": "quarter", "drive_result": "result"})
    return df[["drive_number", "offense", "quarter", "start_pos", "end_pos", "plays", "yards", "result", "time"]]


def top_plays(con: duckdb.DuckDBPyConnection, game_id: int, n: int = 10) -> pd.DataFrame:
    require(con, "plays")
    df = con.execute(
        "SELECT period AS quarter, clock_minutes, clock_seconds, offense, down, distance, ppa AS epa, "
        "play_text AS text FROM plays WHERE game_id = ? AND ppa IS NOT NULL ORDER BY abs(ppa) DESC, id LIMIT ?",
        [game_id, n],
    ).df()
    df["clock"] = _clock(df["clock_minutes"], df["clock_seconds"])
    return df[["quarter", "clock", "offense", "down", "distance", "epa", "text"]]


def game_phase(con: duckdb.DuckDBPyConnection, game_id: int) -> str:
    """The model phase ("early", "mid" or "post") of a game, or "mid" if the game is not found."""
    row = con.execute("SELECT season_type, week FROM games WHERE id = ?", [game_id]).fetchone()
    if row is None:
        return "mid"
    season_type, week = row
    return str(phase_of([season_type], [week])[0])


def win_probability(con: duckdb.DuckDBPyConnection, game_id: int, sigma: float) -> pd.DataFrame:
    require(con, "games", "plays")
    home, _, home_pts, away_pts, *_ = _teams(con, game_id)
    pregame = 0.0
    if has_table(con, "game_predictions"):
        row = con.execute("SELECT pred_margin FROM game_predictions WHERE game_id = ?", [game_id]).fetchone()
        if row is not None and row[0] is not None and not pd.isna(row[0]):
            pregame = float(row[0])
    plays = con.execute(
        "SELECT period, clock_minutes, clock_seconds, offense, offense_score, defense_score FROM plays "
        "WHERE game_id = ? AND period IS NOT NULL ORDER BY drive_number, play_number, id",
        [game_id],
    ).df()
    home_margin = np.where(
        plays["offense"] == home,
        plays["offense_score"] - plays["defense_score"],
        plays["defense_score"] - plays["offense_score"],
    )
    seconds = np.where(
        plays["period"] <= 4,
        (4 - plays["period"]) * 900 + plays["clock_minutes"].fillna(0) * 60 + plays["clock_seconds"].fillna(0),
        0,
    )
    frames = [
        pd.DataFrame({"seconds_left": [REGULATION_SECONDS], "home_margin": [0]}),
        pd.DataFrame({"seconds_left": seconds, "home_margin": home_margin}),
    ]
    if home_pts is not None and away_pts is not None:
        frames.append(pd.DataFrame({"seconds_left": [0], "home_margin": [home_pts - away_pts]}))
    wp = pd.concat(frames, ignore_index=True).astype(float)
    wp["seconds_left"] = wp["seconds_left"].clip(0, REGULATION_SECONDS)
    wp["home_margin"] = wp["home_margin"].ffill()
    wp["home_wp"] = in_game_wp(wp["home_margin"], pregame, wp["seconds_left"] / REGULATION_SECONDS, sigma)
    wp["minute"] = (REGULATION_SECONDS - wp["seconds_left"]) / 60
    wp.insert(0, "play", range(len(wp)))
    return wp
