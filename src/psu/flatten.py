"""Turn raw CFBD JSON (camelCase, nested) into flat snake_case DataFrames, one shape per table."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

Records = list[dict[str, Any]]
Flattener = Callable[[Records, dict[str, Any]], pd.DataFrame]


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _normalize(records: Records) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records, sep=".")
    df.columns = ["_".join(snake(part) for part in col.split(".")) for col in df.columns]
    return df


def _add_context(df: pd.DataFrame, params: dict[str, Any], *, with_week: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["season"] = params["year"]
    if with_week:
        df["week"] = params["week"]
        df["season_type"] = params["season_type"]
    return df


def flatten_games(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    df = _normalize(data)
    for col in ("home_line_scores", "away_line_scores"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, list) else None)
    return df


def flatten_plays(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    return _add_context(_normalize(data), params)


def flatten_drives(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    return _add_context(_normalize(data), params, with_week=False)


def flatten_team_game_stats(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "game_id": game.get("id"),
            "team_id": team.get("teamId"),
            "team": team.get("team"),
            "conference": team.get("conference"),
            "home_away": team.get("homeAway"),
            "points": team.get("points"),
            "category": stat.get("category"),
            "stat": stat.get("stat"),
        }
        for game in data
        for team in game.get("teams") or []
        for stat in team.get("stats") or []
    ]
    return _add_context(pd.DataFrame(rows), params)


def flatten_player_game_stats(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "game_id": game.get("id"),
            "team": team.get("team"),
            "conference": team.get("conference"),
            "home_away": team.get("homeAway"),
            "category": category.get("name"),
            "stat_type": stat_type.get("name"),
            "athlete_id": athlete.get("id"),
            "athlete_name": athlete.get("name"),
            "stat": athlete.get("stat"),
        }
        for game in data
        for team in game.get("teams") or []
        for category in team.get("categories") or []
        for stat_type in category.get("types") or []
        for athlete in stat_type.get("athletes") or []
    ]
    return _add_context(pd.DataFrame(rows), params)


def flatten_lines(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for game in data:
        base = {snake(k): v for k, v in game.items() if k not in ("id", "lines")}
        base["game_id"] = game.get("id")
        for line in game.get("lines") or []:
            rows.append({**base, **{snake(k): v for k, v in line.items()}})
    return pd.DataFrame(rows)


def flatten_wide(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    return _normalize(data)


FLATTENERS: dict[str, Flattener] = {
    "games": flatten_games,
    "plays": flatten_plays,
    "drives": flatten_drives,
    "team_game_stats": flatten_team_game_stats,
    "player_game_stats": flatten_player_game_stats,
    "advanced_season": flatten_wide,
    "ratings_sp": flatten_wide,
    "talent": flatten_wide,
    "recruiting": flatten_wide,
    "returning_production": flatten_wide,
    "lines": flatten_lines,
}
