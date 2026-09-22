"""Team efficiency metrics from enriched plays and box scores. Every function returns a tidy DataFrame."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

SIDES = ("offense", "defense")
HAVOC_CATEGORIES = ("tacklesForLoss", "passesDeflected", "passesIntercepted")


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be 'offense' or 'defense', got {side!r}")


def _masked_mean(df: pd.DataFrame, keys: list[str], value: str, mask: pd.Series) -> pd.Series:
    """Group mean of `value` over rows where mask is True; groups with no such rows get NaN."""
    return df[value].astype(float).where(mask).groupby([df[k] for k in keys]).mean()


def efficiency(
    enriched: pd.DataFrame,
    side: str = "offense",
    by: Sequence[str] = ("season",),
    exclude_garbage: bool = True,
) -> pd.DataFrame:
    _check_side(side)
    df = enriched[~enriched["garbage"]] if exclude_garbage else enriched
    keys = [side, *by]
    grouped = df.groupby(keys)
    rush = df["play_class"] == "rush"
    out = pd.DataFrame({
        "plays": grouped.size(),
        "epa_per_play": grouped["ppa"].mean(),
        "rush_epa": _masked_mean(df, keys, "ppa", rush),
        "pass_epa": _masked_mean(df, keys, "ppa", ~rush),
        "success_rate": grouped["success"].mean(),
        "rush_success_rate": _masked_mean(df, keys, "success", rush),
        "pass_success_rate": _masked_mean(df, keys, "success", ~rush),
        "explosiveness": _masked_mean(df, keys, "ppa", df["success"]),
        "explosive_rate": grouped["explosive"].mean(),
        "third_down_rate": _masked_mean(df, keys, "success", df["down"] == 3),
        "turnover_rate": grouped["turnover"].mean(),
    })
    return out.reset_index().rename(columns={side: "team"})


def _box_wide(team_game_stats: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """One row per (game, team) with a numeric column per category; missing categories count as 0."""
    teams = team_game_stats[["game_id", "season", "team"]].drop_duplicates()
    rows = team_game_stats[team_game_stats["category"].isin(categories)]
    values = rows.assign(value=pd.to_numeric(rows["stat"], errors="coerce")).pivot_table(
        index=["game_id", "season", "team"], columns="category", values="value", aggfunc="sum"
    )
    values = values.reindex(columns=list(categories))
    values.columns.name = None
    wide = teams.merge(values.reset_index(), on=["game_id", "season", "team"], how="left")
    wide[list(categories)] = wide[list(categories)].fillna(0.0)
    return wide


def _with_opponent(wide: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    opponent = wide[["game_id", "team", *columns]].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in columns}}
    )
    merged = wide.merge(opponent, on="game_id")
    return merged[merged["team"] != merged["opponent"]]


def havoc_rate(team_game_stats: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """(TFL + passes defended + interceptions + opponent fumbles) / defensive scrimmage plays, per season.

    Uses all scrimmage plays (garbage time included), matching CFBD's havoc rate.
    """
    wide = _with_opponent(_box_wide(team_game_stats, (*HAVOC_CATEGORIES, "totalFumbles")), ["totalFumbles"])
    wide["havoc_events"] = wide[list(HAVOC_CATEGORIES)].sum(axis=1) + wide["opp_totalFumbles"]
    events = wide.groupby(["season", "team"])["havoc_events"].sum()
    plays = enriched.groupby(["season", "defense"]).size().rename_axis(["season", "team"])
    out = pd.concat({"havoc_events": events, "defensive_plays": plays}, axis=1).dropna()
    out["havoc_rate"] = out["havoc_events"] / out["defensive_plays"]
    return out.reset_index()


def turnover_margin(team_game_stats: pd.DataFrame) -> pd.DataFrame:
    wide = _with_opponent(_box_wide(team_game_stats, ("turnovers",)), ["turnovers"])
    grouped = wide.groupby(["season", "team"])
    out = pd.DataFrame({
        "games": grouped.size(),
        "giveaways": grouped["turnovers"].sum(),
        "takeaways": grouped["opp_turnovers"].sum(),
    })
    out["margin"] = out["takeaways"] - out["giveaways"]
    return out.reset_index()


def red_zone(
    enriched: pd.DataFrame,
    drives: pd.DataFrame,
    side: str = "offense",
    by: Sequence[str] = ("season",),
    line: int = 20,
) -> pd.DataFrame:
    """Drives with a scrimmage play at or inside the `line`: trips, TD rate, and points per trip."""
    _check_side(side)
    reached = enriched.loc[enriched["yards_to_goal"] <= line, "drive_id"].unique()
    trips = drives[drives["id"].isin(reached)].copy()
    trips["points"] = (trips["end_offense_score"] - trips["start_offense_score"]).clip(lower=0)
    trips["touchdown"] = trips["points"] >= 6
    grouped = trips.groupby([side, *by])
    out = pd.DataFrame({
        "trips": grouped.size(),
        "touchdowns": grouped["touchdown"].sum(),
        "points_per_trip": grouped["points"].mean(),
    })
    out["td_rate"] = out["touchdowns"] / out["trips"]
    return out.reset_index().rename(columns={side: "team"})
