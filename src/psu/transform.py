"""Turn raw play-by-play into analysis-ready scrimmage plays.

Adds the rush/pass class, success, explosive, turnover and garbage-time flags,
plus game context (score state, quarter, venue) used by metrics and adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RUSH_TYPES = frozenset({"Rush", "Rushing Touchdown"})
PASS_TYPES = frozenset({
    "Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
})
FUMBLE_TYPES = frozenset({"Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown"})
SCRIMMAGE_TYPES = RUSH_TYPES | PASS_TYPES | FUMBLE_TYPES | {"Safety"}
TURNOVER_TYPES = frozenset({
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Opponent)", "Fumble Return Touchdown",
})
OFFENSIVE_TD_TYPES = frozenset({"Rushing Touchdown", "Passing Touchdown"})
SCORE_STATES = ("down 9+", "down 1-8", "tied", "up 1-8", "up 9+")

ENRICHED_COLUMNS = [
    "id", "game_id", "drive_id", "season", "week", "season_type",
    "offense", "offense_conference", "defense", "defense_conference",
    "period", "quarter", "down", "distance", "yards_to_goal", "yards_gained", "play_type", "ppa",
    "play_class", "success", "explosive", "turnover", "margin", "score_state", "garbage", "venue",
]


@dataclass(frozen=True)
class GarbageTime:
    """A play is garbage time when |score margin| is greater than its quarter's threshold. None = never."""

    q1: int | None = None
    q2: int | None = 38
    q3: int | None = 28
    q4: int | None = 22

    @classmethod
    def parse(cls, spec: str) -> GarbageTime:
        """Parse "38,28,22" (Q2,Q3,Q4 margins) or "off"."""
        if spec.strip().lower() == "off":
            return cls(None, None, None, None)
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected Q2,Q3,Q4 margins like '38,28,22' or 'off', got {spec!r}")
        q2, q3, q4 = (int(p) for p in parts)
        return cls(None, q2, q3, q4)


@dataclass(frozen=True)
class Explosive:
    rush_yards: int = 12
    pass_yards: int = 16


def success_threshold(down, distance) -> np.ndarray:
    down = np.asarray(down)
    fraction = np.select([down == 1, down == 2], [0.5, 0.7], default=1.0)
    return fraction * np.asarray(distance, dtype=float)


def is_success(plays: pd.DataFrame) -> pd.Series:
    """>=50% of yards to go on 1st down, >=70% on 2nd, 100% on 3rd/4th.

    Offensive touchdowns always succeed; turnovers never do.
    """
    gained = plays["yards_gained"] >= success_threshold(plays["down"], plays["distance"])
    touchdown = plays["play_type"].isin(OFFENSIVE_TD_TYPES)
    return (gained | touchdown) & ~plays["play_type"].isin(TURNOVER_TYPES)


def is_garbage(period: pd.Series, margin: pd.Series, garbage: GarbageTime) -> pd.Series:
    limits = pd.to_numeric(
        period.map({1: garbage.q1, 2: garbage.q2, 3: garbage.q3, 4: garbage.q4}), errors="coerce"
    )
    # NaN limit (overtime, or a quarter with no threshold) compares False: never garbage time.
    return (margin.abs() > limits).astype(bool)


def score_state(margin: pd.Series) -> pd.Series:
    buckets = np.select([margin <= -9, margin < 0, margin == 0, margin < 9], list(SCORE_STATES[:4]), default=SCORE_STATES[4])
    return pd.Series(buckets, index=margin.index)


def play_class(play_type: pd.Series, play_text: pd.Series) -> pd.Series:
    """Rush or pass. Fumbles and safeties are classed by their play text."""
    text = play_text.fillna("").str.lower()
    ambiguous = ~play_type.isin(RUSH_TYPES | PASS_TYPES)
    passing = play_type.isin(PASS_TYPES) | (ambiguous & (text.str.contains("pass") | text.str.contains("sack")))
    return pd.Series(np.where(passing, "pass", "rush"), index=play_type.index)


def enrich_plays(
    plays: pd.DataFrame,
    games: pd.DataFrame,
    *,
    garbage: GarbageTime = GarbageTime(),
    explosive: Explosive = Explosive(),
) -> pd.DataFrame:
    df = plays[plays["play_type"].isin(SCRIMMAGE_TYPES) & plays["ppa"].notna()].copy()
    df["play_class"] = play_class(df["play_type"], df["play_text"])
    df["success"] = is_success(df)
    df["turnover"] = df["play_type"].isin(TURNOVER_TYPES)
    long_enough = np.where(df["play_class"] == "rush", explosive.rush_yards, explosive.pass_yards)
    df["explosive"] = (df["yards_gained"] >= long_enough) & ~df["turnover"]
    df["margin"] = df["offense_score"] - df["defense_score"]
    df["garbage"] = is_garbage(df["period"], df["margin"], garbage)
    df["score_state"] = score_state(df["margin"])
    df["quarter"] = np.where(df["period"] > 4, "OT", df["period"].astype(str))
    neutral = df["game_id"].map(games.set_index("id")["neutral_site"]).eq(True)
    df["venue"] = np.where(neutral, "neutral", np.where(df["offense"] == df["home"], "home", "away"))
    return df[ENRICHED_COLUMNS].reset_index(drop=True)
