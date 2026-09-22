"""Pregame features for game prediction. Every feature for a game uses only information from before its slate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from psu.adjust import opponent_adjust

MAX_REST = 21
RATING_COLUMNS = ["off_epa", "def_epa", "off_sr", "def_sr"]
_RATING_FRAME_COLUMNS = [*RATING_COLUMNS, "off_plays", "def_plays"]


def slate_index(games: pd.DataFrame) -> pd.DataFrame:
    """Regular-season week w is slate w; postseason week w follows the season's last regular week."""
    g = games[["id", "season", "week", "season_type"]].copy()
    last_regular = g[g["season_type"] == "regular"].groupby("season")["week"].max()
    offset = g["season"].map(last_regular).fillna(0)
    g["slate"] = np.where(g["season_type"] == "postseason", offset + g["week"], g["week"]).astype(int)
    return g[["id", "season", "slate"]]


def rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game in the same season, capped at MAX_REST (openers get MAX_REST)."""
    sides = []
    for side in ("home", "away"):
        part = games[["id", "season", "start_date", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        sides.append(part.assign(side=side))
    long = pd.concat(sides, ignore_index=True)
    long["start_date"] = pd.to_datetime(long["start_date"])
    long = long.sort_values("start_date")
    previous = long.groupby(["season", "team"])["start_date"].shift()
    long["rest"] = (long["start_date"] - previous).dt.days.clip(upper=MAX_REST).fillna(MAX_REST)
    wide = long.pivot(index="id", columns="side", values="rest")
    wide.columns.name = None
    return wide.rename(columns={"home": "home_rest", "away": "away_rest"}).reset_index()


def vegas_margin(lines: pd.DataFrame) -> pd.DataFrame:
    """Market-expected home margin: minus the median closing home spread across providers."""
    spreads = lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median()
    return (-spreads).rename("vegas_margin").reset_index()


def league_means(enriched: pd.DataFrame) -> dict[str, float]:
    plays = enriched[~enriched["garbage"]]
    return {"epa": float(plays["ppa"].mean()), "sr": float(plays["success"].astype(float).mean())}


def season_ratings(enriched: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Opponent-adjusted EPA and success rate per team over the given plays (one season)."""
    if enriched.empty:
        return pd.DataFrame(columns=_RATING_FRAME_COLUMNS, index=pd.Index([], name="team"), dtype=float)
    epa = opponent_adjust(enriched, "ppa", alpha=alpha).set_index("team")
    sr = opponent_adjust(enriched, "success", alpha=alpha).set_index("team")
    return pd.DataFrame({
        "off_epa": epa["off_adj"], "def_epa": epa["def_adj"],
        "off_sr": sr["off_adj"], "def_sr": sr["def_adj"],
        "off_plays": epa["off_plays"], "def_plays": epa["def_plays"],
    })


def rolling_ratings(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    *,
    alpha: float = 50.0,
    shrink_plays: int = 300,
) -> pd.DataFrame:
    """Ratings for each team as of the start of each slate, blended with last season's final ratings."""
    slates = slate_index(games)
    plays = enriched.merge(slates.rename(columns={"id": "game_id"})[["game_id", "slate"]], on="game_id")
    frames = []
    for season in sorted(slates["season"].unique()):
        previous = plays[plays["season"] == season - 1]
        prior = season_ratings(previous, alpha)
        means = league_means(previous) if not previous.empty else {"epa": np.nan, "sr": np.nan}
        this_season = plays[plays["season"] == season]
        season_games = games[games["season"] == season].merge(slates[["id", "slate"]], on="id")
        for slate, slate_games in season_games.groupby("slate"):
            teams = pd.Index(sorted(set(slate_games["home_team"]) | set(slate_games["away_team"])), name="team")
            current = season_ratings(this_season[this_season["slate"] < slate], alpha).reindex(teams)
            prior_now = prior.reindex(teams)
            out = pd.DataFrame(index=teams)
            for column in RATING_COLUMNS:
                side, kind = column.split("_")
                n = current[f"{side}_plays"].astype(float).fillna(0.0)
                weight = (n / (n + shrink_plays)).fillna(0.0)
                p = prior_now[column].astype(float).fillna(means[kind])
                c = current[column].astype(float)
                blended = weight * c.fillna(0.0) + (1 - weight) * p
                out[column] = blended.where(p.notna(), c)
            out = out.reset_index()
            out.insert(0, "slate", int(slate))
            out.insert(0, "season", int(season))
            frames.append(out)
    return pd.concat(frames, ignore_index=True)
