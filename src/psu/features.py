"""Pregame features for game prediction. Every feature for a game uses only information from before its slate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psu.adjust import opponent_adjust
from psu.priors import empty_returning, project_prior, projection_pairs, returning_pct, season_projections

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
    return pd.DataFrame(
        {
            "off_epa": epa["off_adj"],
            "def_epa": epa["def_adj"],
            "off_sr": sr["off_adj"],
            "def_sr": sr["def_adj"],
            "off_plays": epa["off_plays"],
            "def_plays": epa["def_plays"],
        }
    )


def rolling_ratings(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    *,
    alpha: float,
    shrink_plays: int,
    returning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ratings for each team as of the start of each slate, blended with a projected prior from last season."""
    returning = empty_returning() if returning is None else returning
    slates = slate_index(games)
    plays = enriched.merge(slates.rename(columns={"id": "game_id"})[["game_id", "slate"]], on="game_id")
    seasons = [int(s) for s in sorted(slates["season"].unique())]
    by_season = {s: plays[plays["season"] == s] for s in seasons}
    finals = {s: season_ratings(p, alpha) for s, p in by_season.items() if not p.empty}
    means = {s: league_means(p) for s, p in by_season.items() if not p.empty}
    pairs = {c: projection_pairs(finals, means, returning, c) for c in RATING_COLUMNS}
    no_means = {"epa": np.nan, "sr": np.nan}
    frames = []
    for season in seasons:
        last = finals.get(season - 1)
        season_means = means.get(season - 1, no_means)
        if last is None:
            prior = season_ratings(plays.iloc[:0], alpha)
        else:
            prior = project_prior(
                last, season_means, returning_pct(returning, season, last.index), season_projections(pairs, season)
            )
        this_season = by_season[season]
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
                p = prior_now[column].astype(float).fillna(season_means[kind])
                c = current[column].astype(float)
                blended = weight * c.fillna(0.0) + (1 - weight) * p
                out[column] = blended.where(p.notna(), c)
            out = out.reset_index()
            out.insert(0, "slate", int(slate))
            out.insert(0, "season", int(season))
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def frozen_ratings(ratings: pd.DataFrame, season: int, as_of_slate: int) -> pd.DataFrame:
    """Ratings as they stood at the start of as_of_slate, copied onto every later slate of `season`.

    Each team takes its row from the latest slate at or before as_of_slate. A team whose first game comes later
    uses that first row, which holds only its preseason prior because it had played no games. Other seasons and
    earlier slates are unchanged.
    """
    this = ratings[ratings["season"] == season].sort_values("slate")
    known = this[this["slate"] <= as_of_slate].drop_duplicates("team", keep="last").set_index("team")
    first = this.drop_duplicates("team", keep="first").set_index("team")
    snapshot = pd.concat([known, first[~first.index.isin(known.index)]])[RATING_COLUMNS]
    out = ratings.copy()
    later = (out["season"] == season) & (out["slate"] >= as_of_slate)
    out.loc[later, RATING_COLUMNS] = out.loc[later, ["team"]].join(snapshot, on="team")[RATING_COLUMNS].to_numpy()
    return out


FEATURES = [
    "home_field",
    "d_off_epa",
    "d_def_epa",
    "d_off_sr",
    "d_def_sr",
    "d_prior_sp",
    "d_talent",
    "d_rest",
    "d_returning",
]
GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "slate",
    "start_date",
    "home_team",
    "away_team",
    "neutral_site",
    "completed",
    "home_points",
    "away_points",
    "margin",
    "vegas_margin",
]


def assemble_features(
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    returning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game from precomputed slate ratings: home-minus-away features, Vegas, result."""
    returning = empty_returning() if returning is None else returning
    g = games[(games["home_classification"] == "fbs") & (games["away_classification"] == "fbs")]
    g = g.merge(slate_index(games)[["id", "slate"]], on="id")
    prior_sp = sp.assign(season=sp["year"] + 1)[["season", "team", "rating"]]
    season_talent = talent.rename(columns={"year": "season"})[["season", "team", "talent"]]
    season_returning = returning[["season", "team", "percent_ppa"]].drop_duplicates(["season", "team"])
    for side in ("home", "away"):
        team = f"{side}_team"
        g = g.merge(
            ratings.rename(columns={"team": team, **{c: f"{side}_{c}" for c in RATING_COLUMNS}}),
            on=["season", "slate", team],
            how="left",
        )
        g = g.merge(
            prior_sp.rename(columns={"team": team, "rating": f"{side}_prior_sp"}), on=["season", team], how="left"
        )
        g = g.merge(
            season_talent.rename(columns={"team": team, "talent": f"{side}_talent"}), on=["season", team], how="left"
        )
        g = g.merge(
            season_returning.rename(columns={"team": team, "percent_ppa": f"{side}_returning"}),
            on=["season", team],
            how="left",
        )
    g = g.merge(rest_days(games), on="id", how="left")
    g = g.merge(vegas_margin(lines).rename(columns={"game_id": "id"}), on="id", how="left")
    g["home_field"] = (~g["neutral_site"].eq(True)).astype(int)
    for column in RATING_COLUMNS:
        g[f"d_{column}"] = g[f"home_{column}"] - g[f"away_{column}"]
    g["d_prior_sp"] = g["home_prior_sp"] - g["away_prior_sp"]
    g["d_talent"] = g["home_talent"] - g["away_talent"]
    g["d_rest"] = g["home_rest"] - g["away_rest"]
    g["d_returning"] = g["home_returning"].astype(float) - g["away_returning"].astype(float)
    g["margin"] = np.where(g["completed"].eq(True), g["home_points"] - g["away_points"], np.nan)
    return g.rename(columns={"id": "game_id"})[GAME_COLUMNS + FEATURES].reset_index(drop=True)


def game_features(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    *,
    alpha: float,
    shrink_plays: int,
    returning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game: pregame home-minus-away features, the Vegas margin, and the result."""
    ratings = rolling_ratings(enriched, games, alpha=alpha, shrink_plays=shrink_plays, returning=returning)
    return assemble_features(ratings, games, lines, sp, talent, returning)
