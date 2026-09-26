"""Pregame features for game prediction. Every feature for a game uses only information from before its slate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psu.adjust import opponent_adjust

MAX_REST = 21
METRICS: dict[str, tuple[str, str | None]] = {  # name -> (adjusted value column, play_class filter)
    "epa": ("ppa", None),
    "sr": ("success", None),
    "expl": ("explosive", None),
    "rush_epa": ("ppa", "rush"),
    "pass_epa": ("ppa", "pass"),
}
DEFAULT_METRICS = ("epa", "sr")


def rating_columns(metrics=DEFAULT_METRICS) -> list[str]:
    """["off_epa", "def_epa", "off_sr", "def_sr", ...] -- off/def per metric, in metric order."""
    return [f"{side}_{m}" for m in metrics for side in ("off", "def")]


RATING_COLUMNS = rating_columns()


def _metric_plays(plays: pd.DataFrame, metric: str) -> pd.DataFrame:
    play_class = METRICS[metric][1]
    return plays if play_class is None else plays[plays["play_class"] == play_class]


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


def league_means(enriched: pd.DataFrame, metrics=DEFAULT_METRICS) -> dict[str, float]:
    """Per metric, the mean of its value over non-garbage plays passing its play_class filter."""
    plays = enriched[~enriched["garbage"]]
    return {m: float(_metric_plays(plays, m)[METRICS[m][0]].astype(float).mean()) for m in metrics}


def _nan_means(metrics) -> dict[str, float]:
    return {m: np.nan for m in metrics}


def season_ratings(
    enriched: pd.DataFrame, alpha: float, metrics=DEFAULT_METRICS, weights: str | None = None
) -> pd.DataFrame:
    """Opponent-adjusted ratings per team over the given plays (one season), one off/def pair per metric.

    Also returns each metric's (weighted) play counts as off_plays_<metric>/def_plays_<metric>, and the first
    metric's counts as off_plays/def_plays.
    """
    columns = [
        *rating_columns(metrics),
        "off_plays",
        "def_plays",
        *[f"{side}_plays_{m}" for m in metrics for side in ("off", "def")],
    ]
    if enriched.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="team"), dtype=float)
    data: dict[str, pd.Series] = {}
    for m in metrics:
        value = METRICS[m][0]
        fit = opponent_adjust(_metric_plays(enriched, m), value, alpha=alpha, weights=weights).set_index("team")
        for side in ("off", "def"):
            data[f"{side}_{m}"] = fit[f"{side}_adj"]
            data[f"{side}_plays_{m}"] = fit[f"{side}_plays"]
    first = metrics[0]
    data["off_plays"] = data[f"off_plays_{first}"]
    data["def_plays"] = data[f"def_plays_{first}"]
    out = pd.DataFrame(data)
    out.index.name = "team"
    return out[columns]


def _ratings_at(
    teams: pd.Index,
    this_season: pd.DataFrame,
    slate: int,
    prior: pd.DataFrame,
    season_means: dict[str, float],
    *,
    alpha: float,
    shrink_plays: int,
    metrics=DEFAULT_METRICS,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Ratings for `teams` as of the start of `slate`: plays before `slate`, blended with `prior`.

    With `half_life`, each current-season play is weighted 0.5 ** ((slate - play slate) / half_life), and the
    blend uses the weighted play counts. The prior (last season's finals) is never weighted.
    """
    earlier = this_season[this_season["slate"] < slate]
    weights = None
    if half_life is not None:
        earlier = earlier.assign(_w=0.5 ** ((slate - earlier["slate"]) / half_life))
        weights = "_w"
    current = season_ratings(earlier, alpha, metrics, weights).reindex(teams)
    prior_now = prior.reindex(teams)
    out = pd.DataFrame(index=teams)
    for kind in metrics:
        for side in ("off", "def"):
            column = f"{side}_{kind}"
            n = current[f"{side}_plays_{kind}"].astype(float).fillna(0.0)
            weight = (n / (n + shrink_plays)).fillna(0.0)
            p = prior_now[column].astype(float).fillna(season_means[kind])
            c = current[column].astype(float)
            blended = weight * c.fillna(0.0) + (1 - weight) * p
            out[column] = blended.where(p.notna(), c)
    return out.reset_index()


def rolling_ratings(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    *,
    alpha: float,
    shrink_plays: int,
    metrics=DEFAULT_METRICS,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Ratings for each team as of the start of each slate, blended with last season's final ratings."""
    slates = slate_index(games)
    plays = enriched.merge(slates.rename(columns={"id": "game_id"})[["game_id", "slate"]], on="game_id")
    frames = []
    for season in sorted(slates["season"].unique()):
        previous = plays[plays["season"] == season - 1]
        prior = season_ratings(previous, alpha, metrics)
        means = league_means(previous, metrics) if not previous.empty else _nan_means(metrics)
        this_season = plays[plays["season"] == season]
        season_games = games[games["season"] == season].merge(slates[["id", "slate"]], on="id")
        for slate, slate_games in season_games.groupby("slate"):
            teams = pd.Index(sorted(set(slate_games["home_team"]) | set(slate_games["away_team"])), name="team")
            out = _ratings_at(
                teams,
                this_season,
                slate,
                prior,
                means,
                alpha=alpha,
                shrink_plays=shrink_plays,
                metrics=metrics,
                half_life=half_life,
            )
            out.insert(0, "slate", int(slate))
            out.insert(0, "season", int(season))
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def ratings_as_of(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    *,
    season: int,
    as_of_slate: int,
    alpha: float,
    shrink_plays: int,
    metrics=DEFAULT_METRICS,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Ratings for every team with a game in `season`, as of the start of as_of_slate.

    Unlike `rolling_ratings`, this rates every team in the season from all of its plays before as_of_slate,
    not just the teams scheduled at as_of_slate itself -- so a team coming off a bye is rated from its most
    recent game, not a stale earlier snapshot.
    """
    slates = slate_index(games)
    plays = enriched.merge(slates.rename(columns={"id": "game_id"})[["game_id", "slate"]], on="game_id")
    previous = plays[plays["season"] == season - 1]
    prior = season_ratings(previous, alpha, metrics)
    means = league_means(previous, metrics) if not previous.empty else _nan_means(metrics)
    this_season = plays[plays["season"] == season]
    season_games = games[games["season"] == season]
    teams = pd.Index(sorted(set(season_games["home_team"]) | set(season_games["away_team"])), name="team")
    out = _ratings_at(
        teams,
        this_season,
        as_of_slate,
        prior,
        means,
        alpha=alpha,
        shrink_plays=shrink_plays,
        metrics=metrics,
        half_life=half_life,
    )
    return out.set_index("team")[rating_columns(metrics)]


def frozen_ratings(ratings: pd.DataFrame, season: int, as_of_slate: int, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Ratings as they stood at the start of as_of_slate, copied onto every later slate of `season`.

    `snapshot` (indexed by team, from `ratings_as_of`) supplies the values for every team with a game at or
    after as_of_slate. Earlier slates and other seasons are unchanged. The rating columns replaced are the
    snapshot's columns.
    """
    columns = list(snapshot.columns)
    out = ratings.copy()
    later = (out["season"] == season) & (out["slate"] >= as_of_slate)
    out.loc[later, columns] = out.loc[later, ["team"]].join(snapshot, on="team")[columns].to_numpy()
    return out


def elo_as_of(games: pd.DataFrame, season: int, as_of_slate: int) -> pd.DataFrame:
    """`games`, with every home/away pregame Elo at or after as_of_slate in `season` frozen at the value CFBD
    would show for an unplayed game: the pregame Elo of the team's first game (by slate, then start_date) at
    slate >= as_of_slate. Games before as_of_slate, and every other season, are unchanged.
    """
    out = games.copy()
    if not {"home_pregame_elo", "away_pregame_elo"} <= set(out.columns):
        return out
    slates = slate_index(games)[["id", "slate"]]
    g = out.merge(slates, on="id", how="left")
    later = g[(g["season"] == season) & (g["slate"] >= as_of_slate)]
    if later.empty:
        return out
    long = pd.concat(
        [
            later[["slate", "start_date", f"{side}_team", f"{side}_pregame_elo"]].rename(
                columns={f"{side}_team": "team", f"{side}_pregame_elo": "elo"}
            )
            for side in ("home", "away")
        ],
        ignore_index=True,
    ).sort_values(["slate", "start_date"])
    as_of_elo = long.groupby("team")["elo"].first()
    later_ids = set(later["id"])
    mask = out["id"].isin(later_ids)
    for side in ("home", "away"):
        col = f"{side}_pregame_elo"
        out.loc[mask, col] = out.loc[mask, f"{side}_team"].map(as_of_elo).to_numpy()
    return out


FEATURES = ["home_field", "d_off_epa", "d_def_epa", "d_off_sr", "d_def_sr", "d_prior_sp", "d_talent", "d_rest"]
_OTHER_FEATURES = ["d_elo", "d_prior_sp", "d_talent", "d_rest"]
ALL_FEATURES = ["home_field", *[f"d_{c}" for c in rating_columns(tuple(METRICS))], *_OTHER_FEATURES]
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
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game from precomputed slate ratings: home-minus-away features, Vegas, result.

    Emits d_<col> for every rating column in `ratings`, and d_elo (home minus away pregame Elo, NaN when
    `games` has no Elo columns), so any feature subset can be selected later.
    """
    rating_cols = [c for c in ratings.columns if c.startswith(("off_", "def_")) and "plays" not in c]
    g = games[(games["home_classification"] == "fbs") & (games["away_classification"] == "fbs")]
    g = g.merge(slate_index(games)[["id", "slate"]], on="id")
    prior_sp = sp.assign(season=sp["year"] + 1)[["season", "team", "rating"]]
    season_talent = talent.rename(columns={"year": "season"})[["season", "team", "talent"]]
    for side in ("home", "away"):
        team = f"{side}_team"
        g = g.merge(
            ratings.rename(columns={"team": team, **{c: f"{side}_{c}" for c in rating_cols}}),
            on=["season", "slate", team],
            how="left",
        )
        g = g.merge(
            prior_sp.rename(columns={"team": team, "rating": f"{side}_prior_sp"}), on=["season", team], how="left"
        )
        g = g.merge(
            season_talent.rename(columns={"team": team, "talent": f"{side}_talent"}), on=["season", team], how="left"
        )
    g = g.merge(rest_days(games), on="id", how="left")
    g = g.merge(vegas_margin(lines).rename(columns={"game_id": "id"}), on="id", how="left")
    g["home_field"] = (~g["neutral_site"].eq(True)).astype(int)
    for column in rating_cols:
        g[f"d_{column}"] = g[f"home_{column}"] - g[f"away_{column}"]
    if {"home_pregame_elo", "away_pregame_elo"} <= set(g.columns):
        home_elo = pd.to_numeric(g["home_pregame_elo"], errors="coerce").astype(float)
        away_elo = pd.to_numeric(g["away_pregame_elo"], errors="coerce").astype(float)
        g["d_elo"] = home_elo - away_elo
    else:
        g["d_elo"] = np.nan
    g["d_prior_sp"] = g["home_prior_sp"] - g["away_prior_sp"]
    g["d_talent"] = g["home_talent"] - g["away_talent"]
    g["d_rest"] = g["home_rest"] - g["away_rest"]
    g["margin"] = np.where(g["completed"].eq(True), g["home_points"] - g["away_points"], np.nan)
    features = ["home_field", *[f"d_{c}" for c in rating_cols], *_OTHER_FEATURES]
    return g.rename(columns={"id": "game_id"})[GAME_COLUMNS + features].reset_index(drop=True)


def game_features(
    enriched: pd.DataFrame,
    games: pd.DataFrame,
    lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    *,
    alpha: float,
    shrink_plays: int,
    metrics=DEFAULT_METRICS,
    half_life: float | None = None,
) -> pd.DataFrame:
    """One row per FBS-vs-FBS game: pregame home-minus-away features, the Vegas margin, and the result."""
    ratings = rolling_ratings(
        enriched, games, alpha=alpha, shrink_plays=shrink_plays, metrics=metrics, half_life=half_life
    )
    return assemble_features(ratings, games, lines, sp, talent)
