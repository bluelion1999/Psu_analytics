"""Replay earlier weeks of a season: simulate as of each finished week, with team ratings frozen at that week.

The ratings behind every replayed prediction use only plays from before the as-of week. The model's coefficients
were fitted with this season's finished games in view, so replays are close to, but not exactly, what the model
would have said at the time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from psu.config import TEAM
from psu.features import FEATURES, assemble_features, frozen_ratings, rolling_ratings
from psu.simulate import (
    MissingModel,
    SimResult,
    current_slate,
    history_rows,
    load_inputs,
    load_sigmas,
    run_simulation,
    write_history,
)
from psu.train import load_feature_inputs

PREDICTION_COLUMNS = ["game_id", "home_team", "away_team", "neutral_site", "pred_margin"]


def _model_n_features(model) -> int | None:
    """Number of input features the fitted pipeline expects, or None if it can't be determined."""
    n = getattr(model.pipeline, "n_features_in_", None)
    if n is not None:
        return int(n)
    imputer = model.pipeline.named_steps.get("impute")
    statistics = getattr(imputer, "statistics_", None)
    return None if statistics is None else len(statistics)


def _check_model_features(model) -> None:
    n = _model_n_features(model)
    if n is not None and n != len(FEATURES):
        raise MissingModel("saved model predates the current features; run `psu train` first")


def replay_games(games: pd.DataFrame, as_of_slate: int) -> pd.DataFrame:
    """The season as it stood before as_of_slate: that week and everything after it is unplayed again."""
    replay = games.copy()
    later = (replay["season_type"] != "regular") | (replay["week"] >= as_of_slate)
    replay.loc[later, "completed"] = False
    replay[["home_points", "away_points"]] = replay[["home_points", "away_points"]].astype("Float64")
    replay.loc[later, ["home_points", "away_points"]] = pd.NA
    return replay


def simulate_as_of(
    games: pd.DataFrame,
    predict: Callable[[int], pd.DataFrame],
    *,
    season: int,
    sigma: float | Mapping[str, float],
    team: str,
    n_sims: int,
    seed: int,
    tau: float,
    now: pd.Timestamp | None = None,
) -> list[SimResult]:
    """One simulation per finished week N (1 .. current - 1), from predict(N) and the games as of N."""
    return [
        run_simulation(
            replay_games(games, n),
            predict(n),
            season=season,
            sigma=sigma,
            team=team,
            n_sims=n_sims,
            seed=seed,
            tau=tau,
            now=now,
        )
        for n in range(1, current_slate(games, season, now=now))
    ]


def backfill_history(
    con: duckdb.DuckDBPyConnection,
    *,
    season: int,
    out_dir: Path,
    alpha: float,
    shrink_plays: int,
    team: str = TEAM,
    n_sims: int,
    seed: int,
    tau: float,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    model_path = Path(out_dir) / "models" / "game_model.joblib"
    if not model_path.exists():
        raise MissingModel(f"{model_path} not found; run `psu train` first")
    sigmas = load_sigmas(out_dir)
    model = joblib.load(model_path)
    _check_model_features(model)
    games, _ = load_inputs(con, season)
    inputs = load_feature_inputs(con)
    ratings = rolling_ratings(
        inputs.enriched, inputs.games, alpha=alpha, shrink_plays=shrink_plays, returning=inputs.returning
    )
    season_games = inputs.games[inputs.games["season"] == season]

    def predict(n: int) -> pd.DataFrame:
        feats = assemble_features(
            frozen_ratings(ratings, season, n), season_games, inputs.lines, inputs.sp, inputs.talent, inputs.returning
        )
        feats = feats[feats["slate"] >= n]
        return feats.assign(pred_margin=model.predict_margin(feats))[PREDICTION_COLUMNS]

    results = simulate_as_of(
        games, predict, season=season, sigma=sigmas, team=team, n_sims=n_sims, seed=seed, tau=tau, now=now
    )
    rows = history_rows(results, backfilled=True)
    con.execute("BEGIN TRANSACTION")
    try:
        write_history(con, rows)
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
    return rows
