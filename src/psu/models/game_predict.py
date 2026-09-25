"""Game margin model: pipelines, win probability, scoring, and a time-based backtest against Vegas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from psu.features import FEATURES

MODEL_KINDS = ("linear", "xgboost")


class _MedianImputer(BaseEstimator, TransformerMixin):
    """Behavioral equivalent of sklearn.impute.SimpleImputer(strategy="median", keep_empty_features=True).

    `sklearn.impute` unconditionally imports `sklearn.neighbors` (for KNNImputer), and on this
    machine that pulls in `sklearn/neighbors/_ball_tree*.pyd`, which Windows Smart App Control
    blocks (CodeIntegrity events 3077/3033: "did not meet the Enterprise signing level
    requirements", Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6}) -- confirmed to reproduce
    even after a clean `pip install --force-reinstall` of scikit-learn. `sklearn.base` does not
    touch that import chain, so this stays a normal fit/transform Pipeline step with the same
    per-column median statistics_ and the same keep_empty_features fallback (an all-NaN training
    column imputes to 0.0 instead of being dropped).
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(X, axis=0)
        self.statistics_ = np.where(np.isnan(medians), 0.0, medians)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        nan_mask = np.isnan(X)
        fill = np.broadcast_to(self.statistics_, X.shape)
        X[nan_mask] = fill[nan_mask]
        return X


def make_pipeline(kind: str) -> Pipeline:
    if kind == "linear":
        steps = [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
    elif kind == "xgboost":
        steps = [
            (
                "model",
                XGBRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0, n_jobs=1
                ),
            )
        ]
    else:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")
    return Pipeline([("impute", _MedianImputer()), *steps])


def win_prob(margin, sigma: float) -> np.ndarray:
    """P(home team wins) given a predicted home margin and the margin's error spread."""
    return norm.cdf(np.asarray(margin, dtype=float) / sigma)


PHASES = ("early", "mid", "post")
EARLY_SLATES = 4  # slates 1-4 lean hardest on the preseason prior
MIN_PHASE_N = 30


def phase_of(season_type, slate) -> np.ndarray:
    season_type = np.asarray(season_type, dtype=object)
    slate = np.asarray(slate, dtype=float)
    return np.where(season_type == "postseason", "post", np.where(slate <= EARLY_SLATES, "early", "mid"))


@dataclass
class GameModel:
    kind: str
    pipeline: Pipeline
    sigma: float
    sigma_by_phase: dict[str, float] | None = None

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[FEATURES])

    def row_sigma(self, df: pd.DataFrame) -> np.ndarray:
        if not self.sigma_by_phase or not {"season_type", "slate"} <= set(df.columns):
            return np.full(len(df), self.sigma, dtype=float)
        phases = phase_of(df["season_type"], df["slate"])
        return np.array([self.sigma_by_phase.get(p, self.sigma) for p in phases], dtype=float)

    def win_prob(self, df: pd.DataFrame) -> np.ndarray:
        return win_prob(self.predict_margin(df), self.row_sigma(df))


def fit(train: pd.DataFrame, kind: str) -> GameModel:
    x, y = train[FEATURES], train["margin"]
    out_of_fold = cross_val_predict(make_pipeline(kind), x, y, cv=5)
    sigma = float(np.std(y - out_of_fold, ddof=1))
    return GameModel(kind, make_pipeline(kind).fit(x, y), sigma)


def scores(actual, predicted, prob) -> dict:
    actual = np.asarray(actual, dtype=float)
    if len(actual) == 0:
        return {"n": 0, "mae": None, "brier": None}
    predicted = np.asarray(predicted, dtype=float)
    prob = np.asarray(prob, dtype=float)
    home_won = (actual > 0).astype(float)
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "brier": float(np.mean((prob - home_won) ** 2)),
    }


def oos_residuals(played: pd.DataFrame, seasons: list[int], kind: str) -> pd.DataFrame:
    """Walk-forward residuals: each season after the first, predicted by a model fitted on earlier seasons only."""
    frames = []
    for season in seasons[1:]:
        train = played[played["season"].isin([s for s in seasons if s < season])]
        scored = played[played["season"] == season]
        if train.empty or scored.empty:
            continue
        model = fit(train, kind)
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "phase": phase_of(scored["season_type"], scored["slate"]),
                    "residual": scored["margin"].to_numpy(float) - model.predict_margin(scored),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["season", "phase", "residual"])


def phase_sigmas(residuals: pd.DataFrame, fallback: float, min_n: int = MIN_PHASE_N) -> dict[str, float]:
    """RMS residual per phase; phases with fewer than min_n residuals use the pooled RMS (or fallback if none)."""
    r = residuals["residual"].astype(float).to_numpy()
    pooled = float(np.sqrt(np.mean(r**2))) if len(r) else float(fallback)
    out = {}
    for phase in PHASES:
        rp = r[(residuals["phase"] == phase).to_numpy()]
        out[phase] = float(np.sqrt(np.mean(rp**2))) if len(rp) >= min_n else pooled
    return out


def reliability(prob, won, bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins (labelled by lower edge): count, mean predicted and actual home-win rate."""
    prob = np.asarray(prob, dtype=float)
    won = np.asarray(won, dtype=float)
    edge = np.minimum((prob * bins).astype(int), bins - 1) / bins
    df = pd.DataFrame({"bin": edge, "prob": prob, "won": won})
    table = df.groupby("bin").agg(n=("prob", "size"), mean_pred=("prob", "mean"), actual=("won", "mean"))
    return table.reset_index()[["bin", "n", "mean_pred", "actual"]]


def ece(table: pd.DataFrame) -> float | None:
    """Expected calibration error: bin-size-weighted mean |mean predicted - actual|."""
    n = table["n"].sum()
    if n == 0:
        return None
    return float((table["n"] * (table["mean_pred"] - table["actual"]).abs()).sum() / n)


def _records(table: pd.DataFrame) -> list[dict]:
    return [
        {"bin": float(r.bin), "n": int(r.n), "mean_pred": float(r.mean_pred), "actual": float(r.actual)}
        for r in table.itertuples()
    ]


def usable_seasons(features: pd.DataFrame, current_season: int) -> list[int]:
    first = features["season"].min()
    played = features.loc[features["margin"].notna(), "season"].unique()
    return sorted(int(s) for s in played if first < s < current_season)


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    return features[features["margin"].notna() & (features["season"] > features["season"].min())]


def _vegas_sigma(train: pd.DataFrame) -> float:
    lined = train[train["vegas_margin"].notna()]
    return float(np.std(lined["margin"] - lined["vegas_margin"], ddof=1))


def _score_model(model: GameModel, games: pd.DataFrame) -> dict:
    if games.empty:
        return scores([], [], [])
    return scores(games["margin"], model.predict_margin(games), model.win_prob(games))


def backtest(features: pd.DataFrame, current_season: int, team: str = "Penn State") -> dict:
    seasons = usable_seasons(features, current_season)
    if len(seasons) < 3:
        raise ValueError(f"Need at least 3 complete seasons after the first; have {seasons}")
    played = features[features["margin"].notna()]
    validate_season, test_season = seasons[-2], seasons[-1]

    def split(eval_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        before = [s for s in seasons if s < eval_season]
        return played[played["season"].isin(before)], played[played["season"] == eval_season]

    train, valid = split(validate_season)
    validation = {kind: _score_model(fit(train, kind), valid) for kind in MODEL_KINDS}
    kind = min(MODEL_KINDS, key=lambda k: validation[k]["mae"])
    residuals = oos_residuals(played, seasons, kind)

    train, test = split(test_season)
    model = fit(train, kind)
    model.sigma_by_phase = phase_sigmas(residuals[residuals["season"] < test_season], model.sigma)
    vegas_sigma = _vegas_sigma(train)

    def block(games: pd.DataFrame) -> dict:
        lined = games[games["vegas_margin"].notna()]
        return {
            "model": _score_model(model, games),
            "model_on_lined_games": _score_model(model, lined),
            "vegas": scores(lined["margin"], lined["vegas_margin"], win_prob(lined["vegas_margin"], vegas_sigma)),
        }

    def calibration(prob, won) -> dict:
        table = reliability(prob, won)
        return {"ece": ece(table), "bins": _records(table)}

    is_team = test["home_team"].eq(team) | test["away_team"].eq(team)
    is_early = phase_of(test["season_type"], test["slate"]) == "early"
    lined = test[test["vegas_margin"].notna()]
    won = (lined["margin"] > 0).astype(float).to_numpy()
    return {
        "model_kind": kind,
        "validation_season": validate_season,
        "validation": validation,
        "test_season": test_season,
        "train_seasons": [s for s in seasons if s < test_season],
        "sigma": model.sigma,
        "sigma_by_phase": phase_sigmas(residuals, model.sigma),
        "vegas_sigma": vegas_sigma,
        "test": {"all": block(test), "early": block(test[is_early]), team: block(test[is_team])},
        "calibration": {
            "model": calibration(model.win_prob(lined), won),
            "vegas": calibration(win_prob(lined["vegas_margin"], vegas_sigma), won),
        },
    }
