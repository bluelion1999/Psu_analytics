"""Power ratings fitted to the game model's own predicted margins (used for games with no prediction)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ratings:
    hfa: float
    rating: dict[str, float]

    def neutral_margin(self, a: str, b: str) -> float:
        return self.rating[a] - self.rating[b]


def fit_ratings(upcoming: pd.DataFrame, ridge: float = 1.0) -> Ratings:
    """Least squares on pred_margin ~ hfa * (not neutral) + r[home] - r[away], ridge on r, ratings centred at 0."""
    if upcoming.empty:
        return Ratings(0.0, {})
    teams = sorted(set(upcoming["home_team"]) | set(upcoming["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    rows = np.arange(len(upcoming))
    x = np.zeros((len(upcoming), k + 1))
    x[rows, upcoming["home_team"].map(index).to_numpy()] += 1.0
    x[rows, upcoming["away_team"].map(index).to_numpy()] -= 1.0
    x[:, k] = (~upcoming["neutral_site"].fillna(False).astype(bool)).astype(float).to_numpy()
    y = upcoming["pred_margin"].to_numpy(float)
    penalty = np.full(k + 1, float(ridge))
    penalty[k] = 1e-9  # hfa is unpenalised; the tiny term keeps an all-neutral slate solvable
    coef = np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)
    centred = coef[:k] - coef[:k].mean()
    return Ratings(float(coef[k]), {t: float(v) for t, v in zip(teams, centred)})
