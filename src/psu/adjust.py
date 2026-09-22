"""Opponent adjustment: per-season ridge regression of play outcomes on offense and defense team effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

VALUES = ("ppa", "success")
COLUMNS = ["season", "team", "off_raw", "off_adj", "def_raw", "def_adj", "off_plays", "def_plays"]


def _one_hot(codes: np.ndarray, width: int) -> sparse.csr_matrix:
    n = len(codes)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, width))


def opponent_adjust(
    enriched: pd.DataFrame,
    value: str = "ppa",
    *,
    alpha: float = 50.0,
    exclude_garbage: bool = True,
) -> pd.DataFrame:
    """Fit value ~ intercept + offense + defense + home per season.

    off_adj = intercept + offense effect (higher is better); def_adj = intercept + defense effect
    (lower is better). Teams with few plays are shrunk toward average by `alpha`.
    """
    if value not in VALUES:
        raise ValueError(f"value must be one of {VALUES}, got {value!r}")
    df = enriched[~enriched["garbage"]] if exclude_garbage else enriched
    frames = []
    for season, s in df.groupby("season"):
        teams = pd.Index(sorted(set(s["offense"]) | set(s["defense"])))
        width = len(teams)
        home = np.select([s["venue"].eq("home"), s["venue"].eq("away")], [1.0, -1.0], 0.0)
        x = sparse.hstack([
            _one_hot(teams.get_indexer(s["offense"]), width),
            _one_hot(teams.get_indexer(s["defense"]), width),
            sparse.csr_matrix(home.reshape(-1, 1)),
        ]).tocsr()
        y = s[value].astype(float).to_numpy()
        model = Ridge(alpha=alpha).fit(x, y)
        values = s[value].astype(float)
        frames.append(pd.DataFrame({
            "season": season,
            "team": teams,
            "off_raw": values.groupby(s["offense"]).mean().reindex(teams).to_numpy(),
            "off_adj": model.intercept_ + model.coef_[:width],
            "def_raw": values.groupby(s["defense"]).mean().reindex(teams).to_numpy(),
            "def_adj": model.intercept_ + model.coef_[width:2 * width],
            "off_plays": s.groupby("offense").size().reindex(teams, fill_value=0).to_numpy(),
            "def_plays": s.groupby("defense").size().reindex(teams, fill_value=0).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
