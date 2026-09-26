"""Opponent adjustment: per-season ridge regression of play outcomes on offense and defense team effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

VALUES = ("ppa", "success", "explosive")
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
    weights: str | None = None,
) -> pd.DataFrame:
    """Fit value ~ intercept + offense + defense + home per season.

    off_adj = intercept + offense effect (higher is better); def_adj = intercept + defense effect
    (lower is better). Teams with few plays are shrunk toward average by `alpha`. If `weights`
    names a column of `enriched`, it is used as the per-play regression weight, and play counts
    and raw means become weighted accordingly.
    """
    if value not in VALUES:
        raise ValueError(f"value must be one of {VALUES}, got {value!r}")
    df = enriched[~enriched["garbage"]] if exclude_garbage else enriched
    frames = []
    for season, s in df.groupby("season"):
        teams = pd.Index(sorted(set(s["offense"]) | set(s["defense"])))
        width = len(teams)
        home = np.select([s["venue"].eq("home"), s["venue"].eq("away")], [1.0, -1.0], 0.0)
        x = sparse.hstack(
            [
                _one_hot(teams.get_indexer(s["offense"]), width),
                _one_hot(teams.get_indexer(s["defense"]), width),
                sparse.csr_matrix(home.reshape(-1, 1)),
            ]
        ).tocsr()
        y = s[value].astype(float).to_numpy()
        w = s[weights].astype(float).to_numpy() if weights else None
        model = Ridge(alpha=alpha).fit(x, y, sample_weight=w)
        values = s[value].astype(float)
        c_off, c_def = model.coef_[:width], model.coef_[width : 2 * width]
        if w is None:
            off_plays = s.groupby("offense").size().reindex(teams, fill_value=0).to_numpy()
            def_plays = s.groupby("defense").size().reindex(teams, fill_value=0).to_numpy()
            off_raw = values.groupby(s["offense"]).mean().reindex(teams).to_numpy()
            def_raw = values.groupby(s["defense"]).mean().reindex(teams).to_numpy()
            mean = y.mean()
        else:
            w_series = pd.Series(w, index=s.index)
            off_plays = w_series.groupby(s["offense"]).sum().reindex(teams, fill_value=0.0).to_numpy()
            def_plays = w_series.groupby(s["defense"]).sum().reindex(teams, fill_value=0.0).to_numpy()
            wv = w_series * values
            off_wsum = wv.groupby(s["offense"]).sum().reindex(teams, fill_value=0.0).to_numpy()
            def_wsum = wv.groupby(s["defense"]).sum().reindex(teams, fill_value=0.0).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                off_raw = np.where(off_plays > 0, off_wsum / off_plays, np.nan)
                def_raw = np.where(def_plays > 0, def_wsum / def_plays, np.nan)
            mean = np.average(y, weights=w)
        off_adj = mean + c_off - np.average(c_off, weights=off_plays)
        def_adj = mean + c_def - np.average(c_def, weights=def_plays)
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team": teams,
                    "off_raw": off_raw,
                    "off_adj": off_adj,
                    "def_raw": def_raw,
                    "def_adj": def_adj,
                    "off_plays": off_plays,
                    "def_plays": def_plays,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
