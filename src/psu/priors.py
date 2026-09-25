"""Preseason priors: last season's final ratings, regressed toward the league mean by how much production returns.

Offense ratings regress by b0 + b1 * (share of last season's offensive PPA that returns); CFBD's returning
production covers offense only, so defense ratings use b0 alone. Coefficients for a season are fitted only on
earlier season pairs, so a prior never sees the season it predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FALLBACK_B0 = 0.6  # used when there are too few earlier season pairs to fit
DEFAULT_RETURNING = 0.5  # returning share when a season has no returning-production data at all
MIN_PAIRS = 20
PAIR_COLUMNS = ["season", "team", "x", "y", "ret"]


@dataclass(frozen=True)
class Projection:
    b0: float
    b1: float = 0.0

    def factor(self, ret) -> np.ndarray:
        """How much of last season's distance from the mean carries over, clipped to [0, 1]."""
        return np.clip(self.b0 + self.b1 * np.asarray(ret, dtype=float), 0.0, 1.0)


def empty_returning() -> pd.DataFrame:
    return pd.DataFrame(
        {"season": pd.Series(dtype="int64"), "team": pd.Series(dtype=object), "percent_ppa": pd.Series(dtype=float)}
    )


def returning_pct(returning: pd.DataFrame, season: int, teams) -> pd.Series:
    """Each team's returning share of offensive PPA for `season`; unknown teams get that season's median."""
    teams = pd.Index(teams, name="team")
    rows = returning.loc[returning["season"] == season].drop_duplicates("team").set_index("team")["percent_ppa"]
    known = rows.astype(float).dropna()
    fill = float(known.median()) if len(known) else DEFAULT_RETURNING
    return rows.astype(float).reindex(teams).fillna(fill)


def projection_pairs(
    finals: dict[int, pd.DataFrame], means: dict[int, dict[str, float]], returning: pd.DataFrame, column: str
) -> pd.DataFrame:
    """One row per team per season S: x = last(S-1) - mean(S-1), y = final(S) - mean(S), ret = returning share in S."""
    kind = column.split("_")[1]
    frames = []
    for season in sorted(finals):
        if season - 1 not in finals:
            continue
        both = pd.concat(
            {"last": finals[season - 1][column], "next": finals[season][column]}, axis=1, join="inner"
        ).dropna()
        if both.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team": both.index.to_numpy(),
                    "x": (both["last"] - means[season - 1][kind]).to_numpy(float),
                    "y": (both["next"] - means[season][kind]).to_numpy(float),
                    "ret": returning_pct(returning, season, both.index).to_numpy(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)


def fit_projection(pairs: pd.DataFrame, column: str) -> Projection:
    """Least squares on y = (b0 + b1*ret) * x for offense columns, y = b0 * x for defense columns."""
    pairs = pairs.dropna(subset=["x", "y", "ret"])
    if len(pairs) < MIN_PAIRS:
        return Projection(FALLBACK_B0)
    x, y, ret = (pairs[c].to_numpy(float) for c in ("x", "y", "ret"))
    if column.startswith("off_"):
        if np.ptp(ret) < 1e-9:
            # Returning data is constant (e.g. every pair filled with the same median/default): x and x*ret
            # are collinear, so lstsq would split the slope between b0 and b1 arbitrarily. Fit b0 alone.
            denom = float(x @ x)
            b0 = float(x @ y / denom) if denom > 0 else FALLBACK_B0
            return Projection(b0)
        (b0, b1), *_ = np.linalg.lstsq(np.column_stack([x, x * ret]), y, rcond=None)
        return Projection(float(b0), float(b1))
    denom = float(x @ x)
    return Projection(float(x @ y / denom)) if denom > 0 else Projection(FALLBACK_B0)


def season_projections(pairs: dict[str, pd.DataFrame], season: int) -> dict[str, Projection]:
    """Projections for `season`, fitted only on pairs whose target season is earlier."""
    return {column: fit_projection(p[p["season"] < season], column) for column, p in pairs.items()}


def project_prior(
    last: pd.DataFrame, means: dict[str, float], ret: pd.Series, projections: dict[str, Projection]
) -> pd.DataFrame:
    """prior = mean + factor * (last - mean), per rating column in `projections`."""
    out = last.copy()
    r = ret.reindex(last.index).to_numpy(float)
    for column, projection in projections.items():
        mean = means[column.split("_")[1]]
        out[column] = mean + projection.factor(r) * (last[column].astype(float) - mean)
    return out
