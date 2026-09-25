"""In-game win probability from the score, time left and the pregame expected margin (Stern 1994)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def in_game_wp(home_margin, pregame_margin, frac_left, sigma: float) -> np.ndarray:
    """P(home wins): remaining home scoring ~ N(pregame_margin * r, sigma^2 * r), r = fraction of regulation left."""
    r = np.clip(np.asarray(frac_left, dtype=float), 1e-4, 1.0)
    z = (np.asarray(home_margin, dtype=float) + np.asarray(pregame_margin, dtype=float) * r) / (sigma * np.sqrt(r))
    return norm.cdf(z)
