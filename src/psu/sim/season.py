"""Vectorised game draws: a season-long strength per team per simulation, plus independent game noise."""

from __future__ import annotations

import math

import numpy as np


def game_noise_sd(sigma: float, tau: float) -> float:
    """Game-level noise so each game's total spread (two team draws plus noise) is still sigma."""
    if tau < 0 or 2 * tau**2 >= sigma**2:
        raise ValueError(f"tau must satisfy 0 <= tau < sigma/sqrt(2) (sigma={sigma:.1f}, tau={tau:g})")
    return math.sqrt(sigma**2 - 2 * tau**2)


def draw_strengths(n_sims: int, n_teams: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, tau, size=(n_sims, n_teams))


def draw_margins(
    pred_margin, home_idx, away_idx, strengths: np.ndarray, *, sigma: float, tau: float, rng: np.random.Generator
) -> np.ndarray:
    """Home-minus-away margins, shape (n_sims, n_games): every simulation plays every game."""
    s = game_noise_sd(sigma, tau)
    pred = np.asarray(pred_margin, dtype=float)
    noise = rng.normal(0.0, s, size=(strengths.shape[0], pred.size))
    return pred + strengths[:, home_idx] - strengths[:, away_idx] + noise


def draw_matchups(
    pred_margin, a_idx, b_idx, strengths: np.ndarray, *, sigma: float, tau: float, rng: np.random.Generator
) -> np.ndarray:
    """One game per simulation with its own teams (e.g. the title game): a-minus-b margins, shape (n_sims,)."""
    s = game_noise_sd(sigma, tau)
    rows = np.arange(strengths.shape[0])
    noise = rng.normal(0.0, s, size=rows.size)
    return np.asarray(pred_margin, dtype=float) + strengths[rows, a_idx] - strengths[rows, b_idx] + noise
