"""Pick the two conference title game teams for one simulated season (simplified Big Ten tiebreakers)."""
from __future__ import annotations

import numpy as np


def _break_tie(group: np.ndarray, h2h_wins: np.ndarray, h2h_games: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Order teams tied on conference win %: head-to-head win % (only if every pair played), then a coin flip."""
    if len(group) == 1:
        return [int(group[0])]
    coin = rng.random(len(group))
    games = h2h_games[np.ix_(group, group)]
    every_pair_played = (games + np.eye(len(group), dtype=int) > 0).all()
    if every_pair_played:
        h2h = h2h_wins[np.ix_(group, group)].sum(axis=1) / games.sum(axis=1)
    else:
        h2h = np.zeros(len(group))
    order = np.lexsort((coin, -h2h))  # primary key: head-to-head (desc); then the coin
    return [int(t) for t in group[order]]


def top_two(
    conf_wins, conf_games, h2h_wins: np.ndarray, h2h_games: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    """Indices of the 1st and 2nd placed teams by conference win %, resolving only the ties that decide them."""
    pct = np.asarray(conf_wins, dtype=float) / np.maximum(np.asarray(conf_games, dtype=float), 1.0)
    remaining = np.arange(len(pct))
    order: list[int] = []
    while len(order) < 2:
        tied = np.isclose(pct[remaining], pct[remaining].max())
        order += _break_tie(remaining[tied], h2h_wins, h2h_games, rng)[: 2 - len(order)]
        remaining = remaining[~tied]
    return order[0], order[1]
