import numpy as np

from psu.sim.standings import top_two

FULL = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)  # every pair played once


def h2h(*results):
    """results: (winner, loser) pairs -> wins matrix."""
    wins = np.zeros((4, 4), dtype=int)
    for winner, loser in results:
        wins[winner, loser] += 1
    return wins


def rng(seed=0):
    return np.random.default_rng(seed)


def test_clear_top_two():
    wins = h2h((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert top_two(np.array([3, 2, 1, 0]), np.array([3, 3, 3, 3]), wins, FULL, rng()) == (0, 1)


def test_win_percentage_not_raw_wins():
    games = np.array([6, 4, 4, 4])
    assert top_two(np.array([5, 4, 1, 0]), games, np.zeros((4, 4), int), FULL, rng())[0] == 1


def test_two_way_tie_for_first_goes_to_head_to_head():
    wins = h2h((1, 0), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3))  # 0 and 1 both 2-1; 1 beat 0
    for seed in range(20):
        assert top_two(np.array([2, 2, 1, 1]), np.array([3, 3, 3, 3]), wins, FULL, rng(seed)) == (1, 0)


def test_three_way_tie_for_second_uses_complete_round_robin():
    # 0 is clear first. 1, 2 and 3 tie on conference win % (conf_wins is passed directly); among them
    # 2 beat 1 and 3, and 3 beat 1, so head-to-head (2: 2-0, 3: 1-1, 1: 0-2) puts 2 second every time.
    wins = h2h((0, 1), (0, 2), (0, 3), (2, 1), (2, 3), (3, 1))
    conf_wins = np.array([3, 1, 1, 1])
    for seed in range(20):
        assert top_two(conf_wins, np.array([3, 3, 3, 3]), wins, FULL, rng(seed)) == (0, 2)


def test_incomplete_round_robin_falls_to_a_seeded_coin_flip():
    games = FULL.copy()
    games[1, 3] = games[3, 1] = 0  # 1 and 3 never met
    wins = h2h((0, 1), (0, 2), (0, 3), (2, 1), (2, 3))
    conf_wins = np.array([3, 1, 1, 1])
    seconds = {top_two(conf_wins, np.array([3, 3, 3, 3]), wins, games, rng(seed))[1] for seed in range(50)}
    assert seconds == {1, 2, 3}  # head-to-head skipped although 2 beat both others
    assert top_two(conf_wins, np.array([3, 3, 3, 3]), wins, games, rng(7)) == top_two(
        conf_wins, np.array([3, 3, 3, 3]), wins, games, rng(7)
    )


def test_even_head_to_head_is_a_coin_flip():
    wins = h2h((0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1))  # 1, 2, 3 each 1-1 among themselves
    firsts_seconds = {top_two(np.array([3, 1, 1, 1]), np.array([3, 3, 3, 3]), wins, FULL, rng(s)) for s in range(50)}
    assert {s for _, s in firsts_seconds} == {1, 2, 3}
    assert all(f == 0 for f, _ in firsts_seconds)
