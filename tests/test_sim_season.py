import numpy as np
import pytest
from scipy.stats import norm

from psu.sim.season import draw_margins, draw_matchups, draw_strengths, game_noise_sd

SIGMA = 16.0


def test_noise_sd_keeps_total_spread_at_sigma():
    assert game_noise_sd(SIGMA, 0.0) == pytest.approx(SIGMA)
    assert game_noise_sd(SIGMA, 5.0) == pytest.approx(np.sqrt(SIGMA**2 - 50.0))


def test_noise_sd_rejects_tau_too_large_for_sigma():
    with pytest.raises(ValueError, match="tau"):
        game_noise_sd(SIGMA, 12.0)  # 2 * 144 > 256
    with pytest.raises(ValueError, match="tau"):
        game_noise_sd(SIGMA, -1.0)


def test_single_game_win_rate_matches_normal_cdf_without_team_draws():
    rng = np.random.default_rng(0)
    strengths = draw_strengths(200_000, 2, 0.0, rng)
    margins = draw_margins(np.array([5.0]), np.array([0]), np.array([1]), strengths, sigma=SIGMA, tau=0.0, rng=rng)
    assert margins.shape == (200_000, 1)
    assert (margins[:, 0] > 0).mean() == pytest.approx(norm.cdf(5.0 / SIGMA), abs=0.005)


def test_team_draws_keep_spread_and_correlate_games_sharing_a_team():
    rng = np.random.default_rng(1)
    strengths = draw_strengths(100_000, 3, 8.0, rng)
    margins = draw_margins(
        np.array([0.0, 0.0]), np.array([0, 0]), np.array([1, 2]), strengths, sigma=SIGMA, tau=8.0, rng=rng
    )
    assert margins.std(axis=0) == pytest.approx([SIGMA, SIGMA], rel=0.02)
    corr = np.corrcoef(margins[:, 0], margins[:, 1])[0, 1]
    assert 0.2 < corr < 0.3  # tau^2 / sigma^2 = 64 / 256 = 0.25


def test_games_are_independent_without_team_draws():
    rng = np.random.default_rng(2)
    strengths = draw_strengths(100_000, 3, 0.0, rng)
    margins = draw_margins(
        np.array([0.0, 0.0]), np.array([0, 0]), np.array([1, 2]), strengths, sigma=SIGMA, tau=0.0, rng=rng
    )
    assert abs(np.corrcoef(margins[:, 0], margins[:, 1])[0, 1]) < 0.02


def test_draw_margins_accepts_one_sigma_per_game_and_matches_scalar_when_equal():
    strengths = np.zeros((4000, 3))
    pred, home, away = np.zeros(2), np.array([0, 0]), np.array([1, 2])
    a = draw_margins(pred, home, away, strengths, sigma=SIGMA, tau=0.0, rng=np.random.default_rng(1))
    b = draw_margins(pred, home, away, strengths, sigma=np.array([SIGMA, SIGMA]), tau=0.0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)  # same seed, same draws
    c = draw_margins(pred, home, away, strengths, sigma=np.array([1.0, 30.0]), tau=0.0, rng=np.random.default_rng(1))
    assert c[:, 0].std() == pytest.approx(1.0, rel=0.1) and c[:, 1].std() == pytest.approx(30.0, rel=0.1)


def test_matchups_use_each_simulations_own_teams_and_strengths():
    rng = np.random.default_rng(3)
    n = 50_000
    strengths = np.zeros((n, 3))
    strengths[:, 2] = 100.0  # team 2 is overwhelming in every simulation
    a = np.where(np.arange(n) % 2 == 0, 2, 0)  # even simulations: team 2 vs team 1
    b = np.ones(n, dtype=int)
    margins = draw_matchups(np.zeros(n), a, b, strengths, sigma=SIGMA, tau=0.0, rng=rng)
    assert margins.shape == (n,)
    assert (margins[a == 2] > 0).all()
    assert (margins[a == 0] > 0).mean() == pytest.approx(0.5, abs=0.02)
