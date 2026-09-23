import logging

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_league
from psu.simulate import MissingModel, makes_cfp, run_simulation

SIGMA = 16.0


def run(team="A", games=None, upcoming=None, **kw):
    g, u = synthetic_league()
    kw = {"season": 2026, "sigma": SIGMA, "team": team, "n_sims": 2000, "seed": 0, **kw}
    return run_simulation(g if games is None else games, u if upcoming is None else upcoming, **kw)


def test_probabilities_are_valid_and_win_totals_sum_to_one():
    r = run()
    assert list(r.win_totals.columns) == ["wins", "prob"]
    assert list(r.win_totals["wins"]) == [0, 1, 2, 3, 4, 5]  # A plays 5 regular-season games
    assert r.win_totals["prob"].sum() == pytest.approx(1.0)
    for p in (r.p_10_plus, r.p_title_game, r.p_conf_champ, r.p_cfp):
        assert 0.0 <= p <= 1.0
    assert r.mean_wins == pytest.approx((r.win_totals["wins"] * r.win_totals["prob"]).sum())
    assert r.p_10_plus == 0.0  # only 5 games


def test_completed_games_are_fixed():
    r = run(team="B")  # B already lost to A
    assert list(r.win_totals["wins"]) == [0, 1, 2, 3]
    assert r.win_totals.loc[r.win_totals["wins"] == 3, "prob"].item() == 0.0


def test_same_seed_gives_identical_results_and_other_seeds_differ():
    a, b, c = run(seed=4), run(seed=4), run(seed=5)
    pd.testing.assert_frame_equal(a.win_totals, b.win_totals)
    pd.testing.assert_frame_equal(a.conference, b.conference)
    assert a.p_conf_champ == b.p_conf_champ
    assert not a.conference.equals(c.conference)


def test_conference_odds_add_up():
    r = run()
    assert list(r.conference.columns) == ["team", "mean_conf_wins", "p_title_game", "p_conf_champ"]
    assert list(r.conference["team"]) == ["A", "B", "C", "D"]
    assert r.conference["p_title_game"].sum() == pytest.approx(2.0)
    assert r.conference["p_conf_champ"].sum() == pytest.approx(1.0)


def test_dominant_team_wins_the_conference():
    r = run(sigma=1.0, tau=0.1)
    a = r.conference.set_index("team").loc["A"]
    assert a["mean_conf_wins"] == pytest.approx(3.0)
    assert a["p_title_game"] == pytest.approx(1.0)
    assert r.p_conf_champ > 0.99 and r.p_cfp > 0.99


def test_unrated_team_game_is_a_likely_win_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="psu.simulate"):
        r = run(sigma=1.0, tau=0.1, n_sims=4000)
    assert "no prediction" in caplog.text
    assert r.win_totals.loc[r.win_totals["wins"] == 5, "prob"].item() == pytest.approx(0.95, abs=0.02)


def test_conference_game_without_prediction_is_an_error():
    _, upcoming = synthetic_league()
    with pytest.raises(MissingModel, match="psu train"):
        run(upcoming=upcoming[upcoming["game_id"] != 5])


def test_finished_season_uses_actual_results():
    games, upcoming = synthetic_league()
    games["completed"] = True
    todo = games["home_points"].isna()
    games.loc[todo, "home_points"] = 21  # every remaining game: home team wins 21-14
    games.loc[todo, "away_points"] = 14
    r = run(games=games, upcoming=upcoming.iloc[0:0])
    # A: beat B, beat X, lost at D, beat C, beat F -> 4 wins for certain
    assert r.win_totals.loc[r.win_totals["wins"] == 4, "prob"].item() == 1.0
    # A and B finish 2-1; A won head-to-head, so the title game is A vs B every time
    conf = r.conference.set_index("team")
    assert conf.loc["A", "p_title_game"] == 1.0 and conf.loc["B", "p_title_game"] == 1.0
    assert r.as_of == games["start_date"].max()


def test_as_of_is_latest_completed_game():
    assert run().as_of == pd.Timestamp(2026, 9, 1)


def test_makes_cfp_boundaries():
    losses = np.array([3, 2, 3, 1])
    champion = np.array([True, False, False, False])
    assert makes_cfp(losses, champion).tolist() == [True, True, False, True]
    assert makes_cfp(np.array([2]), np.array([False]), max_losses=1).tolist() == [False]


def test_title_game_loss_counts_toward_cfp_losses():
    r = run(team="B", sigma=1.0, tau=0.1, cfp_max_losses=1)
    # B already lost to A. Whenever B reaches the title game it loses to A (far stronger), giving B at least
    # 2 losses, so with max_losses=1 no season where B plays in the title game can reach the playoff.
    assert r.p_title_game > 0
    assert r.p_cfp <= 1.0 - r.p_title_game + 1e-9
