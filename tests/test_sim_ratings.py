import pandas as pd
import pytest

from psu.sim.ratings import Ratings, fit_ratings

TRUE = {"A": 10.0, "B": 4.0, "C": -3.0, "D": -11.0}  # sums to 0


def round_robin(hfa=3.0):
    rows = [
        {"home_team": h, "away_team": a, "neutral_site": False, "pred_margin": hfa + TRUE[h] - TRUE[a]}
        for h in TRUE
        for a in TRUE
        if h != a
    ]
    rows.append({"home_team": "A", "away_team": "D", "neutral_site": True, "pred_margin": TRUE["A"] - TRUE["D"]})
    return pd.DataFrame(rows)


def test_fit_ratings_recovers_known_ratings_and_home_edge():
    r = fit_ratings(round_robin(), ridge=1e-6)
    assert r.hfa == pytest.approx(3.0, abs=1e-3)
    for team, value in TRUE.items():
        assert r.rating[team] == pytest.approx(value, abs=1e-3)
    assert r.neutral_margin("B", "C") == pytest.approx(7.0, abs=1e-3)


def test_ridge_shrinks_but_keeps_order_and_centres_ratings():
    r = fit_ratings(round_robin(), ridge=5.0)
    values = [r.rating[t] for t in ("A", "B", "C", "D")]
    assert values == sorted(values, reverse=True)
    assert abs(r.rating["A"]) < TRUE["A"]
    assert sum(values) == pytest.approx(0.0, abs=1e-9)


def test_missing_neutral_flag_counts_as_home_game():
    games = round_robin()
    games["neutral_site"] = games["neutral_site"].astype(object)
    games.loc[0, "neutral_site"] = None
    r = fit_ratings(games, ridge=1e-6)
    assert r.hfa == pytest.approx(3.0, abs=1e-3)


def test_fit_ratings_on_no_games_is_empty():
    empty = pd.DataFrame(columns=["home_team", "away_team", "neutral_site", "pred_margin"])
    assert fit_ratings(empty) == Ratings(0.0, {})
