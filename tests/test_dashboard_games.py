import numpy as np
import pytest
from scipy.stats import norm

from conftest import seed_dashboard_db
from psu.dashboard.games import (
    BOX_STATS, box_score, drive_chart, game_options, line_scores, top_plays, win_probability,
)
from psu.dashboard.winprob import in_game_wp
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_in_game_wp_matches_pregame_at_kickoff_and_score_at_the_end():
    assert in_game_wp(0, 7.0, 1.0, 16.0) == pytest.approx(norm.cdf(7.0 / 16.0))
    assert in_game_wp(3, -10.0, 0.0, 16.0) == pytest.approx(1.0)
    assert in_game_wp(-3, 10.0, 0.0, 16.0) == pytest.approx(0.0)
    assert in_game_wp(0, 0.0, 0.5, 16.0) == pytest.approx(0.5)
    out = in_game_wp(np.array([0, 7]), 0.0, np.array([1.0, 0.5]), 16.0)
    assert out.shape == (2,) and out[1] > 0.5


def test_game_options_list_completed_games(con):
    o = game_options(con, 2025, "Penn State")
    assert list(o["game_id"]) == [101, 103]
    assert list(o["label"]) == ["Wk 1 vs Temple (W 34-10)", "Wk 2 at Ohio State (W 27-24)"]


def test_game_options_empty_before_any_game(con):
    o = game_options(con, 2027, "Penn State")
    assert o.empty and list(o.columns) == ["game_id", "label"]


def test_line_scores_away_first_and_add_up(con):
    s = line_scores(con, 101)
    assert list(s["team"]) == ["Temple", "Penn State"]
    assert list(s.columns) == ["team", "Q1", "Q2", "Q3", "Q4", "Total"]
    for _, row in s.iterrows():
        assert row[["Q1", "Q2", "Q3", "Q4"]].sum() == row["Total"]


def test_box_score_away_then_home_with_blanks(con):
    b = box_score(con, 101).set_index("stat")
    assert list(b.columns) == ["Temple", "Penn State"]
    assert list(b.index) == [label for _, label in BOX_STATS]
    assert b.loc["Total yards", "Penn State"] == "334" and b.loc["Total yards", "Temple"] == "310"
    assert b.loc["4th down", "Penn State"] == ""


def test_drive_chart_positions_from_own_goal(con):
    d = drive_chart(con, 101)
    assert list(d.columns) == [
        "drive_number", "offense", "quarter", "start_pos", "end_pos", "plays", "yards", "result", "time",
    ]
    psu = d[d["offense"] == "Penn State"].iloc[0]
    assert (psu["start_pos"], psu["end_pos"], psu["result"], psu["time"]) == (25, 95, "TD", "4:30")


def test_top_plays_by_absolute_epa(con):
    t = top_plays(con, 101, n=3)
    assert len(t) == 3
    assert list(t.columns) == ["quarter", "clock", "offense", "down", "distance", "epa", "text"]
    assert t["epa"].abs().is_monotonic_decreasing
    assert t.iloc[0]["epa"] == pytest.approx(0.7) and t.iloc[0]["clock"] == "5:00"


def test_win_probability_runs_kickoff_to_final(con):
    wp = win_probability(con, 101, 16.0)
    assert list(wp.columns) == ["play", "seconds_left", "home_margin", "home_wp", "minute"]
    assert len(wp) == 16 + 2
    assert wp.iloc[0]["home_wp"] == pytest.approx(norm.cdf(7.0 / 16.0))  # Penn State favoured by 7 at home
    assert wp.iloc[-1]["home_wp"] == pytest.approx(1.0) and wp.iloc[-1]["home_margin"] == 24
    assert wp["minute"].is_monotonic_increasing
    assert wp["home_wp"].between(0, 1).all()


def test_win_probability_without_predictions_starts_even():
    c = connect(":memory:")
    seed_dashboard_db(c, with_model=False)
    assert win_probability(c, 101, 16.0).iloc[0]["home_wp"] == pytest.approx(0.5)
