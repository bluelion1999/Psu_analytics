import pytest

from conftest import seed_dashboard_db
from psu.dashboard.common import MissingData
from psu.dashboard.predictions import next_slate, sim_conference, sim_summary, sim_win_totals, upcoming
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def untrained():
    c = connect(":memory:")
    seed_dashboard_db(c, with_model=False)
    yield c
    c.close()


def test_upcoming_games_only(con):
    u = upcoming(con, 2026, "Penn State")
    assert list(u["opponent"]) == ["Ohio State"] and u.iloc[0]["model_margin"] == pytest.approx(7.0)
    assert upcoming(con, 2025, "Penn State").empty


def test_sim_tables(con):
    s = sim_summary(con, "Penn State")
    assert s["p_cfp"] == pytest.approx(0.4) and s["season"] == 2026 and s["n_sims"] == 1000
    totals = sim_win_totals(con, "Penn State")
    assert list(totals.columns) == ["wins", "prob"] and totals["prob"].sum() == pytest.approx(1.0)
    conf = sim_conference(con)
    assert list(conf["team"]) == ["Ohio State", "Penn State"]


def test_sim_summary_for_a_team_without_a_run(con):
    with pytest.raises(MissingData, match="psu simulate"):
        sim_summary(con, "Temple")


def test_next_slate_is_the_earliest_upcoming_week(con):
    slate = next_slate(con, 2026)
    assert list(slate["week"].unique()) == [2] and len(slate) == 2
    assert list(slate.columns) == [
        "week", "start_date", "away_team", "home_team", "neutral_site", "pred_margin", "home_win_prob", "vegas_margin",
    ]
    assert next_slate(con, 2025).empty


def test_before_train_and_simulate(untrained):
    with pytest.raises(MissingData, match="psu simulate"):
        sim_summary(untrained, "Penn State")
    with pytest.raises(MissingData, match="psu train"):
        next_slate(untrained, 2026)
    assert upcoming(untrained, 2026, "Penn State")["model_margin"].isna().all()
