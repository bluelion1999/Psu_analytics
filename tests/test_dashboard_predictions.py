import pytest
from conftest import seed_dashboard_db

from psu.dashboard.common import MissingData
from psu.dashboard.predictions import next_slate, sim_conference, sim_history, sim_summary, sim_win_totals, upcoming
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
        "week",
        "start_date",
        "away_team",
        "home_team",
        "neutral_site",
        "pred_margin",
        "home_win_prob",
        "vegas_margin",
    ]
    assert next_slate(con, 2025).empty


def test_before_train_and_simulate(untrained):
    with pytest.raises(MissingData, match="psu simulate"):
        sim_summary(untrained, "Penn State")
    with pytest.raises(MissingData, match="psu train"):
        next_slate(untrained, 2026)
    assert upcoming(untrained, 2026, "Penn State")["model_margin"].isna().all()


HISTORY_VIEW = ["as_of_slate", "p_title_game", "p_conf_champ", "p_cfp", "backfilled"]


def test_sim_history_is_empty_without_the_table(con):
    h = sim_history(con, 2026, "Penn State")
    assert h.empty and list(h.columns) == HISTORY_VIEW


def test_sim_history_rows_are_ordered_by_week():
    import pandas as pd

    from psu.simulate import HISTORY_COLUMNS, write_history

    c = connect(":memory:")
    base = {
        "team": "Penn State",
        "run_at": pd.Timestamp("2026-09-20"),
        "mean_wins": 9.0,
        "p_10_plus": 0.4,
        "p_title_game": 0.5,
        "p_conf_champ": 0.3,
    }
    rows = pd.DataFrame(
        [
            {**base, "season": 2026, "as_of_slate": 3, "p_cfp": 0.6, "backfilled": False},
            {**base, "season": 2026, "as_of_slate": 1, "p_cfp": 0.5, "backfilled": True},
            {**base, "season": 2025, "as_of_slate": 1, "p_cfp": 0.9, "backfilled": True},
        ]
    )[HISTORY_COLUMNS]
    write_history(c, rows)
    h = sim_history(c, 2026, "Penn State")
    assert list(h.columns) == HISTORY_VIEW
    assert list(h["as_of_slate"]) == [1, 3] and list(h["p_cfp"]) == [0.5, 0.6]
