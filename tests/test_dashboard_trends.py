import pytest
from conftest import seed_dashboard_db

from psu.dashboard.trends import TREND_METRICS, season_trends, weekly_trends
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_season_trends_long_format(con):
    t = season_trends(con, "Penn State")
    assert list(t.columns) == ["season", "metric", "group", "value"]
    assert set(t["season"]) == {2025, 2026}  # 2027 has no metric rows yet
    assert set(t["metric"]) == {label for label, _, _ in TREND_METRICS}
    assert set(t["group"]) == {"Penn State", "Big Ten avg", "FBS avg"}
    mine = t[(t["season"] == 2025) & (t["metric"] == "Offense EPA/play") & (t["group"] == "Penn State")]["value"]
    expected = con.execute(
        "SELECT epa_per_play FROM team_offense WHERE season = 2025 AND team = 'Penn State'"
    ).fetchone()[0]
    assert mine.item() == pytest.approx(expected)


def test_weekly_trends_one_row_per_game_and_side(con):
    w = weekly_trends(con, 2025, "Penn State")
    assert list(w.columns) == [
        "game",
        "start_date",
        "week",
        "opponent",
        "side",
        "epa_per_play",
        "success_rate",
        "plays",
    ]
    assert list(w["game"]) == ["Wk 1 Temple", "Wk 1 Temple", "Wk 2 Ohio State", "Wk 2 Ohio State"]
    assert set(w["side"]) == {"offense", "defense"}
    offense = w[(w["game"] == "Wk 1 Temple") & (w["side"] == "offense")].iloc[0]
    expected = con.execute(
        "SELECT avg(ppa) FROM plays_enriched WHERE game_id = 101 AND offense = 'Penn State' "
        "AND NOT coalesce(garbage, false) AND ppa IS NOT NULL"
    ).fetchone()[0]
    assert offense["epa_per_play"] == pytest.approx(expected)
    assert w["success_rate"].between(0, 1).all()


def test_weekly_trends_empty_season(con):
    w = weekly_trends(con, 2027, "Penn State")
    assert w.empty and "epa_per_play" in w.columns
