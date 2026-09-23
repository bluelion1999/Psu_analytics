import pytest
from conftest import seed_dashboard_db
from scipy.stats import norm

from psu.dashboard.common import MissingData
from psu.dashboard.overview import METRICS, SCHEDULE_COLUMNS, metric_comparison, record, schedule
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_record_counts_conference_games_and_games_left(con):
    assert record(con, 2025, "Penn State") == {
        "wins": 2,
        "losses": 0,
        "conf_wins": 1,
        "conf_losses": 0,
        "remaining": 0,
    }
    assert record(con, 2026, "Penn State") == {
        "wins": 1,
        "losses": 0,
        "conf_wins": 0,
        "conf_losses": 0,
        "remaining": 1,
    }
    assert record(con, 2027, "Penn State")["remaining"] == 1


def test_schedule_lines_are_from_the_teams_side(con):
    s = schedule(con, 2025, "Penn State")
    assert list(s.columns) == SCHEDULE_COLUMNS
    away = s.set_index("game_id").loc[103]  # Penn State at Ohio State: stored home margin -4, Vegas 3.5
    assert away["model_margin"] == pytest.approx(4.0)
    assert away["vegas_margin"] == pytest.approx(-3.5)
    assert away["win_prob"] == pytest.approx(1 - norm.cdf(-4.0 / 16.0))
    home = schedule(con, 2026, "Penn State").set_index("game_id").loc[203]
    assert home["model_margin"] == pytest.approx(7.0) and home["prediction"] == "upcoming"
    assert home["win_prob"] == pytest.approx(norm.cdf(7.0 / 16.0))
    assert not bool(home["completed"]) and home["result"] is None


def test_schedule_without_predictions_has_blank_lines():
    c = connect(":memory:")
    seed_dashboard_db(c, with_model=False)
    s = schedule(c, 2026, "Penn State")
    assert len(s) == 2 and s["model_margin"].isna().all() and s["win_prob"].isna().all()


def test_metric_comparison_against_conference_and_fbs(con):
    m = metric_comparison(con, 2025, "Penn State").set_index("metric")
    assert list(m.index) == [label for label, _, _, _ in METRICS]
    offense = con.execute("SELECT team, epa_per_play FROM team_offense WHERE season = 2025").df().set_index("team")
    row = m.loc["Offense EPA/play"]
    assert row["value"] == pytest.approx(offense.loc["Penn State", "epa_per_play"])
    assert row["conference_avg"] == pytest.approx(offense.loc[["Penn State", "Ohio State"], "epa_per_play"].mean())
    assert row["national_avg"] == pytest.approx(offense["epa_per_play"].mean())
    assert not bool(m.loc["Defense EPA/play", "higher_is_better"])


def test_metric_comparison_for_a_season_without_stats(con):
    m = metric_comparison(con, 2027, "Penn State")
    assert len(m) == len(METRICS) and m["value"].isna().all()


def test_metric_comparison_needs_build_tables():
    c = connect(":memory:")
    c.execute("CREATE TABLE games AS SELECT 1 AS id")
    with pytest.raises(MissingData, match="psu build"):
        metric_comparison(c, 2025, "Penn State")
