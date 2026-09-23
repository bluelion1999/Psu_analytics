import pandas as pd
import pytest
from conftest import seed_dashboard_db

from psu.dashboard.players import COLUMNS, player_table, teams
from psu.db import SPECS, connect, upsert


@pytest.fixture
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_teams_in_a_season(con):
    assert teams(con, 2025) == ["Buffalo", "Ohio State", "Penn State", "Temple"]


def test_passing_season_totals(con):
    p = player_table(con, 2025, "Penn State", "passing")
    assert list(p.columns) == COLUMNS["passing"]
    row = p.iloc[0]
    assert row["player"] == "Penn State QB" and row["games"] == 2
    assert (row["comp"], row["att"], row["yards"], row["td"], row["int"]) == (40, 60, 361, 4, 2)
    assert row["comp_pct"] == pytest.approx(40 / 60) and row["yds_per_att"] == pytest.approx(361 / 60)


def test_team_pseudo_player_is_excluded(con):
    r = player_table(con, 2025, "Penn State", "rushing")
    assert list(r["player"]) == ["Penn State RB"]
    assert (r.iloc[0]["carries"], r.iloc[0]["yards"], r.iloc[0]["long"]) == (30, 181, 25)
    assert r.iloc[0]["yds_per_carry"] == pytest.approx(181 / 30)


def test_minimum_volume_filter(con):
    assert player_table(con, 2025, "Penn State", "receiving", minimum=12).iloc[0]["receptions"] == 12
    empty = player_table(con, 2025, "Penn State", "receiving", minimum=13)
    assert empty.empty and list(empty.columns) == COLUMNS["receiving"]


def test_malformed_stats_count_as_zero(con):
    rows = [
        {
            "game_id": 101,
            "season": 2025,
            "week": 1,
            "season_type": "regular",
            "team": "Penn State",
            "category": "passing",
            "stat_type": stat_type,
            "athlete_id": "psu-backup",
            "athlete_name": "Backup QB",
            "stat": stat,
        }
        for stat_type, stat in (("YDS", "--"), ("TD", "1"))  # no C/ATT at all
    ]
    upsert(con, SPECS["player_game_stats"], pd.DataFrame(rows))
    p = player_table(con, 2025, "Penn State", "passing").set_index("player")
    backup = p.loc["Backup QB"]
    assert (backup["att"], backup["yards"], backup["td"]) == (0, 0, 1)
    assert pd.isna(backup["comp_pct"])
    assert player_table(con, 2025, "Penn State", "passing", minimum=1)["player"].tolist() == ["Penn State QB"]


def test_unknown_kind_is_an_error(con):
    with pytest.raises(ValueError, match="kind"):
        player_table(con, 2025, "Penn State", "kicking")
