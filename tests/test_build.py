import pandas as pd
import pytest
from conftest import seed_raw_tables

from psu.build import DERIVED_TABLES, build, fbs_teams
from psu.db import connect
from psu.transform import GarbageTime


def test_fbs_teams_uses_game_classifications():
    games = pd.DataFrame(
        [
            {
                "season": 2024,
                "home_team": "Alpha",
                "home_classification": "fbs",
                "away_team": "Gamma",
                "away_classification": "fcs",
            },
            {
                "season": 2024,
                "home_team": "Beta",
                "home_classification": "fbs",
                "away_team": "Alpha",
                "away_classification": "fbs",
            },
        ]
    )
    assert set(map(tuple, fbs_teams(games)[["season", "team"]].to_numpy())) == {(2024, "Alpha"), (2024, "Beta")}


def test_build_writes_fbs_only_tables_and_is_repeatable():
    con = connect(":memory:")
    seed_raw_tables(con)
    first = build(con, alpha=1.0)
    assert set(first) == set(DERIVED_TABLES)
    assert first["plays_enriched"] == 24  # punts and null-ppa plays filtered out
    for table in DERIVED_TABLES[1:]:
        teams = {r[0] for r in con.execute(f"SELECT DISTINCT team FROM {table}").fetchall()}
        assert teams <= {"Alpha", "Beta"} and teams, table
    assert build(con, alpha=1.0) == first  # CREATE OR REPLACE, not append
    alpha = con.execute("SELECT epa_per_play, success_rate FROM team_offense WHERE team = 'Alpha'").fetchone()
    assert alpha[0] == pytest.approx(0.4) and alpha[1] > 0.5
    splits = {r[0] for r in con.execute("SELECT DISTINCT split FROM team_splits").fetchall()}
    assert splits == {"down", "quarter", "score_state", "venue", "opp_conference"}
    adjusted = con.execute(
        "SELECT off_epa_adj, def_epa_adj, off_sr_adj FROM team_adjusted WHERE team = 'Alpha'"
    ).fetchone()
    assert all(v is not None for v in adjusted)
    assert con.execute("SELECT margin FROM team_turnovers WHERE team = 'Alpha'").fetchone()[0] == 1


def test_build_honours_garbage_setting():
    con = connect(":memory:")
    seed_raw_tables(con)
    assert build(con, garbage=GarbageTime.parse("off"), alpha=1.0)["team_offense"] == 2
