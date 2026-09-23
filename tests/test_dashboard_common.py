from pathlib import Path

import duckdb
import pytest

from conftest import seed_dashboard_db
from psu.dashboard.common import (
    MissingData, benchmark, conference_members, db_path, read, require, seasons, team_conference, team_games,
)
from psu.db import connect


@pytest.fixture(scope="module")
def con():
    c = connect(":memory:")
    seed_dashboard_db(c)
    yield c
    c.close()


def test_seasons_newest_first(con):
    assert seasons(con) == [2027, 2026, 2025]


def test_team_games_from_the_teams_side(con):
    g = team_games(con, 2025, "Penn State").set_index("game_id")
    assert list(g.index) == [101, 103]
    assert g.loc[103, "venue"] == "away" and g.loc[103, "opponent"] == "Ohio State"
    assert (g.loc[103, "team_points"], g.loc[103, "opp_points"], g.loc[103, "result"]) == (27, 24, "W")
    assert bool(g.loc[103, "conference_game"]) and not bool(g.loc[101, "conference_game"])
    assert bool(g.loc[101, "is_home"]) and not bool(g.loc[103, "is_home"])


def test_unplayed_games_have_no_result(con):
    g = team_games(con, 2026, "Penn State").set_index("game_id")
    assert not bool(g.loc[203, "completed"]) and g.loc[203, "result"] is None
    assert bool(g.loc[201, "completed"]) and g.loc[201, "result"] == "W"


def test_conference_lookup(con):
    assert team_conference(con, 2025, "Penn State") == "Big Ten"
    assert conference_members(con, 2025, "Big Ten") == ["Ohio State", "Penn State"]
    assert team_conference(con, 2025, "Nobody") is None


def test_benchmark_matches_the_metric_table(con):
    rows = con.execute("SELECT team, epa_per_play FROM team_offense WHERE season = 2025").df().set_index("team")
    b = benchmark(con, 2025, "Penn State", "team_offense", "epa_per_play", {"Penn State", "Ohio State"})
    assert b["value"] == pytest.approx(rows.loc["Penn State", "epa_per_play"])
    assert b["conference_avg"] == pytest.approx(rows.loc[["Penn State", "Ohio State"], "epa_per_play"].mean())
    assert b["national_avg"] == pytest.approx(rows["epa_per_play"].mean())


def test_require_names_the_command():
    c = connect(":memory:")
    with pytest.raises(MissingData, match="psu train"):
        require(c, "game_predictions")
    with pytest.raises(MissingData, match="psu ingest"):
        require(c, "games")


def test_db_path_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PSU_DB_PATH", str(tmp_path / "copy.duckdb"))
    assert db_path() == tmp_path / "copy.duckdb"


def test_read_missing_file_is_missing_data(tmp_path):
    with pytest.raises(MissingData, match="psu ingest"):
        read(lambda c: 1, path=tmp_path / "nope.duckdb")


def test_read_opens_read_only(tmp_path):
    path = tmp_path / "psu.duckdb"
    connect(path).close()
    assert read(lambda c: c.execute("SELECT 41 + 1").fetchone()[0], path=path) == 42
    with pytest.raises(MissingData):  # a write through the dashboard connection is refused
        read(lambda c: c.execute("CREATE TABLE t (x INTEGER)"), path=path)


def test_read_busy_database_is_missing_data(tmp_path):
    path = tmp_path / "psu.duckdb"
    writer = duckdb.connect(str(path))  # another command holding a write connection
    try:
        with pytest.raises(MissingData, match="Try again"):
            read(lambda c: 1, path=path)
    finally:
        writer.close()
