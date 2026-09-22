from datetime import datetime, timedelta, timezone

from psu import db
from psu.client import CachedClient
from psu.ingest import ingest

NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
SEASON_LEVEL = 7  # games, drives, lines, advanced_season, ratings_sp, talent, recruiting
WEEKLY = 3  # plays, team_game_stats, player_game_stats


def run(tmp_path, fake, seasons, now=NOW, con=None):
    con = con or db.connect(tmp_path / "psu.duckdb")
    client = CachedClient(tmp_path / "raw", fake, min_interval_s=0, now=lambda: now)
    return con, ingest(client, con, seasons, current=2026, now=now)


def test_completed_season_rerun_makes_no_api_calls_and_no_duplicates(tmp_path, fake_cfbd):
    con, first = run(tmp_path, fake_cfbd, [2024])
    assert first.api_calls == 1 + SEASON_LEVEL + 3 * WEEKLY  # calendar + season-level + 3 game weeks
    assert first.row_counts["plays"] == 9 and first.row_counts["games"] == 2
    assert all(n > 0 for n in first.row_counts.values())
    fake_cfbd.calls.clear()
    _, second = run(tmp_path, fake_cfbd, [2024], now=NOW + timedelta(days=30), con=con)
    assert second.api_calls == 0 and fake_cfbd.calls == []
    assert second.row_counts == first.row_counts


def test_non_game_season_types_are_skipped(tmp_path, fake_cfbd):
    run(tmp_path, fake_cfbd, [2024])
    assert [c for c in fake_cfbd.calls if c[1].get("season_type") == "allstar"] == []


def test_current_season_skips_future_weeks_and_refreshes_only_live_data(tmp_path, fake_cfbd):
    con, first = run(tmp_path, fake_cfbd, [2026])
    weeks = {(p["season_type"], p["week"]) for e, p in fake_cfbd.calls if e == "plays"}
    assert weeks == {("regular", 1), ("regular", 4)}  # week 5 and the postseason haven't started
    assert first.api_calls == 1 + SEASON_LEVEL + 2 * WEEKLY

    fake_cfbd.calls.clear()
    _, second = run(tmp_path, fake_cfbd, [2026], now=NOW + timedelta(hours=25), con=con)
    refetched = {(e, p.get("week")) for e, p in fake_cfbd.calls}
    assert ("plays", 1) not in refetched  # week 1 ended more than 3 days ago: final
    assert ("plays", 4) in refetched  # week 4 is still in progress
    assert second.api_calls == 1 + SEASON_LEVEL + WEEKLY
    assert second.row_counts["plays"] == first.row_counts["plays"]


def test_deleted_database_is_rebuilt_from_cache_without_api_calls(tmp_path, fake_cfbd):
    con, first = run(tmp_path, fake_cfbd, [2024])
    con.close()
    (tmp_path / "psu.duckdb").unlink()
    fake_cfbd.calls.clear()
    _, second = run(tmp_path, fake_cfbd, [2024])
    assert second.api_calls == 0
    assert second.row_counts == first.row_counts
