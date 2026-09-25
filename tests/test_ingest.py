from datetime import UTC, datetime, timedelta

from psu import db
from psu.client import CachedClient
from psu.ingest import ingest

NOW = datetime(2026, 9, 22, 12, tzinfo=UTC)
SEASON_LEVEL = 8  # games, drives, lines, advanced_season, ratings_sp, talent, recruiting, returning_production
WEEKLY = 3  # plays, team_game_stats, player_game_stats


def run(tmp_path, fake, seasons, now=NOW, con=None, current=2026):
    con = con or db.connect(tmp_path / "psu.duckdb")
    client = CachedClient(tmp_path / "raw", fake, min_interval_s=0, now=lambda: now)
    return con, ingest(client, con, seasons, current=current, now=now)


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
    # talent, recruiting, returning_production, and calendar use the 7-day slow-refresh window
    # not refetched: games, drives, lines, advanced_season, ratings_sp + week 4's three weekly endpoints.
    assert second.api_calls == 5 + WEEKLY
    assert "returning_production" not in {e for e, _ in fake_cfbd.calls}  # slow refresh, like talent
    assert second.row_counts["plays"] == first.row_counts["plays"]


def test_in_progress_week_is_refetched_once_after_it_becomes_final(tmp_path, fake_cfbd):
    con, first = run(tmp_path, fake_cfbd, [2026])

    fake_cfbd.calls.clear()
    _, second = run(tmp_path, fake_cfbd, [2026], now=datetime(2026, 10, 5, 12, tzinfo=UTC), con=con)
    week4_calls = [c for c in fake_cfbd.calls if c[1].get("week") == 4]
    assert len(week4_calls) == WEEKLY  # plays, team_game_stats, player_game_stats: each refetched once
    assert {e for e, _ in week4_calls} == set(("plays", "team_game_stats", "player_game_stats"))

    fake_cfbd.calls.clear()
    _, third = run(tmp_path, fake_cfbd, [2026], now=datetime(2026, 10, 6, 12, tzinfo=UTC), con=con)
    assert [c for c in fake_cfbd.calls if c[0] == "plays" and c[1].get("week") == 4] == []


def test_season_level_data_is_refetched_once_after_the_season_ends(tmp_path, fake_cfbd):
    con, first = run(tmp_path, fake_cfbd, [2026], current=2026)

    fake_cfbd.calls.clear()
    _, second = run(tmp_path, fake_cfbd, [2026], now=datetime(2027, 3, 2, 12, tzinfo=UTC), con=con, current=2027)
    assert len([c for c in fake_cfbd.calls if c[0] == "games"]) == 1

    fake_cfbd.calls.clear()
    _, third = run(tmp_path, fake_cfbd, [2026], now=datetime(2027, 3, 3, 12, tzinfo=UTC), con=con, current=2027)
    assert [c for c in fake_cfbd.calls if c[0] == "games"] == []


def test_deleted_database_is_rebuilt_from_cache_without_api_calls(tmp_path, fake_cfbd):
    con, first = run(tmp_path, fake_cfbd, [2024])
    con.close()
    (tmp_path / "psu.duckdb").unlink()
    fake_cfbd.calls.clear()
    _, second = run(tmp_path, fake_cfbd, [2024])
    assert second.api_calls == 0
    assert second.row_counts == first.row_counts
