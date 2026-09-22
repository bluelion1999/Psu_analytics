"""Pull CFBD data for a set of seasons into DuckDB, calling the API only for data not already cached."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb

from psu import db
from psu.client import CacheEntry, CachedClient
from psu.flatten import FLATTENERS

log = logging.getLogger(__name__)

WEEKLY_ENDPOINTS = ("plays", "team_game_stats", "player_game_stats")
GAME_SEASON_TYPES = ("regular", "postseason")


@dataclass(frozen=True)
class Request:
    endpoint: str
    params: dict[str, Any]
    max_age: timedelta | None  # None: a cached copy is final


@dataclass(frozen=True)
class IngestResult:
    api_calls: int
    row_counts: dict[str, int]


def season_requests(season: int) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("games", {"year": season, "season_type": "both", "classification": "fbs"}),
        ("drives", {"year": season, "season_type": "both", "classification": "fbs"}),
        ("lines", {"year": season, "season_type": "both"}),
        ("advanced_season", {"year": season, "classification": "fbs"}),
        ("ratings_sp", {"year": season}),
        ("talent", {"year": season}),
        ("recruiting", {"year": season}),
    ]


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def weekly_requests(
    season: int,
    calendar: list[dict[str, Any]],
    *,
    now: datetime,
    current: int,
    final_after: timedelta,
    refresh_after: timedelta,
) -> list[Request]:
    requests = []
    for week in calendar:
        season_type = week.get("seasonType")
        if season_type not in GAME_SEASON_TYPES:
            continue
        max_age: timedelta | None = None
        if season == current:
            start, end = _parse_ts(week.get("startDate")), _parse_ts(week.get("endDate"))
            if start is None or start > now:
                continue  # not started: nothing to fetch, and never cache an empty week as final
            if end is None or end > now - final_after:
                max_age = refresh_after
        params = {"year": season, "week": week["week"], "season_type": season_type, "classification": "fbs"}
        requests.extend(Request(endpoint, params, max_age) for endpoint in WEEKLY_ENDPOINTS)
    return requests


def _load(con: duckdb.DuckDBPyConnection, entry: CacheEntry, params: dict[str, Any]) -> int:
    fetched_at = entry.fetched_at.isoformat()
    if db.already_loaded(con, entry.endpoint, entry.key, fetched_at):
        return 0
    df = FLATTENERS[entry.endpoint](entry.data, params)
    rows = db.upsert(con, db.SPECS[entry.endpoint], df)
    db.record_load(con, entry.endpoint, entry.key, fetched_at, rows)
    return rows


def ingest(
    client: CachedClient,
    con: duckdb.DuckDBPyConnection,
    seasons: list[int],
    *,
    current: int,
    now: datetime | None = None,
    final_after: timedelta = timedelta(days=3),
    refresh_after: timedelta = timedelta(hours=24),
) -> IngestResult:
    now = now or datetime.now(timezone.utc)
    for season in seasons:
        season_age = refresh_after if season == current else None
        calendar = client.get("calendar", {"year": season}, max_age=season_age).data
        requests = [Request(e, p, season_age) for e, p in season_requests(season)]
        requests += weekly_requests(
            season, calendar, now=now, current=current, final_after=final_after, refresh_after=refresh_after
        )
        log.info("Season %d: %d requests", season, len(requests))
        for req in requests:
            entry = client.get(req.endpoint, req.params, max_age=req.max_age)
            _load(con, entry, req.params)
    return IngestResult(client.api_calls, db.row_counts(con))
