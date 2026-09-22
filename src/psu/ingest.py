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
SLOW_REFRESH = timedelta(days=7)  # talent, recruiting and the calendar change rarely


@dataclass(frozen=True)
class Request:
    endpoint: str
    params: dict[str, Any]
    max_age: timedelta | None  # None: a cached copy is final (subject to final_at)
    final_at: datetime | None = None  # once a copy is fetched at/after this, it's final forever


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
    final_after: timedelta,
    refresh_after: timedelta,
) -> list[Request]:
    requests = []
    for week in calendar:
        season_type = week.get("seasonType")
        if season_type not in GAME_SEASON_TYPES:
            continue
        start, end = _parse_ts(week.get("startDate")), _parse_ts(week.get("endDate"))
        if start is None or start > now:
            continue  # not started: nothing to fetch, and never cache an empty week as final
        final_at = end + final_after if end is not None else None
        params = {"year": season, "week": week["week"], "season_type": season_type, "classification": "fbs"}
        requests.extend(Request(endpoint, params, refresh_after, final_at) for endpoint in WEEKLY_ENDPOINTS)
    return requests


def _season_final_at(calendar: list[dict[str, Any]], final_after: timedelta) -> datetime | None:
    ends = [ts for ts in (_parse_ts(week.get("endDate")) for week in calendar) if ts is not None]
    return max(ends) + final_after if ends else None


def _replace_scope(endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if endpoint in WEEKLY_ENDPOINTS:
        return {"season": params["year"], "week": params["week"], "season_type": params["season_type"]}
    if endpoint in ("games", "drives", "lines"):
        return {"season": params["year"]}
    return None


def _load(con: duckdb.DuckDBPyConnection, entry: CacheEntry, params: dict[str, Any]) -> int:
    fetched_at = entry.fetched_at.isoformat()
    if db.already_loaded(con, entry.endpoint, entry.key, fetched_at):
        return 0
    df = FLATTENERS[entry.endpoint](entry.data, params)
    scope = _replace_scope(entry.endpoint, params)
    rows = db.upsert(con, db.SPECS[entry.endpoint], df, replace_scope=scope)
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
        calendar_max_age = SLOW_REFRESH if season >= current else None
        calendar = client.get("calendar", {"year": season}, max_age=calendar_max_age).data
        season_final_at = _season_final_at(calendar, final_after)
        requests = [
            Request(e, p, SLOW_REFRESH if e in ("talent", "recruiting") else refresh_after, season_final_at)
            for e, p in season_requests(season)
        ]
        requests += weekly_requests(
            season, calendar, now=now, final_after=final_after, refresh_after=refresh_after
        )
        log.info("Season %d: %d requests", season, len(requests))
        for req in requests:
            entry = client.get(req.endpoint, req.params, max_age=req.max_age, final_at=req.final_at)
            _load(con, entry, req.params)
    return IngestResult(client.api_calls, db.row_counts(con))
