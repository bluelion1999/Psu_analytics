"""DuckDB storage: declared table schemas and an idempotent upsert."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSpec:
    name: str
    key: tuple[str, ...]
    columns: dict[str, str]
    extra_type: str | None = None  # type for undeclared columns; None drops them


def _sides(**cols: str) -> dict[str, str]:
    return {f"{side}_{col}": typ for side in ("home", "away") for col, typ in cols.items()}


_GAME_CONTEXT = {"season": "INTEGER", "week": "INTEGER", "season_type": "VARCHAR"}

SPECS: dict[str, TableSpec] = {
    spec.name: spec
    for spec in [
        TableSpec(
            "games",
            ("id",),
            {
                "id": "BIGINT",
                **_GAME_CONTEXT,
                "start_date": "TIMESTAMP",
                "start_time_tbd": "BOOLEAN",
                "completed": "BOOLEAN",
                "neutral_site": "BOOLEAN",
                "conference_game": "BOOLEAN",
                "attendance": "INTEGER",
                "venue_id": "INTEGER",
                "venue": "VARCHAR",
                **_sides(
                    id="INTEGER",
                    team="VARCHAR",
                    conference="VARCHAR",
                    classification="VARCHAR",
                    points="INTEGER",
                    line_scores="VARCHAR",
                    postgame_win_probability="DOUBLE",
                    pregame_elo="DOUBLE",
                    postgame_elo="DOUBLE",
                ),
                "excitement_index": "DOUBLE",
                "notes": "VARCHAR",
                "playoff_competition": "VARCHAR",
                "playoff_format": "VARCHAR",
                "playoff_round": "VARCHAR",
                "playoff_round_name": "VARCHAR",
                "playoff_bracket_slot": "VARCHAR",
                "playoff_home_seed": "INTEGER",
                "playoff_away_seed": "INTEGER",
                "playoff_bowl_name": "VARCHAR",
            },
        ),
        TableSpec(
            "plays",
            ("id",),
            {
                "id": "VARCHAR",
                "drive_id": "VARCHAR",
                "game_id": "BIGINT",
                **_GAME_CONTEXT,
                "drive_number": "INTEGER",
                "play_number": "INTEGER",
                "offense": "VARCHAR",
                "offense_conference": "VARCHAR",
                "offense_score": "INTEGER",
                "defense": "VARCHAR",
                "defense_conference": "VARCHAR",
                "defense_score": "INTEGER",
                "home": "VARCHAR",
                "away": "VARCHAR",
                "period": "INTEGER",
                "clock_minutes": "INTEGER",
                "clock_seconds": "INTEGER",
                "offense_timeouts": "INTEGER",
                "defense_timeouts": "INTEGER",
                "yardline": "INTEGER",
                "yards_to_goal": "INTEGER",
                "down": "INTEGER",
                "distance": "INTEGER",
                "yards_gained": "INTEGER",
                "scoring": "BOOLEAN",
                "play_type": "VARCHAR",
                "play_text": "VARCHAR",
                "ppa": "DOUBLE",
                "wallclock": "VARCHAR",
            },
        ),
        TableSpec(
            "drives",
            ("id",),
            {
                "id": "VARCHAR",
                "game_id": "BIGINT",
                "season": "INTEGER",
                "offense": "VARCHAR",
                "offense_conference": "VARCHAR",
                "defense": "VARCHAR",
                "defense_conference": "VARCHAR",
                "drive_number": "INTEGER",
                "scoring": "BOOLEAN",
                "start_period": "INTEGER",
                "start_yardline": "INTEGER",
                "start_yards_to_goal": "INTEGER",
                "start_time_minutes": "INTEGER",
                "start_time_seconds": "INTEGER",
                "end_period": "INTEGER",
                "end_yardline": "INTEGER",
                "end_yards_to_goal": "INTEGER",
                "end_time_minutes": "INTEGER",
                "end_time_seconds": "INTEGER",
                "elapsed_minutes": "INTEGER",
                "elapsed_seconds": "INTEGER",
                "plays": "INTEGER",
                "yards": "INTEGER",
                "drive_result": "VARCHAR",
                "is_home_offense": "BOOLEAN",
                "start_offense_score": "INTEGER",
                "start_defense_score": "INTEGER",
                "end_offense_score": "INTEGER",
                "end_defense_score": "INTEGER",
            },
        ),
        TableSpec(
            "team_game_stats",
            ("game_id", "team", "category"),
            {
                "game_id": "BIGINT",
                **_GAME_CONTEXT,
                "team_id": "INTEGER",
                "team": "VARCHAR",
                "conference": "VARCHAR",
                "home_away": "VARCHAR",
                "points": "INTEGER",
                "category": "VARCHAR",
                "stat": "VARCHAR",
            },
        ),
        TableSpec(
            "player_game_stats",
            ("game_id", "team", "category", "stat_type", "athlete_id"),
            {
                "game_id": "BIGINT",
                **_GAME_CONTEXT,
                "team": "VARCHAR",
                "conference": "VARCHAR",
                "home_away": "VARCHAR",
                "category": "VARCHAR",
                "stat_type": "VARCHAR",
                "athlete_id": "VARCHAR",
                "athlete_name": "VARCHAR",
                "stat": "VARCHAR",
            },
        ),
        TableSpec(
            "advanced_season",
            ("season", "team"),
            {"season": "INTEGER", "team": "VARCHAR", "conference": "VARCHAR"},
            extra_type="DOUBLE",
        ),
        TableSpec(
            "ratings_sp",
            ("year", "team"),
            {"year": "INTEGER", "team": "VARCHAR", "conference": "VARCHAR"},
            extra_type="DOUBLE",
        ),
        TableSpec("talent", ("year", "team"), {"year": "INTEGER", "team": "VARCHAR", "talent": "DOUBLE"}),
        TableSpec(
            "recruiting",
            ("year", "team"),
            {"year": "INTEGER", "team": "VARCHAR", "rank": "INTEGER", "points": "DOUBLE"},
        ),
        TableSpec(
            "returning_production",
            ("season", "team"),
            {
                "season": "INTEGER",
                "team": "VARCHAR",
                "conference": "VARCHAR",
                "percent_ppa": "DOUBLE",
                "percent_passing_ppa": "DOUBLE",
                "percent_receiving_ppa": "DOUBLE",
                "percent_rushing_ppa": "DOUBLE",
                "usage": "DOUBLE",
            },
            extra_type="DOUBLE",
        ),
        TableSpec(
            "lines",
            ("game_id", "provider"),
            {
                "game_id": "BIGINT",
                **_GAME_CONTEXT,
                "start_date": "TIMESTAMP",
                **_sides(
                    team_id="INTEGER",
                    team="VARCHAR",
                    conference="VARCHAR",
                    classification="VARCHAR",
                    score="INTEGER",
                ),
                "provider": "VARCHAR",
                "spread": "DOUBLE",
                "formatted_spread": "VARCHAR",
                "spread_open": "DOUBLE",
                "over_under": "DOUBLE",
                "over_under_open": "DOUBLE",
                "home_moneyline": "DOUBLE",
                "away_moneyline": "DOUBLE",
            },
        ),
    ]
}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect(path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if str(path) != ":memory:" and not read_only:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(
            "CREATE TABLE IF NOT EXISTS _loads ("
            "endpoint VARCHAR, cache_key VARCHAR, fetched_at VARCHAR, rows INTEGER, "
            "PRIMARY KEY (endpoint, cache_key))"
        )
    return con


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    result = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]).fetchall()
    return {r[0] for r in result}


def _ensure_table(con: duckdb.DuckDBPyConnection, spec: TableSpec, extras: list[str]) -> None:
    cols = ", ".join(f"{_q(c)} {t}" for c, t in spec.columns.items())
    con.execute(f"CREATE TABLE IF NOT EXISTS {_q(spec.name)} ({cols})")
    existing = _table_columns(con, spec.name)
    # Add newly declared columns that are missing
    for col, typ in spec.columns.items():
        if col not in existing:
            con.execute(f"ALTER TABLE {_q(spec.name)} ADD COLUMN {_q(col)} {typ}")
    # Add extra columns
    for col in extras:
        if col not in existing:
            con.execute(f"ALTER TABLE {_q(spec.name)} ADD COLUMN {_q(col)} {spec.extra_type}")


def upsert(
    con: duckdb.DuckDBPyConnection,
    spec: TableSpec,
    df: pd.DataFrame,
    replace_scope: dict[str, Any] | None = None,
) -> int:
    """Insert df into spec's table, replacing rows with the same key.

    If replace_scope is given and df is non-empty, first delete every row in that scope (e.g. a
    season) so rows CFBD removed on a refresh don't linger as ghosts. An empty df is a no-op: it
    never wipes a slice just because the response happened to be empty.
    Returns rows written.
    """
    if df.empty:
        return 0
    extras = [c for c in df.columns if c not in spec.columns] if spec.extra_type else []
    types = {**spec.columns, **{c: spec.extra_type for c in extras}}
    # Register only the columns we keep: stray nested/list columns can confuse DuckDB's type sniffing.
    incoming = df[[c for c in df.columns if c in types]]
    select = ", ".join(
        f"TRY_CAST({_q(c)} AS {t}) AS {_q(c)}" if c in incoming.columns else f"CAST(NULL AS {t}) AS {_q(c)}"
        for c, t in types.items()
    )
    keys = ", ".join(_q(k) for k in spec.key)
    not_null = " AND ".join(f"{_q(k)} IS NOT NULL" for k in spec.key)
    match = " AND ".join(f"s.{_q(k)} = {_q(spec.name)}.{_q(k)}" for k in spec.key)

    _ensure_table(con, spec, extras)
    con.register("_incoming", incoming)
    try:
        con.execute("BEGIN")
        if replace_scope:
            where = " AND ".join(f"{_q(col)} = ?" for col in replace_scope)
            con.execute(f"DELETE FROM {_q(spec.name)} WHERE {where}", list(replace_scope.values()))
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE _staged AS "
            f"SELECT * FROM (SELECT {select} FROM _incoming) WHERE {not_null} "
            f"QUALIFY row_number() OVER (PARTITION BY {keys}) = 1"
        )
        written = con.execute("SELECT count(*) FROM _staged").fetchone()[0]
        con.execute(f"DELETE FROM {_q(spec.name)} WHERE EXISTS (SELECT 1 FROM _staged s WHERE {match})")
        con.execute(f"INSERT INTO {_q(spec.name)} BY NAME SELECT * FROM _staged")
        con.execute("DROP TABLE _staged")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_incoming")
    dropped = len(incoming) - written
    if dropped:
        log.warning("%s: skipped %d rows with a missing key or a duplicate key", spec.name, dropped)
    return written


def already_loaded(con: duckdb.DuckDBPyConnection, endpoint: str, cache_key: str, fetched_at: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM _loads WHERE endpoint = ? AND cache_key = ? AND fetched_at = ?",
        [endpoint, cache_key, fetched_at],
    ).fetchone()
    return row is not None


def record_load(con: duckdb.DuckDBPyConnection, endpoint: str, cache_key: str, fetched_at: str, rows: int) -> None:
    con.execute("INSERT OR REPLACE INTO _loads VALUES (?, ?, ?, ?)", [endpoint, cache_key, fetched_at, rows])


def row_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    existing = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    return {
        name: con.execute(f"SELECT count(*) FROM {_q(name)}").fetchone()[0] if name in existing else 0 for name in SPECS
    }
