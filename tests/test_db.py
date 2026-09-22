import pandas as pd
import pytest

from psu.db import SPECS, TableSpec, already_loaded, connect, record_load, row_counts, upsert

SPEC = TableSpec("t", key=("id",), columns={"id": "BIGINT", "name": "VARCHAR", "score": "INTEGER"})
WIDE = TableSpec("w", key=("season", "team"), columns={"season": "INTEGER", "team": "VARCHAR"}, extra_type="DOUBLE")


@pytest.fixture
def con():
    c = connect(":memory:")
    yield c
    c.close()


def rows(con, sql):
    return con.execute(sql).fetchall()


def test_upsert_is_idempotent(con):
    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "score": [10, 20]})
    assert upsert(con, SPEC, df) == 2
    assert upsert(con, SPEC, df) == 2
    assert rows(con, "SELECT id, name, score FROM t ORDER BY id") == [(1, "a", 10), (2, "b", 20)]


def test_upsert_replaces_changed_rows(con):
    upsert(con, SPEC, pd.DataFrame({"id": [1], "name": ["a"], "score": [10]}))
    upsert(con, SPEC, pd.DataFrame({"id": [1], "name": ["a"], "score": [17]}))
    assert rows(con, "SELECT id, score FROM t") == [(1, 17)]


def test_missing_columns_become_null_and_unknown_columns_are_dropped(con):
    upsert(con, SPEC, pd.DataFrame({"id": [1], "surprise": ["x"]}))
    assert rows(con, "SELECT id, name, score FROM t") == [(1, None, None)]
    assert "surprise" not in [r[0] for r in rows(con, "DESCRIBE t")]


def test_duplicate_keys_in_one_batch_are_collapsed(con):
    assert upsert(con, SPEC, pd.DataFrame({"id": [1, 1, 2], "name": ["a", "a", "b"]})) == 2
    assert rows(con, "SELECT count(*) FROM t") == [(2,)]


def test_rows_with_null_keys_are_skipped(con):
    upsert(con, SPEC, pd.DataFrame({"id": [1, None], "name": ["a", "b"]}))
    assert rows(con, "SELECT id FROM t") == [(1,)]


def test_empty_frame_is_a_no_op(con):
    assert upsert(con, SPEC, pd.DataFrame()) == 0


def test_wide_tables_grow_new_columns_as_double(con):
    upsert(con, WIDE, pd.DataFrame({"season": [2014], "team": ["Penn State"], "offense_ppa": [0.1]}))
    upsert(
        con,
        WIDE,
        pd.DataFrame({"season": [2025], "team": ["Penn State"], "offense_ppa": [0.3], "offense_havoc_total": [0.2]}),
    )
    got = rows(con, "SELECT season, offense_ppa, offense_havoc_total FROM w ORDER BY season")
    assert got == [(2014, 0.1, None), (2025, 0.3, 0.2)]


def test_bad_values_in_non_key_columns_become_null(con):
    upsert(con, SPEC, pd.DataFrame({"id": [1], "score": ["n/a"]}))
    assert rows(con, "SELECT score FROM t") == [(None,)]


def test_load_log_round_trip(con):
    ts = "2026-09-22T12:00:00+00:00"
    assert not already_loaded(con, "games", "year=2024", ts)
    record_load(con, "games", "year=2024", ts, 5)
    assert already_loaded(con, "games", "year=2024", ts)
    assert not already_loaded(con, "games", "year=2024", "2026-09-23T12:00:00+00:00")
    record_load(con, "games", "year=2024", "2026-09-23T12:00:00+00:00", 6)  # re-recording replaces
    assert rows(con, "SELECT count(*) FROM _loads") == [(1,)]


def test_row_counts_lists_every_table_even_before_load(con):
    counts = row_counts(con)
    assert set(counts) == set(SPECS)
    assert all(v == 0 for v in counts.values())


def test_replace_scope_removes_rows_missing_from_a_refresh(con):
    scoped = TableSpec("scoped", key=("id",), columns={"id": "BIGINT", "season": "INTEGER"})
    upsert(con, scoped, pd.DataFrame({"id": [1, 2, 3], "season": [2026, 2026, 2026]}))
    upsert(con, scoped, pd.DataFrame({"id": [9], "season": [2025]}))

    upsert(
        con,
        scoped,
        pd.DataFrame({"id": [1, 2], "season": [2026, 2026]}),
        replace_scope={"season": 2026},
    )
    assert {r[0] for r in rows(con, "SELECT id FROM scoped")} == {1, 2, 9}

    assert upsert(con, scoped, pd.DataFrame(), replace_scope={"season": 2026}) == 0
    assert {r[0] for r in rows(con, "SELECT id FROM scoped")} == {1, 2, 9}


def test_new_declared_columns_are_added_to_an_existing_table(con):
    spec_v1 = TableSpec("mig", key=("id",), columns={"id": "BIGINT", "a": "VARCHAR"})
    upsert(con, spec_v1, pd.DataFrame({"id": [1], "a": ["x"]}))

    spec_v2 = TableSpec("mig", key=("id",), columns={"id": "BIGINT", "a": "VARCHAR", "b": "INTEGER"})
    upsert(con, spec_v2, pd.DataFrame({"id": [2], "a": ["y"], "b": [42]}))

    got = rows(con, "SELECT id, a, b FROM mig ORDER BY id")
    assert got == [(1, "x", None), (2, "y", 42)]
