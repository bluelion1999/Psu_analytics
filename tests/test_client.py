import json
from datetime import datetime, timedelta, timezone

import pytest

from psu.client import BudgetExceeded, CachedClient, cache_key

T0 = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = T0
        self.mono = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.mono += seconds


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        return [{"n": len(self.calls)}]


def make_client(tmp_path, fetch, clock, **kw):
    return CachedClient(
        tmp_path, fetch, now=lambda: clock.now, monotonic=lambda: clock.mono, sleep=clock.sleep, **kw
    )


def test_cache_key_is_sorted_and_skips_none():
    assert cache_key({"year": 2024, "week": 3, "team": None}) == "week=3_year=2024"
    assert cache_key({}) == "all"


def test_second_get_is_served_from_cache(tmp_path):
    clock, fetch = Clock(), Recorder()
    c = make_client(tmp_path, fetch, clock)
    first = c.get("games", {"year": 2024})
    second = c.get("games", {"year": 2024})
    assert fetch.calls == [("games", {"year": 2024})]
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.data == [{"n": 1}]
    assert c.api_calls == 1


def test_cache_survives_new_client(tmp_path):
    clock = Clock()
    make_client(tmp_path, Recorder(), clock).get("games", {"year": 2024})
    fetch = Recorder()
    c2 = make_client(tmp_path, fetch, clock)
    assert c2.get("games", {"year": 2024}).data == [{"n": 1}]
    assert fetch.calls == [] and c2.api_calls == 0


def test_max_age_refetches_stale_entries_only(tmp_path):
    clock, fetch = Clock(), Recorder()
    c = make_client(tmp_path, fetch, clock, min_interval_s=0)
    c.get("games", {"year": 2026}, max_age=timedelta(hours=24))
    clock.now += timedelta(hours=23)
    c.get("games", {"year": 2026}, max_age=timedelta(hours=24))
    assert len(fetch.calls) == 1
    clock.now += timedelta(hours=2)
    entry = c.get("games", {"year": 2026}, max_age=timedelta(hours=24))
    assert len(fetch.calls) == 2
    assert entry.data == [{"n": 2}] and entry.fetched_at == clock.now


def test_budget_blocks_the_call_that_would_exceed_it(tmp_path):
    clock, fetch = Clock(), Recorder()
    c = make_client(tmp_path, fetch, clock, max_calls=2, min_interval_s=0)
    c.get("games", {"year": 2014})
    c.get("games", {"year": 2015})
    with pytest.raises(BudgetExceeded):
        c.get("games", {"year": 2016})
    assert len(fetch.calls) == 2
    assert c.get("games", {"year": 2014}).from_cache  # cached reads still work once the budget is spent


def test_rate_limit_spaces_network_calls(tmp_path):
    clock, fetch = Clock(), Recorder()
    c = make_client(tmp_path, fetch, clock, min_interval_s=1.0)
    c.get("games", {"year": 2014})
    clock.mono += 0.25
    c.get("games", {"year": 2015})
    c.get("games", {"year": 2014})  # cache hit: no sleep
    assert clock.slept == [pytest.approx(0.75)]


def test_failed_fetch_leaves_no_cache_file(tmp_path):
    clock = Clock()

    def boom(endpoint, params):
        raise ConnectionError("network down")

    c = make_client(tmp_path, boom, clock)
    with pytest.raises(ConnectionError):
        c.get("plays", {"year": 2024, "week": 1})
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert c.api_calls == 1  # a failed request still counts against quota


def test_cache_file_layout(tmp_path):
    clock = Clock()
    c = make_client(tmp_path, Recorder(), clock)
    c.get("plays", {"year": 2024, "week": 1, "season_type": "regular"})
    path = tmp_path / "plays" / "season_type=regular_week=1_year=2024.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["params"] == {"year": 2024, "week": 1, "season_type": "regular"}
    assert payload["data"] == [{"n": 1}]
