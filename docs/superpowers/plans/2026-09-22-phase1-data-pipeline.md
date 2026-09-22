# Phase 1 — Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **In this repo:** dispatch each task to the `psu-implementer` agent and review it with the `psu-reviewer` agent (both in `.claude/agents/`). Give an agent only its own task section, plus **Global Constraints** and **Review Focus**. Don't give it the whole plan.

**Goal:** `psu ingest --seasons 2022-2026` (the last 5 seasons) pulls all-FBS CFBD data into `data/psu.duckdb` (10 tables). Every response is cached under `data/raw/`, and re-running makes zero API calls for completed seasons.

**Architecture:** Each module does one job:
- `client.py` caches every response (keyed by endpoint and params) and enforces a per-run call budget and rate limit. It has no cfbd dependency: the fetch function is injected.
- `cfbd_api.py` is the only module that talks to the `cfbd` package, and returns raw JSON.
- `flatten.py` turns nested camelCase JSON into flat snake_case DataFrames.
- `db.py` owns the declared table schemas and an idempotent upsert.
- `ingest.py` decides which requests to make for a season.
- `cli.py` wires it all together.

**Tech Stack:** Python ≥3.11 (dev machine has 3.13), `cfbd` 5.29 (pydantic-v1 generated client), DuckDB ≥1.1, pandas ≥2.2 (3.x installed), python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/psu-analytics-build-plan.md` (this plan covers "Phase 1 — Data pipeline" only; later phases get their own plans)

## Global Constraints

- Python `>=3.11`; package lives at repo root as `src/psu/` (the user chose repo root over a `psu-analytics/` subfolder).
- `cfbd>=5.29,<6`. Verified method names (cfbd 5.29.0): `GamesApi.get_calendar`, `GamesApi.get_games`, `PlaysApi.get_plays` (year **and** week required), `DrivesApi.get_drives` (year required), `GamesApi.get_game_team_stats` / `get_game_player_stats` (year + one of week/team/conference), `StatsApi.get_advanced_season_stats`, `RatingsApi.get_sp`, `TeamsApi.get_talent` (year required), `RecruitingApi.get_team_recruiting_rankings`, `BettingApi.get_lines`.
- Auth: `cfbd.Configuration(access_token=<key>)` (bearer token). Key comes from env var `CFBD_API_KEY`, loaded from `.env` by python-dotenv. **Never commit `.env` or the key.**
- The free tier has a monthly call limit (about 1,000/month; the user should check their quota on collegefootballdata.com). A full 2022–2026 pull is about 295 calls, so it fits under the default 300-call per-run budget. **No task except Task 8 may make a live API call.** Every test uses an injected fake fetch.
- Cache every response to `data/raw/<endpoint>/<params>.json`. Never re-request a cached response unless it belongs to the current season and is not yet final.
- Pull all FBS data (`classification="fbs"` where the endpoint accepts it), seasons 2022 (`FIRST_SEASON`) through the current season: the user limited the pull to the last 5 seasons.
- Tables: `games`, `plays`, `drives`, `team_game_stats`, `player_game_stats`, `advanced_season`, `ratings_sp`, `talent`, `recruiting`, `lines`.
- Git: work on branch `feat/phase1-data-pipeline`, never on `main`. Each task ends with one Conventional Commits commit (`feat(scope): ...`, `test: ...`, `docs: ...`) that stages only that task's files by explicit path (never `git add -A` or `git add .`), and ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The orchestrator pushes the branch and opens a PR into `main` after Task 8.
- Commands below are for Windows PowerShell from the repo root (`C:\Users\bryce\Documents\PSU_analytics`), using the project venv at `.venv`.

## Review Focus

1. **Burning the monthly quota:** a bug that re-fetches cached data, or a loop, must not burn the monthly quota. Expected: a hard per-run `max_calls` cap, and a re-run of a completed season makes 0 calls. Tested in Task 2 (`test_budget_blocks_the_call_that_would_exceed_it`) and Task 6 (`test_completed_season_rerun_makes_no_api_calls_and_no_duplicates`).
2. **Interrupted run** (network error, Ctrl-C) mid-request: expected to leave no half-written cache file that later passes as valid. Tested in Task 2 (`test_failed_fetch_leaves_no_cache_file`).
3. **Older seasons with missing or null fields** (no `ppa`, missing nested lists, bad values): expected to load as NULLs, not crash. Tested in Task 4 (`test_missing_columns_become_null_and_unknown_columns_are_dropped`, `test_bad_values_in_non_key_columns_become_null`) and Task 5 (`test_missing_nested_lists_produce_no_rows`).
4. **Current-season weeks that haven't been played:** expected to be skipped, never cached as permanent empty results. Only in-progress data is refreshed. Tested in Task 6 (`test_current_season_skips_future_weeks_and_refreshes_only_live_data`).
5. **Duplicate or null keys within one API response:** expected to collapse to one row, never duplicate across re-runs. Tested in Task 4 (`test_duplicate_keys_in_one_batch_are_collapsed`, `test_rows_with_null_keys_are_skipped`).

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, deps, `psu` console script, pytest config |
| `.gitignore`, `.env.example` | keep key, cache, and DB out of git |
| `src/psu/config.py` | team/season constants, `current_season`, `parse_seasons`, `Settings`, `load_settings` |
| `src/psu/client.py` | `CachedClient` (disk cache, budget, rate limit), `BudgetExceeded`, `MissingApiKey`, `cache_key` |
| `src/psu/cfbd_api.py` | `ENDPOINTS` registry, `make_cfbd_fetch` (raw JSON via cfbd) |
| `src/psu/db.py` | `TableSpec`, `SPECS`, `connect`, `upsert`, load log, `row_counts` |
| `src/psu/flatten.py` | per-endpoint flatteners, `FLATTENERS` registry |
| `src/psu/ingest.py` | request planning per season + load loop |
| `src/psu/cli.py` | `psu ingest`, `psu status` |
| `tests/conftest.py` | `FakeCFBD` fetch + calendars (shared by ingest/CLI tests) |
| `README.md` | setup + Phase 1 how-to |

---

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/psu/__init__.py`, `src/psu/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `psu.config.TEAM: str = "Penn State"`, `FIRST_SEASON: int = 2022`, `PROJECT_ROOT: Path`
  - `current_season(today: date | None = None) -> int`
  - `parse_seasons(spec: str, first: int = FIRST_SEASON, last: int | None = None) -> list[int]`: sorted and unique; raises `ValueError` on bad or out-of-range input
  - `@dataclass(frozen=True) Settings(api_key: str | None, raw_dir: Path, db_path: Path, current_season: int, max_calls: int = 300, min_interval_s: float = 1.0, refresh_after: timedelta = 24h, final_after: timedelta = 3 days)`
  - `load_settings(root: Path = PROJECT_ROOT) -> Settings`

- [ ] **Step 1: Create packaging and ignore files**

`pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "psu-analytics"
version = "0.1.0"
description = "Penn State football analytics on CollegeFootballData"
requires-python = ">=3.11"
dependencies = [
    "cfbd>=5.29,<6",
    "duckdb>=1.1",
    "pandas>=2.2",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
psu = "psu.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/psu"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`.gitignore`:
```gitignore
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
data/raw/
data/*.duckdb
data/*.duckdb.wal
```

`.env.example`:
```
# Free key: https://collegefootballdata.com/key
CFBD_API_KEY=
# Optional: max CFBD API calls per `psu ingest` run (default 300)
PSU_MAX_CALLS=300
```

`src/psu/__init__.py`:
```python
"""Penn State football analytics on CollegeFootballData."""
```

- [ ] **Step 2: Create the venv and install**

Run:
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```
Expected: ends with `Successfully installed ... psu-analytics-0.1.0 ...`.

(`src/psu/cli.py` doesn't exist yet, so the `psu` script won't run until Task 7. That's expected.)

- [ ] **Step 3: Write the failing tests**

`tests/test_config.py`:
```python
from datetime import date

import pytest

from psu.config import current_season, parse_seasons


def test_current_season_rolls_over_in_march():
    assert current_season(date(2026, 9, 22)) == 2026
    assert current_season(date(2027, 1, 10)) == 2026  # bowl season belongs to the prior fall
    assert current_season(date(2027, 3, 1)) == 2027


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("2024", [2024]),
        ("2022-2024", [2022, 2023, 2024]),
        ("2022, 2024-2025", [2022, 2024, 2025]),
        ("2025,2024,2025", [2024, 2025]),
    ],
)
def test_parse_seasons(spec, expected):
    assert parse_seasons(spec, first=2022, last=2026) == expected


@pytest.mark.parametrize("spec", ["", "2021", "2027", "2025-2023", "twenty", "2022-"])
def test_parse_seasons_rejects(spec):
    with pytest.raises(ValueError):
        parse_seasons(spec, first=2022, last=2026)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_config.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.config'`

- [ ] **Step 5: Implement `src/psu/config.py`**

```python
"""Project-wide settings: team, season range, paths, and API budget knobs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

TEAM = "Penn State"
FIRST_SEASON = 2022  # last 5 seasons (2022-2026); lower this to pull more history
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def current_season(today: date | None = None) -> int:
    """Seasons are named by their fall year; January/February bowl games belong to the prior season."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def parse_seasons(spec: str, first: int = FIRST_SEASON, last: int | None = None) -> list[int]:
    """Parse "2024", "2022-2026" or "2022,2024-2025" into a sorted list of seasons."""
    last = current_season() if last is None else last
    seasons: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    raise ValueError(f"start is after end in {part!r}")
                seasons.update(range(lo, hi + 1))
            else:
                seasons.add(int(part))
    except ValueError as e:
        raise ValueError(f"Bad season spec {spec!r}: {e}") from e
    if not seasons:
        raise ValueError("No seasons given")
    ordered = sorted(seasons)
    if ordered[0] < first or ordered[-1] > last:
        raise ValueError(f"Seasons must be within {first}-{last}, got {spec!r}")
    return ordered


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    raw_dir: Path
    db_path: Path
    current_season: int
    max_calls: int = 300
    min_interval_s: float = 1.0
    refresh_after: timedelta = timedelta(hours=24)  # current-season data older than this is re-fetched
    final_after: timedelta = timedelta(days=3)  # a week this long past its end date is treated as final


def load_settings(root: Path = PROJECT_ROOT) -> Settings:
    load_dotenv(root / ".env")
    return Settings(
        api_key=os.environ.get("CFBD_API_KEY") or None,
        raw_dir=root / "data" / "raw",
        db_path=root / "data" / "psu.duckdb",
        current_season=current_season(),
        max_calls=int(os.environ.get("PSU_MAX_CALLS", "300")),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_config.py`
Expected: `11 passed`

---

### Task 2: Cached client (cache, budget, rate limit)

**Files:**
- Create: `src/psu/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Fetch = Callable[[str, dict[str, Any]], list[dict[str, Any]]]`: `(endpoint, params) -> raw JSON records`
  - `class BudgetExceeded(RuntimeError)`, `class MissingApiKey(RuntimeError)`
  - `cache_key(params: dict) -> str`: e.g. `{"year": 2024, "week": 3}` → `"week=3_year=2024"`; `{}` → `"all"`
  - `@dataclass(frozen=True) CacheEntry(endpoint: str, key: str, fetched_at: datetime, data: list[dict], from_cache: bool)`
  - `CachedClient(raw_dir: Path, fetch: Fetch, *, max_calls=300, min_interval_s=1.0, now=utc-now callable, monotonic=time.monotonic, sleep=time.sleep)`
    - `.get(endpoint: str, params: dict, *, max_age: timedelta | None = None) -> CacheEntry`: `max_age=None` means any cached copy is final
    - `.api_calls: int`: network calls attempted by this instance

- [ ] **Step 1: Write the failing tests**

`tests/test_client.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_client.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.client'`

- [ ] **Step 3: Implement `src/psu/client.py`**

```python
"""CFBD access with an on-disk JSON cache, a per-run call budget and simple rate limiting.

The free CFBD tier has a monthly call limit, so every response is cached under
<raw_dir>/<endpoint>/<params>.json and served from there unless the caller
passes a max_age the cached copy is older than.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

Fetch = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


class BudgetExceeded(RuntimeError):
    """Raised before a network call that would exceed the per-run call budget."""


class MissingApiKey(RuntimeError):
    """Raised when a request is not cached and no API key is configured."""


@dataclass(frozen=True)
class CacheEntry:
    endpoint: str
    key: str
    fetched_at: datetime
    data: list[dict[str, Any]]
    from_cache: bool


def cache_key(params: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(params.items()) if v is not None]
    return re.sub(r"[^A-Za-z0-9=._-]", "-", "_".join(parts)) or "all"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CachedClient:
    def __init__(
        self,
        raw_dir: Path,
        fetch: Fetch,
        *,
        max_calls: int = 300,
        min_interval_s: float = 1.0,
        now: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.raw_dir = Path(raw_dir)
        self.max_calls = max_calls
        self.min_interval_s = min_interval_s
        self.api_calls = 0
        self._fetch = fetch
        self._now = now
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_call: float | None = None

    def path_for(self, endpoint: str, params: dict[str, Any]) -> Path:
        return self.raw_dir / endpoint / f"{cache_key(params)}.json"

    def get(self, endpoint: str, params: dict[str, Any], *, max_age: timedelta | None = None) -> CacheEntry:
        path = self.path_for(endpoint, params)
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if max_age is None or self._now() - fetched_at < max_age:
                return CacheEntry(endpoint, path.stem, fetched_at, cached["data"], True)
        data = self._fetch_limited(endpoint, params)
        fetched_at = self._now()
        payload = {"endpoint": endpoint, "params": params, "fetched_at": fetched_at.isoformat(), "data": data}
        self._write_atomic(path, payload)
        return CacheEntry(endpoint, path.stem, fetched_at, data, False)

    def _fetch_limited(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self.api_calls >= self.max_calls:
            raise BudgetExceeded(
                f"Per-run API budget of {self.max_calls} calls reached before {endpoint} {params}. "
                "Cached data is kept; re-run (or raise --max-calls) to continue."
            )
        if self._last_call is not None:
            wait = self.min_interval_s - (self._monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self.api_calls += 1  # counted before the request: a failed call still uses quota
        self._last_call = self._monotonic()
        log.info("CFBD %s %s", endpoint, params)
        return self._fetch(endpoint, params)

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_client.py`
Expected: `8 passed`

---

### Task 3: cfbd adapter

**Files:**
- Create: `src/psu/cfbd_api.py`
- Test: `tests/test_cfbd_api.py`

**Interfaces:**
- Consumes: `psu.client.Fetch`
- Produces:
  - `ENDPOINTS: dict[str, tuple[str, str]]`: endpoint name → (cfbd API class name, method name). Keys: `calendar, games, plays, drives, team_game_stats, player_game_stats, advanced_season, ratings_sp, talent, recruiting, lines`
  - `make_cfbd_fetch(api_key: str) -> Fetch`

**Why raw JSON:** cfbd 5.x models are strict pydantic v1 classes. A field that older seasons return as null can make deserialization raise. So the adapter calls `<method>_with_http_info(..., _preload_content=False)` and parses `ApiResponse.raw_data` itself (verified in `cfbd/api_client.py`: with `_preload_content=False`, `raw_data` is the response body bytes).

- [ ] **Step 1: Write the failing tests**

`tests/test_cfbd_api.py`:
```python
from types import SimpleNamespace
from unittest.mock import patch

import cfbd

from psu.cfbd_api import ENDPOINTS, make_cfbd_fetch


def test_every_endpoint_maps_to_a_real_cfbd_method():
    for endpoint, (cls_name, method) in ENDPOINTS.items():
        assert hasattr(getattr(cfbd, cls_name), f"{method}_with_http_info"), endpoint


def test_fetch_returns_raw_json_and_converts_enums():
    fake = SimpleNamespace(raw_data=b'[{"id": "1", "playType": "Rush"}]')
    with patch.object(cfbd.PlaysApi, "get_plays_with_http_info", return_value=fake) as method:
        fetch = make_cfbd_fetch("test-key")
        data = fetch(
            "plays",
            {"year": 2024, "week": 1, "season_type": "regular", "classification": "fbs", "team": None},
        )
    assert data == [{"id": "1", "playType": "Rush"}]
    kwargs = method.call_args.kwargs
    assert kwargs["year"] == 2024 and kwargs["week"] == 1
    assert kwargs["season_type"] == cfbd.SeasonType("regular")
    assert kwargs["classification"] == cfbd.DivisionClassification("fbs")
    assert kwargs["_preload_content"] is False
    assert "team" not in kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_cfbd_api.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.cfbd_api'`

- [ ] **Step 3: Implement `src/psu/cfbd_api.py`**

```python
"""Map endpoint names to the official cfbd package and return raw JSON records.

Raw JSON (not cfbd's pydantic models) because strict model validation can
reject null fields that older seasons contain.
"""
from __future__ import annotations

import json
from typing import Any

import cfbd

from psu.client import Fetch

ENDPOINTS: dict[str, tuple[str, str]] = {
    "calendar": ("GamesApi", "get_calendar"),
    "games": ("GamesApi", "get_games"),
    "plays": ("PlaysApi", "get_plays"),
    "drives": ("DrivesApi", "get_drives"),
    "team_game_stats": ("GamesApi", "get_game_team_stats"),
    "player_game_stats": ("GamesApi", "get_game_player_stats"),
    "advanced_season": ("StatsApi", "get_advanced_season_stats"),
    "ratings_sp": ("RatingsApi", "get_sp"),
    "talent": ("TeamsApi", "get_talent"),
    "recruiting": ("RecruitingApi", "get_team_recruiting_rankings"),
    "lines": ("BettingApi", "get_lines"),
}

_ENUM_PARAMS = {"season_type": cfbd.SeasonType, "classification": cfbd.DivisionClassification}


def make_cfbd_fetch(api_key: str) -> Fetch:
    api_client = cfbd.ApiClient(cfbd.Configuration(access_token=api_key))

    def fetch(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        cls_name, method_name = ENDPOINTS[endpoint]
        api = getattr(cfbd, cls_name)(api_client)
        kwargs = {
            k: _ENUM_PARAMS[k](v) if k in _ENUM_PARAMS else v for k, v in params.items() if v is not None
        }
        response = getattr(api, f"{method_name}_with_http_info")(**kwargs, _preload_content=False)
        return json.loads(response.raw_data)

    return fetch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_cfbd_api.py`
Expected: `2 passed`

---

### Task 4: DuckDB schemas and idempotent upsert

**Files:**
- Create: `src/psu/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass(frozen=True) TableSpec(name: str, key: tuple[str, ...], columns: dict[str, str], extra_type: str | None = None)`. With `extra_type=None`, undeclared columns are dropped. Otherwise they're added to the table with that type.
  - `SPECS: dict[str, TableSpec]`, keyed by the same names as the tables and `FLATTENERS`
  - `connect(path: Path | str) -> duckdb.DuckDBPyConnection`: creates parent dir and the `_loads` log table
  - `upsert(con, spec: TableSpec, df: pd.DataFrame) -> int`: returns rows written. Replaces existing rows with the same key.
  - `already_loaded(con, endpoint: str, cache_key: str, fetched_at: str) -> bool`, `record_load(con, endpoint, cache_key, fetched_at: str, rows: int) -> None`
  - `row_counts(con) -> dict[str, int]`: every table in `SPECS`, 0 if not created yet

**Design notes:**
- **Casting:** values are cast in SQL with `TRY_CAST`, so a bad value becomes NULL instead of aborting a season. Rows whose key casts to NULL are dropped with a warning.
- **Dates:** start dates are stored as `TIMESTAMP` holding UTC (CFBD sends `...Z`). This deliberately avoids `TIMESTAMPTZ`, which needs `pytz` to fetch into Python.
- **Load log:** `_loads.fetched_at` is an ISO string for the same reason.

- [ ] **Step 1: Write the failing tests**

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_db.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.db'`

- [ ] **Step 3: Implement `src/psu/db.py`**

```python
"""DuckDB storage: declared table schemas and an idempotent upsert."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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


def connect(path: Path | str) -> duckdb.DuckDBPyConnection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE IF NOT EXISTS _loads ("
        "endpoint VARCHAR, cache_key VARCHAR, fetched_at VARCHAR, rows INTEGER, "
        "PRIMARY KEY (endpoint, cache_key))"
    )
    return con


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    result = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]
    ).fetchall()
    return {r[0] for r in result}


def _ensure_table(con: duckdb.DuckDBPyConnection, spec: TableSpec, extras: list[str]) -> None:
    cols = ", ".join(f"{_q(c)} {t}" for c, t in spec.columns.items())
    con.execute(f"CREATE TABLE IF NOT EXISTS {_q(spec.name)} ({cols})")
    existing = _table_columns(con, spec.name)
    for col in extras:
        if col not in existing:
            con.execute(f"ALTER TABLE {_q(spec.name)} ADD COLUMN {_q(col)} {spec.extra_type}")


def upsert(con: duckdb.DuckDBPyConnection, spec: TableSpec, df: pd.DataFrame) -> int:
    """Insert df into spec's table, replacing rows with the same key. Returns rows written."""
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
        name: con.execute(f"SELECT count(*) FROM {_q(name)}").fetchone()[0] if name in existing else 0
        for name in SPECS
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_db.py`
Expected: `10 passed` (a pandas NaN key reaches DuckDB as NULL and is filtered out; verified on DuckDB 1.5.5)

---

### Task 5: Flatteners (raw JSON → table-shaped DataFrames)

**Files:**
- Create: `src/psu/flatten.py`
- Test: `tests/test_flatten.py`

**Interfaces:**
- Consumes: `psu.db.SPECS` and `psu.cfbd_api.ENDPOINTS` (consistency test only)
- Produces:
  - `snake(name: str) -> str`
  - `Flattener = Callable[[list[dict], dict], pd.DataFrame]`, taking `(raw records, request params)`
  - `flatten_games`, `flatten_plays`, `flatten_drives`, `flatten_team_game_stats`, `flatten_player_game_stats`, `flatten_lines`, `flatten_wide`
  - `FLATTENERS: dict[str, Flattener]`, with the same keys as `SPECS`

**Raw shapes** (camelCase, verified against cfbd 5.29 models):
- `games`, `plays`, `drives`: flat records. The only nesting is `clock` / `startTime` / `endTime` / `elapsed` = `{minutes, seconds}`.
- `team_game_stats`: `[{id, teams: [{teamId, team, conference, homeAway, points, stats: [{category, stat}]}]}]`
- `player_game_stats`: `[{id, teams: [{team, conference, homeAway, points, categories: [{name, types: [{name, athletes: [{id, name, stat}]}]}]}]}]`
- `lines`: `[{id, season, seasonType, week, startDate, homeTeam..., lines: [{provider, spread, formattedSpread, spreadOpen, overUnder, overUnderOpen, homeMoneyline, awayMoneyline}]}]`
- `advanced_season` and `ratings_sp`: nested dicts of numbers (e.g. `offense.passingPlays.successRate`). `talent` and `recruiting`: flat.

Plays, drives, and the game-stat endpoints don't carry a season (and some lack week), so those flatteners copy `year`/`week`/`season_type` from the request params.

- [ ] **Step 1: Write the failing tests**

`tests/test_flatten.py`:
```python
import json

import pandas as pd

from psu.cfbd_api import ENDPOINTS
from psu.db import SPECS
from psu.flatten import (
    FLATTENERS,
    flatten_drives,
    flatten_games,
    flatten_lines,
    flatten_player_game_stats,
    flatten_plays,
    flatten_team_game_stats,
    flatten_wide,
    snake,
)

WEEK = {"year": 2024, "week": 3, "season_type": "regular", "classification": "fbs"}


def test_registries_agree():
    assert set(FLATTENERS) == set(SPECS) == set(ENDPOINTS) - {"calendar"}


def test_snake():
    assert snake("yardsToGoal") == "yards_to_goal"
    assert snake("totalPPA") == "total_ppa"
    assert snake("startTimeTBD") == "start_time_tbd"
    assert snake("ppa") == "ppa"


def test_games_keeps_line_scores_as_json():
    df = flatten_games(
        [{"id": 1, "season": 2024, "homeTeam": "Penn State", "homeLineScores": [7, 14, 0, 3], "awayLineScores": None}],
        {"year": 2024},
    )
    row = df.iloc[0]
    assert row["home_team"] == "Penn State"
    assert json.loads(row["home_line_scores"]) == [7, 14, 0, 3]
    assert pd.isna(row["away_line_scores"])


def test_plays_flatten_clock_and_add_request_context():
    df = flatten_plays(
        [{"id": "401", "gameId": 9, "clock": {"minutes": 12, "seconds": 5}, "yardsToGoal": 75, "ppa": None}], WEEK
    )
    row = df.iloc[0].to_dict()
    assert (row["clock_minutes"], row["clock_seconds"]) == (12, 5)
    assert row["yards_to_goal"] == 75
    assert (row["season"], row["week"], row["season_type"]) == (2024, 3, "regular")


def test_drives_add_season_only():
    df = flatten_drives(
        [{"id": "d1", "gameId": 9, "startTime": {"minutes": 15, "seconds": 0}}], {"year": 2016, "season_type": "both"}
    )
    assert df.iloc[0]["season"] == 2016
    assert df.iloc[0]["start_time_minutes"] == 15
    assert "season_type" not in df.columns


def test_team_game_stats_long_format():
    data = [
        {
            "id": 9,
            "teams": [
                {
                    "teamId": 213,
                    "team": "Penn State",
                    "conference": "Big Ten",
                    "homeAway": "home",
                    "points": 34,
                    "stats": [{"category": "totalYards", "stat": "450"}, {"category": "thirdDownEff", "stat": "6-13"}],
                }
            ],
        }
    ]
    df = flatten_team_game_stats(data, WEEK)
    assert list(df["category"]) == ["totalYards", "thirdDownEff"]
    assert set(df["game_id"]) == {9} and set(df["team"]) == {"Penn State"}
    assert set(df["team_id"]) == {213} and set(df["week"]) == {3}


def test_player_game_stats_long_format():
    data = [
        {
            "id": 9,
            "teams": [
                {
                    "team": "Penn State",
                    "conference": "Big Ten",
                    "homeAway": "home",
                    "points": 34,
                    "categories": [
                        {"name": "passing", "types": [{"name": "YDS", "athletes": [{"id": "111", "name": "QB One", "stat": "250"}]}]}
                    ],
                }
            ],
        }
    ]
    row = flatten_player_game_stats(data, WEEK).iloc[0].to_dict()
    assert row["game_id"] == 9 and row["season"] == 2024
    assert (row["category"], row["stat_type"]) == ("passing", "YDS")
    assert (row["athlete_id"], row["athlete_name"], row["stat"]) == ("111", "QB One", "250")


def test_lines_one_row_per_provider():
    data = [
        {
            "id": 9,
            "season": 2024,
            "seasonType": "regular",
            "week": 3,
            "homeTeam": "Penn State",
            "awayTeam": "Bowling Green",
            "lines": [
                {"provider": "Bovada", "spread": -34.5, "overUnder": 55.5},
                {"provider": "ESPN Bet", "spread": -35.0, "overUnder": None},
            ],
        }
    ]
    df = flatten_lines(data, {"year": 2024, "season_type": "both"})
    assert list(df["provider"]) == ["Bovada", "ESPN Bet"]
    assert list(df["spread"]) == [-34.5, -35.0]
    assert set(df["game_id"]) == {9} and set(df["home_team"]) == {"Penn State"}
    assert "lines" not in df.columns


def test_wide_tables_flatten_nested_sections():
    df = flatten_wide(
        [{"season": 2024, "team": "Penn State", "offense": {"ppa": 0.3, "passingPlays": {"successRate": 0.5}}}],
        {"year": 2024},
    )
    assert df.iloc[0]["offense_passing_plays_success_rate"] == 0.5
    assert df.iloc[0]["offense_ppa"] == 0.3


def test_missing_nested_lists_produce_no_rows():
    ctx = {"year": 2014, "week": 1, "season_type": "regular"}
    assert flatten_team_game_stats([{"id": 9, "teams": None}], ctx).empty
    assert flatten_player_game_stats([{"id": 9}], ctx).empty
    assert flatten_lines([{"id": 9, "lines": []}], {"year": 2014}).empty
    assert flatten_plays([], ctx).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_flatten.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.flatten'`

- [ ] **Step 3: Implement `src/psu/flatten.py`**

```python
"""Turn raw CFBD JSON (camelCase, nested) into flat snake_case DataFrames, one shape per table."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import pandas as pd

Records = list[dict[str, Any]]
Flattener = Callable[[Records, dict[str, Any]], pd.DataFrame]


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _normalize(records: Records) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records, sep=".")
    df.columns = ["_".join(snake(part) for part in col.split(".")) for col in df.columns]
    return df


def _add_context(df: pd.DataFrame, params: dict[str, Any], *, with_week: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["season"] = params["year"]
    if with_week:
        df["week"] = params["week"]
        df["season_type"] = params["season_type"]
    return df


def flatten_games(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    df = _normalize(data)
    for col in ("home_line_scores", "away_line_scores"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, list) else None)
    return df


def flatten_plays(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    return _add_context(_normalize(data), params)


def flatten_drives(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    return _add_context(_normalize(data), params, with_week=False)


def flatten_team_game_stats(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "game_id": game.get("id"),
            "team_id": team.get("teamId"),
            "team": team.get("team"),
            "conference": team.get("conference"),
            "home_away": team.get("homeAway"),
            "points": team.get("points"),
            "category": stat.get("category"),
            "stat": stat.get("stat"),
        }
        for game in data
        for team in game.get("teams") or []
        for stat in team.get("stats") or []
    ]
    return _add_context(pd.DataFrame(rows), params)


def flatten_player_game_stats(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "game_id": game.get("id"),
            "team": team.get("team"),
            "conference": team.get("conference"),
            "home_away": team.get("homeAway"),
            "category": category.get("name"),
            "stat_type": stat_type.get("name"),
            "athlete_id": athlete.get("id"),
            "athlete_name": athlete.get("name"),
            "stat": athlete.get("stat"),
        }
        for game in data
        for team in game.get("teams") or []
        for category in team.get("categories") or []
        for stat_type in category.get("types") or []
        for athlete in stat_type.get("athletes") or []
    ]
    return _add_context(pd.DataFrame(rows), params)


def flatten_lines(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for game in data:
        base = {snake(k): v for k, v in game.items() if k not in ("id", "lines")}
        base["game_id"] = game.get("id")
        for line in game.get("lines") or []:
            rows.append({**base, **{snake(k): v for k, v in line.items()}})
    return pd.DataFrame(rows)


def flatten_wide(data: Records, params: dict[str, Any]) -> pd.DataFrame:
    return _normalize(data)


FLATTENERS: dict[str, Flattener] = {
    "games": flatten_games,
    "plays": flatten_plays,
    "drives": flatten_drives,
    "team_game_stats": flatten_team_game_stats,
    "player_game_stats": flatten_player_game_stats,
    "advanced_season": flatten_wide,
    "ratings_sp": flatten_wide,
    "talent": flatten_wide,
    "recruiting": flatten_wide,
    "lines": flatten_lines,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_flatten.py`
Expected: `10 passed`

---

### Task 6: Ingest orchestration

**Files:**
- Create: `src/psu/ingest.py`, `tests/conftest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `CachedClient.get`, `CacheEntry`, `db.connect/upsert/already_loaded/record_load/row_counts/SPECS`, `FLATTENERS`
- Produces:
  - `@dataclass(frozen=True) Request(endpoint: str, params: dict, max_age: timedelta | None)`
  - `season_requests(season: int) -> list[tuple[str, dict]]`: the 7 season-level requests
  - `weekly_requests(season, calendar: list[dict], *, now, current, final_after, refresh_after) -> list[Request]`
  - `@dataclass(frozen=True) IngestResult(api_calls: int, row_counts: dict[str, int])`
  - `ingest(client, con, seasons: list[int], *, current: int, now: datetime | None = None, final_after=timedelta(days=3), refresh_after=timedelta(hours=24)) -> IngestResult`
  - `tests/conftest.py`: fixture `fake_cfbd` → `FakeCFBD` (a `Fetch` that records `.calls: list[tuple[str, dict]]`), with calendars for 2024 (past) and 2026 (current)

**Request plan per season** (about 59 calls for a full past season):
- `calendar {year}`: decides which weeks exist
- Season-level: `games`, `drives` (`season_type="both"`, `classification="fbs"`), `lines` (`season_type="both"`), `advanced_season` (`classification="fbs"`), `ratings_sp`, `talent`, `recruiting`
- Per calendar week, for season types `regular` and `postseason` only: `plays`, `team_game_stats`, `player_game_stats` with `{year, week, season_type, classification: "fbs"}`

**Freshness policy:**
- Past seasons: `max_age=None`. Anything cached is final.
- Current season:
  - Season-level requests and the calendar use `max_age=refresh_after`.
  - A week that hasn't started (`startDate > now`) isn't requested at all.
  - A week whose `endDate` is more than `final_after` ago is final.
  - Any other week uses `refresh_after`.

**Load log:** a cache entry is loaded into DuckDB only if `(endpoint, cache_key, fetched_at)` isn't already in `_loads`. That makes re-runs cheap, and a deleted DB rebuilds from `data/raw/` with zero API calls.

- [ ] **Step 1: Write the shared fake**

`tests/conftest.py`:
```python
"""Shared test doubles. FakeCFBD fabricates small, consistent CFBD payloads from request params."""
import pytest


def _week(season, n, start, end, season_type="regular"):
    return {"season": season, "week": n, "seasonType": season_type, "startDate": start, "endDate": end}


CALENDARS = {
    2024: [
        _week(2024, 1, "2024-08-24T07:00:00.000Z", "2024-09-02T06:59:59.000Z"),
        _week(2024, 2, "2024-09-02T07:00:00.000Z", "2024-09-09T06:59:59.000Z"),
        _week(2024, 1, "2024-12-14T08:00:00.000Z", "2025-01-21T07:59:59.000Z", "postseason"),
        _week(2024, 1, "2025-01-25T08:00:00.000Z", "2025-02-01T07:59:59.000Z", "allstar"),
    ],
    2026: [
        _week(2026, 1, "2026-08-29T07:00:00.000Z", "2026-09-04T06:59:59.000Z"),
        _week(2026, 4, "2026-09-19T07:00:00.000Z", "2026-09-26T06:59:59.000Z"),
        _week(2026, 5, "2026-09-26T07:00:00.000Z", "2026-10-03T06:59:59.000Z"),
        _week(2026, 1, "2026-12-13T08:00:00.000Z", "2027-01-20T07:59:59.000Z", "postseason"),
    ],
}


class FakeCFBD:
    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        y = params["year"]
        week = params.get("week", 0)
        season_type = params.get("season_type", "")
        game_id = y * 1000 + (500 if season_type == "postseason" else 0) + week  # one fake game per week
        if endpoint == "calendar":
            return CALENDARS[y]
        if endpoint == "games":
            return [
                {"id": y * 1000 + i, "season": y, "week": i, "seasonType": "regular", "homeTeam": "Penn State",
                 "awayTeam": "Opponent", "homePoints": 30, "awayPoints": 10}
                for i in (1, 2)
            ]
        if endpoint == "plays":
            return [
                {"id": f"{y}-{season_type}-{week}-{n}", "gameId": game_id, "offense": "Penn State",
                 "defense": "Opponent", "down": 1, "distance": 10, "yardsGained": 5, "ppa": 0.1}
                for n in range(3)
            ]
        if endpoint == "team_game_stats":
            return [{"id": game_id, "teams": [{"teamId": 213, "team": "Penn State", "homeAway": "home",
                                               "points": 30, "stats": [{"category": "totalYards", "stat": "400"}]}]}]
        if endpoint == "player_game_stats":
            return [{"id": game_id, "teams": [{"team": "Penn State", "categories": [{"name": "passing", "types": [
                {"name": "YDS", "athletes": [{"id": "1", "name": "QB One", "stat": "250"}]}]}]}]}]
        if endpoint == "drives":
            return [{"id": f"{y}-d1", "gameId": y * 1000 + 1, "offense": "Penn State", "defense": "Opponent"}]
        if endpoint == "lines":
            return [{"id": y * 1000 + 1, "season": y, "lines": [{"provider": "consensus", "spread": -7.0}]}]
        if endpoint == "advanced_season":
            return [{"season": y, "team": "Penn State", "offense": {"ppa": 0.25}}]
        if endpoint == "ratings_sp":
            return [{"year": y, "team": "Penn State", "rating": 20.0}]
        if endpoint == "talent":
            return [{"year": y, "team": "Penn State", "talent": 850.0}]
        if endpoint == "recruiting":
            return [{"year": y, "team": "Penn State", "rank": 10, "points": 280.0}]
        raise AssertionError(f"unexpected endpoint {endpoint}")


@pytest.fixture
def fake_cfbd():
    return FakeCFBD()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_ingest.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_ingest.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'psu.ingest'`

- [ ] **Step 4: Implement `src/psu/ingest.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_ingest.py`
Expected: `4 passed`

- [ ] **Step 6: Run the whole suite**

Run: `.venv\Scripts\python -m pytest`
Expected: all tests pass (`45 passed`)

---

### Task 7: CLI (`psu ingest`, `psu status`)

**Files:**
- Create: `src/psu/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `config.load_settings`, `config.parse_seasons`, `config.FIRST_SEASON`, `CachedClient`, `BudgetExceeded`, `MissingApiKey`, `ingest`, `db.connect`, `db.row_counts`, `make_cfbd_fetch`
- Produces:
  - `make_fetch(settings: Settings) -> Fetch`. With no API key, it returns a fetch that raises `MissingApiKey`. This means a fully cached run still works without a key.
  - `main(argv: list[str] | None = None) -> int`. Exit codes: 0 = ok, 2 = usage error or missing key, 3 = budget reached.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import pytest

from psu import cli, config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = config.Settings(
        api_key=None,
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "psu.duckdb",
        current_season=2026,
        min_interval_s=0,
    )
    monkeypatch.setattr(config, "load_settings", lambda: s)
    return s


def test_ingest_without_key_or_cache_explains_how_to_fix(settings, capsys):
    assert cli.main(["ingest", "--seasons", "2024"]) == 2
    assert "CFBD_API_KEY" in capsys.readouterr().err


def test_bad_season_spec_is_a_usage_error(settings, capsys):
    assert cli.main(["ingest", "--seasons", "2031"]) == 2
    assert "2022-2026" in capsys.readouterr().err


def test_status_lists_all_tables(settings, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "plays" in out and "ratings_sp" in out


def test_budget_exhaustion_exits_cleanly(settings, capsys, monkeypatch, fake_cfbd):
    monkeypatch.setattr(cli, "make_fetch", lambda s: fake_cfbd)
    assert cli.main(["ingest", "--seasons", "2024", "--max-calls", "3"]) == 3
    assert "budget" in capsys.readouterr().err.lower()
    assert len(fake_cfbd.calls) == 3


def test_ingest_success_reports_calls_and_counts(settings, capsys, monkeypatch, fake_cfbd):
    monkeypatch.setattr(cli, "make_fetch", lambda s: fake_cfbd)
    assert cli.main(["ingest", "--seasons", "2024"]) == 0
    out = capsys.readouterr().out
    assert "API calls this run: 17" in out
    assert "plays" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`
Expected: FAIL with `ImportError: cannot import name 'cli' from 'psu'`

- [ ] **Step 3: Implement `src/psu/cli.py`**

```python
"""Command-line entry point: `psu ingest` and `psu status`."""
from __future__ import annotations

import argparse
import logging
import sys

from psu import config, db
from psu.client import BudgetExceeded, CachedClient, Fetch, MissingApiKey
from psu.ingest import ingest


def make_fetch(settings: config.Settings) -> Fetch:
    if not settings.api_key:

        def missing_key(endpoint, params):
            raise MissingApiKey(
                f"CFBD_API_KEY is not set and {endpoint} {params} is not cached. "
                "Copy .env.example to .env and add your free key from https://collegefootballdata.com/key"
            )

        return missing_key
    from psu.cfbd_api import make_cfbd_fetch  # imported lazily: cfbd is slow to import

    return make_cfbd_fetch(settings.api_key)


def _print_counts(counts: dict[str, int]) -> None:
    for table, n in counts.items():
        print(f"{table:<18} {n:>10,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="psu", description="Penn State football analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Pull CFBD data into DuckDB (cached; safe to re-run)")
    ing.add_argument("--seasons", help='e.g. "2024", "2022-2026" or "2022,2024-2025" (default: every season)')
    ing.add_argument("--max-calls", type=int, help="Stop before making more than this many API calls")
    sub.add_parser("status", help="Show row counts per table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = config.load_settings()

    if args.command == "status":
        con = db.connect(settings.db_path)
        try:
            _print_counts(db.row_counts(con))
        finally:
            con.close()
        return 0

    try:
        seasons = config.parse_seasons(
            args.seasons or f"{config.FIRST_SEASON}-{settings.current_season}", last=settings.current_season
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    client = CachedClient(
        settings.raw_dir,
        make_fetch(settings),
        max_calls=settings.max_calls if args.max_calls is None else args.max_calls,
        min_interval_s=settings.min_interval_s,
    )
    con = db.connect(settings.db_path)
    try:
        result = ingest(
            client,
            con,
            seasons,
            current=settings.current_season,
            final_after=settings.final_after,
            refresh_after=settings.refresh_after,
        )
    except MissingApiKey as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BudgetExceeded as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 3
    finally:
        con.close()
    _print_counts(result.row_counts)
    print(f"API calls this run: {result.api_calls}")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`
Expected: `5 passed`

- [ ] **Step 5: Run the whole suite and check the console script**

Run: `.venv\Scripts\python -m pytest` then `.venv\Scripts\psu --help`
Expected: `50 passed`; help lists `ingest` and `status`.

---

### Task 8: Live smoke test on one season, README, phase commit

**This is the only task that makes live API calls. The orchestrator (main session) runs it, not a subagent, because it needs the user's key and uses real quota.**

**Files:**
- Create: `README.md`
- Possibly modify: `src/psu/ingest.py` (only if Step 3's fallback is needed)

**Interfaces:**
- Consumes: everything above
- Produces: a populated `data/psu.duckdb` for 2024, the README Phase 1 section, and one local commit

- [ ] **Step 1: Confirm the key is configured (the user adds it to `.env` themselves)**

Run: `.venv\Scripts\python -c "from psu.config import load_settings; print(bool(load_settings().api_key))"`
Expected: `True`. If `False`, stop and ask the user to copy `.env.example` to `.env` and paste their key.

- [ ] **Step 2: Ingest one season with a tight budget**

Run: `.venv\Scripts\psu ingest --seasons 2024 --max-calls 80`
Expected: about 60 `CFBD ...` log lines, a row-count table where every table is > 0, and `API calls this run: ` between 55 and 65. Rough sizes for 2024: `games` ~900, `plays` ~150k, `drives` ~25k, `talent` / `recruiting` / `ratings_sp` / `advanced_season` ~130–250 rows each.

- [ ] **Step 3: Only if a season-level request fails with HTTP 400 on `season_type="both"`**

Replace that endpoint's single request in `season_requests` with one request per game season type. For example, for `lines`:
```python
        *[("lines", {"year": season, "season_type": st}) for st in GAME_SEASON_TYPES],
```
Do the same for `games` or `drives` if they're the ones that failed. Update `SEASON_LEVEL` in `tests/test_ingest.py` and the `17` in `tests/test_cli.py` to match the new request count. Run `.venv\Scripts\python -m pytest`, then repeat Step 2. The requests that succeeded are cached and won't be re-fetched.

- [ ] **Step 4: Verify the re-run is free**

Run: `.venv\Scripts\psu ingest --seasons 2024`
Expected: `API calls this run: 0`, with the same row counts as Step 2.

- [ ] **Step 5: Sanity-check Penn State data**

Run:
```powershell
@'
import duckdb
con = duckdb.connect("data/psu.duckdb", read_only=True)
print(con.execute("""
    SELECT count(*) AS plays, round(avg(ppa), 3) AS ppa_per_play
    FROM plays WHERE season = 2024 AND offense = 'Penn State' AND ppa IS NOT NULL
""").fetchall())
print(con.execute("""
    SELECT count(*) FROM games
    WHERE season = 2024 AND 'Penn State' IN (home_team, away_team) AND completed
""").fetchall())
print(con.execute("SELECT year, rating, ranking FROM ratings_sp WHERE team = 'Penn State' AND year = 2024").fetchall())
'@ | .venv\Scripts\python -
```
Expected: roughly 900–1,300 PSU offensive plays with a positive average PPA; 16 completed games (2024 went 13-3, including the CFP); one SP+ row. Tell the user about any big mismatch before continuing.

- [ ] **Step 6: Write `README.md`**

````markdown
# PSU Analytics

Penn State football analytics built on the [CollegeFootballData](https://collegefootballdata.com) API:
a cached data pipeline into DuckDB, efficiency metrics, predictive models, and a Streamlit dashboard.
The full build plan is in `docs/superpowers/specs/psu-analytics-build-plan.md`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env   # then paste your free key from https://collegefootballdata.com/key
.venv\Scripts\python -m pytest
```

## Phase 1: Data pipeline

```powershell
.venv\Scripts\psu ingest --seasons 2024        # one season (~60 API calls)
.venv\Scripts\psu ingest --seasons 2022-2026   # everything (~295 calls the first time)
.venv\Scripts\psu status                       # row counts per table
```

- Every API response is cached in `data/raw/<endpoint>/<params>.json`. Past seasons are never
  re-requested. For the current season, data less than 24 hours old is reused, weeks that haven't
  started are skipped, and a week is treated as final 3 days after it ends.
- The CFBD free tier has a monthly call limit. Each run stops before it exceeds `--max-calls`
  (default 300, or `PSU_MAX_CALLS` in `.env`). Whatever was fetched is kept, so re-running continues
  where it stopped. A full 2022–2026 pull (~295 calls) fits in one default run. To pull more
  history, lower `FIRST_SEASON` in `src/psu/config.py` (about 60 calls per extra season).
- Data lands in `data/psu.duckdb`: `games`, `plays`, `drives`, `team_game_stats`,
  `player_game_stats`, `advanced_season`, `ratings_sp`, `talent`, `recruiting`, `lines`. Game-level
  stats are stored long (one row per team/stat or player/stat). Start dates are UTC.
- Deleting `data/psu.duckdb` is safe: the next `psu ingest` rebuilds it from the cache without
  calling the API.
````

- [ ] **Step 7: Full suite, commit, push, open the PR**

Run: `.venv\Scripts\python -m pytest` (expected: all pass), then:
```powershell
git add README.md
git status --short   # confirm no .env, data/raw, or .duckdb files are staged
git commit -m "docs: add setup and Phase 1 pipeline guide to README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/phase1-data-pipeline
```
Open a PR `feat/phase1-data-pipeline` → `main` titled "Phase 1: cached CFBD ingestion into DuckDB". The body covers a summary, the Step 2 row counts, the Step 5 sanity numbers, API calls used, and a test plan. Then **stop for the user's Phase 1 review**: don't merge, and ask before running the full 2022–2026 pull (about 235 more calls).
