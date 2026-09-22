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
