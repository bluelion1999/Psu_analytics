"""Project-wide settings: team, season range, paths, and API budget knobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from psu.modelconfig import ModelConfig

TEAM = "Penn State"
FIRST_SEASON = 2019  # 2019 on (2020 is the COVID season); lower this to pull more history

# Tuned pipeline defaults, shared by the individual commands and `psu refresh`.
GARBAGE = "38,28,22"  # Q2,Q3,Q4 garbage-time margins
BUILD_ALPHA = 50.0  # ridge shrinkage for opponent adjustment
TRAIN_ALPHA = 20.0  # ridge shrinkage for rolling team ratings
SHRINK_PLAYS = 75  # plays before a season's own data outweighs last season
SIM_N = 10_000  # simulated seasons
SIM_SEED = 0
SIM_TAU = 5.0  # spread (points) of each team's season-long strength draw
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Production model configuration. `psu train` and `psu simulate --backfill` follow this; `psu experiment`
# searches for a better one and, once adopted, this is updated to match.
# Adopted from `psu experiment` (2023-2025 walk-forward, 2,398 games): pregame Elo and explosive-play ratings
# cut MAE 12.75 -> 12.57 and Brier 0.1889 -> 0.1841 over the previous features. See data/reports/experiments.md.
MODEL_CONFIG = ModelConfig(
    features=(*ModelConfig().features, "d_elo", "d_off_expl", "d_def_expl"),
    metrics=("epa", "sr", "expl"),
)


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
