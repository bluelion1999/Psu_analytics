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
