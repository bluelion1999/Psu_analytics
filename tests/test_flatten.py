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
                        {
                            "name": "passing",
                            "types": [{"name": "YDS", "athletes": [{"id": "111", "name": "QB One", "stat": "250"}]}],
                        }
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


def test_returning_production_flattens_to_snake_case():
    from psu.flatten import FLATTENERS

    df = FLATTENERS["returning_production"](
        [{"season": 2025, "team": "Penn State", "conference": "Big Ten", "percentPPA": 0.62, "totalPassingPPA": 80.5}],
        {"year": 2025},
    )
    assert df.loc[0, "season"] == 2025 and df.loc[0, "percent_ppa"] == 0.62
    assert df.loc[0, "total_passing_ppa"] == 80.5
