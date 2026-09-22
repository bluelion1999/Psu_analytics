import pandas as pd
import pytest

from psu.transform import Explosive, GarbageTime, enrich_plays, is_garbage, is_success, score_state

BASE = dict(
    game_id=1, drive_id="d1", season=2024, week=1, season_type="regular",
    offense="Penn State", offense_conference="Big Ten", defense="Opp", defense_conference="MAC",
    home="Penn State", away="Opp", period=1, down=1, distance=10, yards_to_goal=75, yards_gained=5,
    play_type="Rush", play_text="RB One run for 5 yds", ppa=0.1, offense_score=0, defense_score=0,
)
GAMES = pd.DataFrame({"id": [1, 2], "neutral_site": [False, True]})
DRIVES = pd.DataFrame([
    {"id": "d1", "offense": "Penn State", "start_offense_score": 0, "start_defense_score": 0},
])


def plays(*overrides):
    return pd.DataFrame([{**BASE, "id": str(i), **o} for i, o in enumerate(overrides)])


@pytest.mark.parametrize(
    "down,distance,gained,expected",
    [(1, 10, 5, True), (1, 10, 4, False), (2, 10, 7, True), (2, 10, 6, False),
     (3, 4, 4, True), (3, 4, 3, False), (4, 1, 1, True), (4, 1, 0, False)],
)
def test_success_thresholds(down, distance, gained, expected):
    df = plays({"down": down, "distance": distance, "yards_gained": gained})
    assert bool(is_success(df).iloc[0]) is expected


def test_touchdowns_always_succeed_and_turnovers_never_do():
    df = plays(
        {"play_type": "Rushing Touchdown", "down": 3, "distance": 10, "yards_gained": 2},
        {"play_type": "Pass Interception Return", "yards_gained": 25},
        {"play_type": "Fumble Recovery (Opponent)", "yards_gained": 8},
    )
    assert list(is_success(df)) == [True, False, False]


@pytest.mark.parametrize(
    "period,margin,expected",
    [(1, 50, False), (2, 38, False), (2, 39, True), (3, 29, True), (3, -29, True),
     (3, 28, False), (4, 23, True), (4, 22, False), (5, 40, False)],
)
def test_garbage_time_default_thresholds(period, margin, expected):
    got = is_garbage(pd.Series([period]), pd.Series([margin]), GarbageTime())
    assert bool(got.iloc[0]) is expected


def test_garbage_time_is_configurable():
    custom = GarbageTime.parse("30,20,10")
    assert custom == GarbageTime(None, 30, 20, 10)
    assert bool(is_garbage(pd.Series([4]), pd.Series([11]), custom).iloc[0])
    assert not bool(is_garbage(pd.Series([4]), pd.Series([60]), GarbageTime.parse("off")).iloc[0])
    with pytest.raises(ValueError):
        GarbageTime.parse("30,20")


def test_score_state_buckets():
    got = score_state(pd.Series([-9, -8, -1, 0, 1, 8, 9]))
    assert list(got) == ["down 9+", "down 1-8", "down 1-8", "tied", "up 1-8", "up 1-8", "up 9+"]


def test_enrich_filters_to_scrimmage_plays_with_ppa():
    df = plays({"play_type": "Rush"}, {"play_type": "Punt"}, {"play_type": "Timeout"}, {"play_type": "Rush", "ppa": None})
    assert list(enrich_plays(df, GAMES, DRIVES)["id"]) == ["0"]


def test_enrich_classifies_rush_pass_and_fumbles():
    df = plays(
        {"play_type": "Rush"},
        {"play_type": "Sack"},
        {"play_type": "Pass Incompletion"},
        {"play_type": "Fumble Recovery (Own)", "play_text": "QB One pass complete to WR Two, fumbled"},
        {"play_type": "Fumble Recovery (Own)", "play_text": "RB One run for 3 yds, fumbled"},
    )
    assert list(enrich_plays(df, GAMES, DRIVES)["play_class"]) == ["rush", "pass", "pass", "pass", "rush"]


def test_explosive_uses_separate_rush_and_pass_thresholds():
    df = plays(
        {"play_type": "Rush", "yards_gained": 12},
        {"play_type": "Rush", "yards_gained": 11},
        {"play_type": "Pass Reception", "yards_gained": 16},
        {"play_type": "Pass Reception", "yards_gained": 15},
    )
    assert list(enrich_plays(df, GAMES, DRIVES)["explosive"]) == [True, False, True, False]
    custom = enrich_plays(df, GAMES, DRIVES, explosive=Explosive(rush_yards=10, pass_yards=20))
    assert list(custom["explosive"]) == [True, True, False, False]


def test_context_columns():
    df = plays(
        {"offense_score": 21, "defense_score": 0, "period": 3, "drive_id": "d2"},
        {"offense": "Opp", "defense": "Penn State", "offense_score": 0, "defense_score": 35, "period": 3},
        {"game_id": 2, "period": 5},
    )
    drives = pd.concat([DRIVES, pd.DataFrame([
        {"id": "d2", "offense": "Penn State", "start_offense_score": 21, "start_defense_score": 0},
    ])], ignore_index=True)
    out = enrich_plays(df, GAMES, drives)
    assert list(out["venue"]) == ["home", "away", "neutral"]
    assert list(out["score_state"]) == ["up 9+", "down 9+", "tied"]
    assert list(out["garbage"]) == [False, True, False]
    assert list(out["quarter"]) == ["3", "3", "OT"]
    assert list(out["margin"]) == [21, -35, 0]


def test_enrich_keeps_alternate_play_type_names():
    df = plays(
        {"play_type": "Pass Completion"},
        {"play_type": "Pass"},
        {"play_type": "Fumble", "play_text": "RB One run for 2 yds, fumbled"},
    )
    out = enrich_plays(df, GAMES, DRIVES)
    assert list(out["id"]) == ["0", "1", "2"]
    assert list(out["play_class"]) == ["pass", "pass", "rush"]


def test_margin_uses_pre_play_score_on_scoring_plays():
    df = plays({"play_type": "Passing Touchdown", "offense_score": 7, "defense_score": 0})
    out = enrich_plays(df, GAMES, DRIVES)
    assert list(out["margin"]) == [0]
    assert list(out["score_state"]) == ["tied"]

    drives = pd.DataFrame([
        {"id": "d3", "offense": "Penn State", "start_offense_score": 22, "start_defense_score": 0},
    ])
    q4 = plays({
        "play_type": "Passing Touchdown", "period": 4, "drive_id": "d3",
        "offense_score": 29, "defense_score": 0,
    })
    out_q4 = enrich_plays(q4, GAMES, drives)
    assert list(out_q4["margin"]) == [22]
    assert list(out_q4["garbage"]) == [False]
