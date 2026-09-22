import pandas as pd
import pytest

from psu.metrics import efficiency, havoc_rate, red_zone, turnover_margin

BASE = dict(season=2024, offense="A", defense="B", ppa=0.0, success=False, explosive=False,
            turnover=False, play_class="rush", down=1, garbage=False, yards_to_goal=50, drive_id="d1")


def enriched(*rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


def box(rows):
    return pd.DataFrame([{"game_id": g, "season": 2024, "team": t, "category": c, "stat": s} for g, t, c, s in rows])


def test_efficiency_offense_basic():
    df = enriched(
        {"ppa": 1.0, "success": True, "play_class": "pass", "explosive": True},
        {"ppa": -0.5},
        {"ppa": 0.5, "success": True, "down": 3},
        {"ppa": 9.0, "success": True, "garbage": True},
    )
    row = efficiency(df, "offense").iloc[0]
    assert (row["team"], row["season"], row["plays"]) == ("A", 2024, 3)
    assert row["epa_per_play"] == pytest.approx(1 / 3)
    assert row["rush_epa"] == pytest.approx(0.0)
    assert row["pass_epa"] == pytest.approx(1.0)
    assert row["success_rate"] == pytest.approx(2 / 3)
    assert row["rush_success_rate"] == pytest.approx(0.5)
    assert row["pass_success_rate"] == pytest.approx(1.0)
    assert row["explosiveness"] == pytest.approx(0.75)
    assert row["explosive_rate"] == pytest.approx(1 / 3)
    assert row["third_down_rate"] == pytest.approx(1.0)


def test_efficiency_can_include_garbage_time():
    df = enriched({"ppa": 1.0}, {"ppa": 9.0, "garbage": True})
    assert efficiency(df, "offense", exclude_garbage=False).iloc[0]["plays"] == 2


def test_efficiency_defense_side_and_splits():
    df = enriched(
        {"offense": "A", "defense": "B", "down": 1, "ppa": 0.2},
        {"offense": "C", "defense": "B", "down": 2, "ppa": 0.4},
        {"offense": "B", "defense": "A", "ppa": -1.0},
    )
    out = efficiency(df, "defense", by=("season", "down"))
    b = out[out["team"] == "B"].sort_values("down")
    assert list(b["down"]) == [1, 2]
    assert list(b["epa_per_play"]) == pytest.approx([0.2, 0.4])


def test_missing_play_class_gives_nan_not_error():
    row = efficiency(enriched({"play_class": "pass", "ppa": 0.3}), "offense").iloc[0]
    assert pd.isna(row["rush_epa"]) and pd.isna(row["third_down_rate"])
    assert row["pass_epa"] == pytest.approx(0.3)


def test_efficiency_rejects_unknown_side():
    with pytest.raises(ValueError):
        efficiency(enriched({}), "special_teams")


def test_havoc_rate_counts_box_score_events_and_forced_fumbles():
    stats = box([
        (1, "A", "tacklesForLoss", "5"), (1, "A", "passesDeflected", "3"),
        (1, "A", "passesIntercepted", "1"), (1, "A", "totalFumbles", "0"),
        (1, "B", "tacklesForLoss", "2"), (1, "B", "totalFumbles", "2"),  # B has no passesDeflected row
    ])
    plays = enriched(*[{"offense": "B", "defense": "A"}] * 40, *[{"offense": "A", "defense": "B"}] * 50)
    out = havoc_rate(stats, plays).set_index("team")
    assert out.loc["A", "havoc_events"] == 11  # 5 TFL + 3 PD + 1 INT + 2 opponent fumbles
    assert out.loc["A", "havoc_rate"] == pytest.approx(11 / 40)
    assert out.loc["B", "havoc_events"] == 2
    assert out.loc["B", "havoc_rate"] == pytest.approx(2 / 50)


def test_turnover_margin():
    stats = box([
        (1, "A", "turnovers", "1"), (1, "B", "turnovers", "3"),
        (2, "A", "turnovers", "2"), (2, "C", "totalYards", "300"),  # C has no turnovers row: counts as 0
    ])
    out = turnover_margin(stats).set_index("team")
    assert out.loc["A", ["games", "giveaways", "takeaways", "margin"]].tolist() == [2, 3, 3, 0]
    assert out.loc["B", "margin"] == -2
    assert out.loc["C", "margin"] == 2


def test_red_zone_trips_touchdowns_and_points():
    plays = enriched(
        {"drive_id": "d1", "yards_to_goal": 15}, {"drive_id": "d1", "yards_to_goal": 3},
        {"drive_id": "d2", "yards_to_goal": 18}, {"drive_id": "d3", "yards_to_goal": 40},
        {"drive_id": "d4", "yards_to_goal": 5},
    )
    drives = pd.DataFrame([
        {"id": "d1", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 0,
         "end_offense_score": 7, "drive_result": "TD"},
        {"id": "d2", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 7,
         "end_offense_score": 10, "drive_result": "FG"},
        {"id": "d3", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 10,
         "end_offense_score": 10, "drive_result": "PUNT"},
        {"id": "d4", "season": 2024, "offense": "A", "defense": "B", "start_offense_score": 10,
         "end_offense_score": 11, "drive_result": "TD"},
    ])
    row = red_zone(plays, drives).iloc[0]
    assert (row["team"], row["trips"], row["touchdowns"]) == ("A", 3, 2)
    assert row["td_rate"] == pytest.approx(2 / 3)
    assert row["points_per_trip"] == pytest.approx(16 / 3)
    assert red_zone(plays, drives, "defense").iloc[0]["team"] == "B"
