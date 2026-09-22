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


def seed_raw_tables(con):
    """Load a tiny 2024 season into the raw Phase 1 tables: FBS Alpha and Beta, FCS Gamma."""
    import pandas as pd

    from psu.db import SPECS, upsert

    games = pd.DataFrame([
        {"id": 1, "season": 2024, "week": 1, "season_type": "regular", "neutral_site": False,
         "home_team": "Alpha", "home_classification": "fbs", "home_points": 28,
         "away_team": "Beta", "away_classification": "fbs", "away_points": 14},
        {"id": 2, "season": 2024, "week": 2, "season_type": "regular", "neutral_site": False,
         "home_team": "Alpha", "home_classification": "fbs", "home_points": 42,
         "away_team": "Gamma", "away_classification": "fcs", "away_points": 3},
    ])
    plays, drives, box = [], [], []
    n = 0
    for game_id, home, away in ((1, "Alpha", "Beta"), (2, "Alpha", "Gamma")):
        for offense, defense in ((home, away), (away, home)):
            drive_id = f"{game_id}-{offense}"
            drives.append({"id": drive_id, "game_id": game_id, "season": 2024, "offense": offense,
                           "defense": defense, "start_offense_score": 0, "start_defense_score": 0,
                           "end_offense_score": 7 if offense == "Alpha" else 0,
                           "drive_result": "TD" if offense == "Alpha" else "PUNT"})
            for i in range(6):
                n += 1
                plays.append({
                    "id": str(n), "game_id": game_id, "drive_id": drive_id, "season": 2024, "week": game_id,
                    "season_type": "regular", "offense": offense, "defense": defense,
                    "offense_conference": None if offense == "Gamma" else "Big Ten",
                    "defense_conference": None if defense == "Gamma" else "Big Ten",
                    "home": home, "away": away, "period": 1 + i % 4, "down": 1 + i % 3, "distance": 10,
                    "yards_to_goal": 60 - i * 10, "yards_gained": 8 if offense == "Alpha" else 2,
                    "play_type": "Rush" if i % 2 else "Pass Reception", "play_text": "",
                    "ppa": 0.4 if offense == "Alpha" else -0.2, "offense_score": 0, "defense_score": 0,
                })
            plays.append({**plays[-1], "id": f"punt-{n}", "play_type": "Punt", "ppa": None})
            for category, stat in (("turnovers", "1" if offense == "Beta" else "0"), ("tacklesForLoss", "2")):
                box.append({"game_id": game_id, "season": 2024, "week": game_id, "season_type": "regular",
                            "team": offense, "category": category, "stat": stat})
    upsert(con, SPECS["games"], games)
    upsert(con, SPECS["plays"], pd.DataFrame(plays))
    upsert(con, SPECS["drives"], pd.DataFrame(drives))
    upsert(con, SPECS["team_game_stats"], pd.DataFrame(box))
