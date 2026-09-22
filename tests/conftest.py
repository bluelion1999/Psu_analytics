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
