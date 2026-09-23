"""Shared test doubles. FakeCFBD fabricates small, consistent CFBD payloads from request params."""

import pandas as pd
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
                {
                    "id": y * 1000 + i,
                    "season": y,
                    "week": i,
                    "seasonType": "regular",
                    "homeTeam": "Penn State",
                    "awayTeam": "Opponent",
                    "homePoints": 30,
                    "awayPoints": 10,
                }
                for i in (1, 2)
            ]
        if endpoint == "plays":
            return [
                {
                    "id": f"{y}-{season_type}-{week}-{n}",
                    "gameId": game_id,
                    "offense": "Penn State",
                    "defense": "Opponent",
                    "down": 1,
                    "distance": 10,
                    "yardsGained": 5,
                    "ppa": 0.1,
                }
                for n in range(3)
            ]
        if endpoint == "team_game_stats":
            return [
                {
                    "id": game_id,
                    "teams": [
                        {
                            "teamId": 213,
                            "team": "Penn State",
                            "homeAway": "home",
                            "points": 30,
                            "stats": [{"category": "totalYards", "stat": "400"}],
                        }
                    ],
                }
            ]
        if endpoint == "player_game_stats":
            return [
                {
                    "id": game_id,
                    "teams": [
                        {
                            "team": "Penn State",
                            "categories": [
                                {
                                    "name": "passing",
                                    "types": [
                                        {"name": "YDS", "athletes": [{"id": "1", "name": "QB One", "stat": "250"}]}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
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

    games = pd.DataFrame(
        [
            {
                "id": 1,
                "season": 2024,
                "week": 1,
                "season_type": "regular",
                "neutral_site": False,
                "home_team": "Alpha",
                "home_classification": "fbs",
                "home_points": 28,
                "away_team": "Beta",
                "away_classification": "fbs",
                "away_points": 14,
            },
            {
                "id": 2,
                "season": 2024,
                "week": 2,
                "season_type": "regular",
                "neutral_site": False,
                "home_team": "Alpha",
                "home_classification": "fbs",
                "home_points": 42,
                "away_team": "Gamma",
                "away_classification": "fcs",
                "away_points": 3,
            },
        ]
    )
    plays, drives, box = [], [], []
    n = 0
    for game_id, home, away in ((1, "Alpha", "Beta"), (2, "Alpha", "Gamma")):
        for offense, defense in ((home, away), (away, home)):
            drive_id = f"{game_id}-{offense}"
            drives.append(
                {
                    "id": drive_id,
                    "game_id": game_id,
                    "season": 2024,
                    "offense": offense,
                    "defense": defense,
                    "start_offense_score": 0,
                    "start_defense_score": 0,
                    "end_offense_score": 7 if offense == "Alpha" else 0,
                    "drive_result": "TD" if offense == "Alpha" else "PUNT",
                }
            )
            for i in range(6):
                n += 1
                plays.append(
                    {
                        "id": str(n),
                        "game_id": game_id,
                        "drive_id": drive_id,
                        "season": 2024,
                        "week": game_id,
                        "season_type": "regular",
                        "offense": offense,
                        "defense": defense,
                        "offense_conference": None if offense == "Gamma" else "Big Ten",
                        "defense_conference": None if defense == "Gamma" else "Big Ten",
                        "home": home,
                        "away": away,
                        "period": 1 + i % 4,
                        "down": 1 + i % 3,
                        "distance": 10,
                        "yards_to_goal": 60 - i * 10,
                        "yards_gained": 8 if offense == "Alpha" else 2,
                        "play_type": "Rush" if i % 2 else "Pass Reception",
                        "play_text": "",
                        "ppa": 0.4 if offense == "Alpha" else -0.2,
                        "offense_score": 0,
                        "defense_score": 0,
                    }
                )
            plays.append({**plays[-1], "id": f"punt-{n}", "play_type": "Punt", "ppa": None})
            for category, stat in (("turnovers", "1" if offense == "Beta" else "0"), ("tacklesForLoss", "2")):
                box.append(
                    {
                        "game_id": game_id,
                        "season": 2024,
                        "week": game_id,
                        "season_type": "regular",
                        "team": offense,
                        "category": category,
                        "stat": stat,
                    }
                )
    upsert(con, SPECS["games"], games)
    upsert(con, SPECS["plays"], pd.DataFrame(plays))
    upsert(con, SPECS["drives"], pd.DataFrame(drives))
    upsert(con, SPECS["team_game_stats"], pd.DataFrame(box))


def synthetic_features(seed=0, games_per_season=240):
    """Feature rows for 2022-2026 with a known linear signal; half of 2026 is still to be played."""
    import numpy as np
    import pandas as pd

    from psu.features import FEATURES

    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(30)] + ["Penn State"]
    rows, game_id = [], 0
    for season in range(2022, 2027):
        for i in range(games_per_season):
            game_id += 1
            home, away = rng.choice(teams, 2, replace=False)
            x = {c: float(rng.normal()) for c in FEATURES}
            x["home_field"] = float(rng.random() < 0.9)
            signal = 8 * x["d_off_epa"] - 6 * x["d_def_epa"] + 4 * x["d_prior_sp"] + 2.5 * x["home_field"]
            completed = season < 2026 or i < games_per_season // 2
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": 1 + i % 15,
                    "season_type": "regular",
                    "home_team": str(home),
                    "away_team": str(away),
                    "completed": completed,
                    "margin": signal + rng.normal(0, 12) if completed else np.nan,
                    "vegas_margin": signal + rng.normal(0, 3) if i % 10 else np.nan,
                    "start_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (i % 15)),
                    "neutral_site": False,
                    **x,
                }
            )
    return pd.DataFrame(rows)


def synthetic_league():
    """A 2026 four-team 'Big Ten' (A-D), an outsider X ('Other') and an FCS team F.

    Played: A beat B 30-10 (conference) and X beat C 24-14 (non-conference), both on 2026-09-01. Still to play:
    the other five conference games, A-X, and A-F. Predictions cover every remaining game except A-F (FCS).
    Returns (games, predictions) shaped like the `games` table and `game_predictions` rows.
    """
    rows = [
        # id, week, home, away, home_conf, away_conf, completed, home_pts, away_pts, pred_margin, notes
        (1, 1, "A", "B", "Big Ten", "Big Ten", True, 30, 10, None, None),
        (2, 1, "X", "C", "Other", "Big Ten", True, 24, 14, None, None),
        (3, 2, "C", "D", "Big Ten", "Big Ten", False, None, None, 0.0, None),
        (4, 2, "A", "X", "Big Ten", "Other", False, None, None, 20.0, None),
        (5, 3, "B", "C", "Big Ten", "Big Ten", False, None, None, 2.0, None),
        (6, 3, "D", "A", "Big Ten", "Big Ten", False, None, None, -20.0, None),
        (7, 4, "A", "C", "Big Ten", "Big Ten", False, None, None, 20.0, None),
        (8, 4, "B", "D", "Big Ten", "Big Ten", False, None, None, -1.0, None),
        (9, 5, "A", "F", "Big Ten", "FCS", False, None, None, None, None),
    ]
    games = pd.DataFrame(
        [
            {
                "id": gid,
                "season": 2026,
                "week": week,
                "season_type": "regular",
                "start_date": pd.Timestamp(2026, 9, 1) + pd.Timedelta(days=7 * (week - 1)),
                "completed": done,
                "home_team": home,
                "away_team": away,
                "home_conference": home_conf,
                "away_conference": away_conf,
                "home_points": home_pts,
                "away_points": away_pts,
                "notes": notes,
            }
            for gid, week, home, away, home_conf, away_conf, done, home_pts, away_pts, _, notes in rows
        ]
    )
    games[["home_points", "away_points"]] = games[["home_points", "away_points"]].astype("Int64")
    predictions = pd.DataFrame(
        [
            {"game_id": gid, "home_team": home, "away_team": away, "neutral_site": False, "pred_margin": pred}
            for gid, _, home, away, _, _, done, _, _, pred, _ in rows
            if not done and pred is not None
        ]
    )
    return games, predictions


def seed_league_db(con):
    """Load synthetic_league() into a psu.db connection: rows into `games`, plus a `game_predictions` table."""
    from psu.db import SPECS, upsert

    games, predictions = synthetic_league()
    upsert(con, SPECS["games"], games)  # creates the declared `games` table (connect() does not)
    con.register("_preds", predictions.assign(season=2026, split="upcoming"))
    try:
        con.execute("CREATE OR REPLACE TABLE game_predictions AS SELECT * FROM _preds")
    finally:
        con.unregister("_preds")


DASH_CONF = {"Penn State": "Big Ten", "Ohio State": "Big Ten", "Temple": "American Athletic", "Buffalo": "Mid-American"}
DASH_STRENGTH = {"Penn State": 0.3, "Ohio State": 0.25, "Temple": -0.1, "Buffalo": -0.2}
DASH_GAMES = [
    # id, season, week, home, away, home_points, away_points (None = not played yet)
    (101, 2025, 1, "Penn State", "Temple", 34, 10),
    (102, 2025, 1, "Ohio State", "Buffalo", 40, 7),
    (103, 2025, 2, "Ohio State", "Penn State", 24, 27),
    (104, 2025, 2, "Temple", "Buffalo", 20, 17),
    (201, 2026, 1, "Penn State", "Buffalo", 45, 0),
    (202, 2026, 1, "Temple", "Ohio State", 3, 38),
    (203, 2026, 2, "Penn State", "Ohio State", None, None),
    (204, 2026, 2, "Buffalo", "Temple", None, None),
    (301, 2027, 1, "Penn State", "Temple", None, None),
]


def _dash_quarters(points):
    import json

    base = points // 4
    return json.dumps([base, base, base, points - 3 * base])


def _dash_pred_margin(home, away):
    return 7.0 if home == "Penn State" else -4.0 if away == "Penn State" else 1.0


def seed_dashboard_db(con, *, with_model=True):
    """A small four-team world for dashboard tests, loaded like the real pipeline (upsert, then build).

    Penn State goes 2-0 in 2025 (1-0 in the Big Ten, winning 27-24 at Ohio State) and is 1-0 in 2026 with
    Ohio State still to play; 2027 has one unplayed game. Each played game has one 8-play drive per team
    (home in quarters 1-2, away in quarters 3-4), box stats and QB/RB/WR box lines plus CFBD's " Team" row.
    with_model=False skips game_predictions and the sim_* tables (the state before `psu train`).
    """
    from scipy.stats import norm

    from psu.build import _write, build
    from psu.db import SPECS, upsert

    games, plays, drives, box, players = [], [], [], [], []
    for gid, season, week, home, away, hp, ap in DASH_GAMES:
        done = hp is not None
        games.append(
            {
                "id": gid,
                "season": season,
                "week": week,
                "season_type": "regular",
                "start_date": pd.Timestamp(season, 9, 6) + pd.Timedelta(days=7 * (week - 1)),
                "completed": done,
                "neutral_site": False,
                "conference_game": DASH_CONF[home] == DASH_CONF[away],
                "home_team": home,
                "home_conference": DASH_CONF[home],
                "home_classification": "fbs",
                "away_team": away,
                "away_conference": DASH_CONF[away],
                "away_classification": "fbs",
                "home_points": hp,
                "away_points": ap,
                "home_line_scores": _dash_quarters(hp) if done else None,
                "away_line_scores": _dash_quarters(ap) if done else None,
                "notes": None,
            }
        )
        if not done:
            continue
        for offense, defense, points in ((home, away, hp), (away, home, ap)):
            first = offense == home
            prefix = offense.split()[0].lower()
            side = "home" if first else "away"
            before = 0 if first else min(hp, 14)  # home scored before the away drive; keep margins out of garbage time
            scored = points >= 20
            drives.append(
                {
                    "id": f"{gid}-{prefix}",
                    "game_id": gid,
                    "season": season,
                    "offense": offense,
                    "offense_conference": DASH_CONF[offense],
                    "defense": defense,
                    "defense_conference": DASH_CONF[defense],
                    "drive_number": 1 if first else 2,
                    "start_period": 1 if first else 3,
                    "start_yards_to_goal": 75,
                    "end_yards_to_goal": 5 if scored else 40,
                    "plays": 8,
                    "yards": 70 if scored else 35,
                    "drive_result": "TD" if scored else "PUNT",
                    "elapsed_minutes": 4,
                    "elapsed_seconds": 30,
                    "start_offense_score": 0,
                    "start_defense_score": before,
                    "end_offense_score": 7 if scored else 0,
                    "end_defense_score": before,
                }
            )
            for i in range(8):
                plays.append(
                    {
                        "id": f"{gid}-{prefix}-{i}",
                        "game_id": gid,
                        "drive_id": f"{gid}-{prefix}",
                        "season": season,
                        "week": week,
                        "season_type": "regular",
                        "drive_number": 1 if first else 2,
                        "play_number": i + 1,
                        "offense": offense,
                        "offense_conference": DASH_CONF[offense],
                        "defense": defense,
                        "defense_conference": DASH_CONF[defense],
                        "home": home,
                        "away": away,
                        "period": 1 + i // 4 + (0 if first else 2),
                        "clock_minutes": 14 - 3 * (i % 4),
                        "clock_seconds": 0,
                        "offense_score": 0,
                        "defense_score": before,
                        "down": 1 + i % 3,
                        "distance": 10,
                        "yards_to_goal": 75 - 8 * i,
                        "yards_gained": 8,
                        "play_type": "Rush" if i % 2 else "Pass Reception",
                        "play_text": f"{offense} play {i + 1}",
                        "ppa": round(DASH_STRENGTH[offense] + (0.4 if i == 3 else -0.1 if i % 2 else 0.1), 3),
                    }
                )
            for category, stat in (
                ("totalYards", str(300 + points)),
                ("netPassingYards", "200"),
                ("rushingYards", str(100 + points)),
                ("firstDowns", "18"),
                ("thirdDownEff", "5-12"),
                ("turnovers", "0" if scored else "1"),
                ("tacklesForLoss", "4"),
                ("possessionTime", "30:00"),
            ):
                box.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "week": week,
                        "season_type": "regular",
                        "team": offense,
                        "conference": DASH_CONF[offense],
                        "home_away": side,
                        "points": points,
                        "category": category,
                        "stat": stat,
                    }
                )
            for category, athlete, stats in (
                ("passing", "QB", {"C/ATT": "20/30", "YDS": str(150 + points), "TD": "2", "INT": "1", "AVG": "7.0"}),
                ("rushing", "RB", {"CAR": "15", "YDS": str(60 + points), "TD": "1", "LONG": "25", "AVG": "5.0"}),
                ("receiving", "WR", {"REC": "6", "YDS": "90", "TD": "1", "LONG": "40", "AVG": "15.0"}),
                ("rushing", " Team", {"CAR": "2", "YDS": "-3", "TD": "0", "LONG": "0"}),
            ):
                name = athlete if athlete == " Team" else f"{offense} {athlete}"
                for stat_type, stat in stats.items():
                    players.append(
                        {
                            "game_id": gid,
                            "season": season,
                            "week": week,
                            "season_type": "regular",
                            "team": offense,
                            "conference": DASH_CONF[offense],
                            "home_away": side,
                            "category": category,
                            "stat_type": stat_type,
                            "athlete_id": f"{prefix}-{athlete.strip().lower()}",
                            "athlete_name": name,
                            "stat": stat,
                        }
                    )
    games = pd.DataFrame(games)
    games[["home_points", "away_points"]] = games[["home_points", "away_points"]].astype("Int64")
    upsert(con, SPECS["games"], games)
    upsert(con, SPECS["plays"], pd.DataFrame(plays))
    upsert(con, SPECS["drives"], pd.DataFrame(drives))
    upsert(con, SPECS["team_game_stats"], pd.DataFrame(box))
    upsert(con, SPECS["player_game_stats"], pd.DataFrame(players))
    build(con)
    if not with_model:
        return
    preds = pd.DataFrame(
        [
            {
                "game_id": gid,
                "season": season,
                "week": week,
                "season_type": "regular",
                "start_date": pd.Timestamp(season, 9, 6) + pd.Timedelta(days=7 * (week - 1)),
                "neutral_site": False,
                "home_team": home,
                "away_team": away,
                "completed": hp is not None,
                "margin": float(hp - ap) if hp is not None else float("nan"),
                "vegas_margin": 3.5,
                "pred_margin": _dash_pred_margin(home, away),
                "home_win_prob": float(norm.cdf(_dash_pred_margin(home, away) / 16.0)),
                "split": "upcoming" if hp is None else "no_prior" if season == 2025 else "in_sample",
            }
            for gid, season, week, home, away, hp, ap in DASH_GAMES
        ]
    )
    _write(con, "game_predictions", preds)
    _write(
        con,
        "sim_team_summary",
        pd.DataFrame(
            [
                {
                    "season": 2026,
                    "team": "Penn State",
                    "n_sims": 1000,
                    "seed": 0,
                    "tau": 5.0,
                    "as_of": pd.Timestamp(2026, 9, 6),
                    "mean_wins": 1.6,
                    "p_10_plus": 0.0,
                    "p_title_game": 0.55,
                    "p_conf_champ": 0.3,
                    "p_cfp": 0.4,
                }
            ]
        ),
    )
    _write(
        con,
        "sim_win_totals",
        pd.DataFrame({"season": 2026, "team": "Penn State", "wins": [0, 1, 2], "prob": [0.0, 0.4, 0.6]}),
    )
    _write(
        con,
        "sim_conference",
        pd.DataFrame(
            {
                "season": 2026,
                "team": ["Ohio State", "Penn State"],
                "mean_conf_wins": [0.6, 0.4],
                "p_title_game": [1.0, 1.0],
                "p_conf_champ": [0.7, 0.3],
            }
        ),
    )
