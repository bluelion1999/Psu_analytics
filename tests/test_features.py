import numpy as np
import pandas as pd
import pytest

from psu.features import (
    MAX_REST, league_means, rest_days, rolling_ratings, season_ratings, slate_index, vegas_margin,
)

SCHEDULE = [(1, "A", "B"), (1, "C", "D"), (2, "A", "C"), (2, "B", "D"), (3, "A", "D"), (3, "B", "C")]
QUALITY = {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4, "E": 0.0}


def make_games(season, schedule=SCHEDULE, start_id=0):
    return pd.DataFrame([
        {"id": season * 100 + start_id + i, "season": season, "week": week, "season_type": "regular",
         "start_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
         "neutral_site": False, "completed": True, "home_team": home, "away_team": away,
         "home_classification": "fbs", "away_classification": "fbs", "home_points": 28, "away_points": 21}
        for i, (week, home, away) in enumerate(schedule)
    ])


def make_plays(games, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in games.itertuples():
        for off, de in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            for v in QUALITY[off] - QUALITY[de] + rng.normal(0, 0.1, n):
                rows.append({"game_id": g.id, "season": g.season, "offense": off, "defense": de,
                             "ppa": v, "success": v > 0, "venue": "neutral", "garbage": False})
    return pd.DataFrame(rows)


def test_slate_index_puts_postseason_after_regular_season():
    games = pd.DataFrame([
        {"id": 1, "season": 2024, "week": 14, "season_type": "regular"},
        {"id": 2, "season": 2024, "week": 1, "season_type": "postseason"},
        {"id": 3, "season": 2024, "week": 3, "season_type": "regular"},
    ])
    assert slate_index(games).set_index("id")["slate"].to_dict() == {1: 14, 2: 15, 3: 3}


def test_rest_days_between_games_and_capped_for_openers():
    rest = rest_days(make_games(2024)).set_index("id")
    assert rest.loc[202400, "home_rest"] == MAX_REST
    assert rest.loc[202402, ["home_rest", "away_rest"]].tolist() == [7, 7]


def test_vegas_margin_is_minus_median_spread():
    lines = pd.DataFrame({"game_id": [1, 1, 1, 2], "spread": [-7.0, -6.5, -7.5, None]})
    assert vegas_margin(lines).set_index("game_id")["vegas_margin"].to_dict() == {1: 7.0}


def test_league_means_skip_garbage():
    plays = pd.DataFrame({"ppa": [1.0, 0.0, 9.0], "success": [True, False, True], "garbage": [False, False, True]})
    assert league_means(plays) == {"epa": 0.5, "sr": 0.5}


def test_rolling_ratings_use_only_earlier_slates():
    games = make_games(2024)
    plays = make_plays(games)
    base = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10)

    late = plays.copy()
    late.loc[late["game_id"].isin(games.loc[games["week"] == 3, "id"]), "ppa"] = 100.0
    pd.testing.assert_frame_equal(base, rolling_ratings(late, games, alpha=1.0, shrink_plays=10))

    mid = plays.copy()
    mid.loc[mid["game_id"].isin(games.loc[games["week"] == 2, "id"]), "ppa"] = 100.0
    changed = rolling_ratings(mid, games, alpha=1.0, shrink_plays=10).set_index(["season", "slate", "team"])
    b = base.set_index(["season", "slate", "team"])
    pd.testing.assert_frame_equal(b.xs(2, level="slate"), changed.xs(2, level="slate"))
    assert not b.xs(3, level="slate").equals(changed.xs(3, level="slate"))


def test_first_slate_uses_last_season_and_league_mean_for_new_teams():
    g23 = make_games(2023)
    games = pd.concat([g23, make_games(2024, schedule=[(1, "A", "E")])], ignore_index=True)
    plays = make_plays(g23)
    out = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10).set_index(["season", "slate", "team"])
    prior = season_ratings(plays, alpha=1.0)
    means = league_means(plays)
    assert out.loc[(2024, 1, "A"), "off_epa"] == pytest.approx(prior.loc["A", "off_epa"])
    assert out.loc[(2024, 1, "E"), "off_epa"] == pytest.approx(means["epa"])
    assert out.loc[(2024, 1, "E"), "def_sr"] == pytest.approx(means["sr"])
    first = out.xs(2023, level="season").xs(1, level="slate")
    assert first.isna().all().all()  # first season, first slate: nothing known yet


def test_shrinkage_blends_prior_and_current():
    g23, g24 = make_games(2023), make_games(2024)
    games = pd.concat([g23, g24], ignore_index=True)
    plays = pd.concat([make_plays(g23), make_plays(g24, seed=1)], ignore_index=True)
    early_2024 = plays[plays["game_id"].isin(g24.loc[g24["week"] < 3, "id"])]
    current = season_ratings(early_2024, alpha=1.0)
    prior = season_ratings(plays[plays["season"] == 2023], alpha=1.0)
    no_shrink = rolling_ratings(plays, games, alpha=1.0, shrink_plays=0).set_index(["season", "slate", "team"])
    all_prior = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10**9).set_index(["season", "slate", "team"])
    assert no_shrink.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(current.loc["A", "off_epa"])
    assert all_prior.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(prior.loc["A", "off_epa"], abs=1e-6)
