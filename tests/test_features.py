import numpy as np
import pandas as pd
import pytest

from psu.features import (
    FEATURES,
    MAX_REST,
    game_features,
    league_means,
    rest_days,
    rolling_ratings,
    season_ratings,
    slate_index,
    vegas_margin,
)
from psu.priors import FALLBACK_B0

SCHEDULE = [(1, "A", "B"), (1, "C", "D"), (2, "A", "C"), (2, "B", "D"), (3, "A", "D"), (3, "B", "C")]
QUALITY = {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4, "E": 0.0}


def make_games(season, schedule=SCHEDULE, start_id=0):
    return pd.DataFrame(
        [
            {
                "id": season * 100 + start_id + i,
                "season": season,
                "week": week,
                "season_type": "regular",
                "start_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                "neutral_site": False,
                "completed": True,
                "home_team": home,
                "away_team": away,
                "home_classification": "fbs",
                "away_classification": "fbs",
                "home_points": 28,
                "away_points": 21,
            }
            for i, (week, home, away) in enumerate(schedule)
        ]
    )


def make_plays(games, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in games.itertuples():
        for off, de in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            for v in QUALITY[off] - QUALITY[de] + rng.normal(0, 0.1, n):
                rows.append(
                    {
                        "game_id": g.id,
                        "season": g.season,
                        "offense": off,
                        "defense": de,
                        "ppa": v,
                        "success": v > 0,
                        "venue": "neutral",
                        "garbage": False,
                    }
                )
    return pd.DataFrame(rows)


def test_slate_index_puts_postseason_after_regular_season():
    games = pd.DataFrame(
        [
            {"id": 1, "season": 2024, "week": 14, "season_type": "regular"},
            {"id": 2, "season": 2024, "week": 1, "season_type": "postseason"},
            {"id": 3, "season": 2024, "week": 3, "season_type": "regular"},
        ]
    )
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
    projected = means["epa"] + FALLBACK_B0 * (prior.loc["A", "off_epa"] - means["epa"])
    assert out.loc[(2024, 1, "A"), "off_epa"] == pytest.approx(projected)
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
    last = plays[plays["season"] == 2023]
    prior = season_ratings(last, alpha=1.0)
    mean = league_means(last)["epa"]
    projected = mean + FALLBACK_B0 * (prior.loc["A", "off_epa"] - mean)
    no_shrink = rolling_ratings(plays, games, alpha=1.0, shrink_plays=0).set_index(["season", "slate", "team"])
    all_prior = rolling_ratings(plays, games, alpha=1.0, shrink_plays=10**9).set_index(["season", "slate", "team"])
    assert no_shrink.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(current.loc["A", "off_epa"])
    assert all_prior.loc[(2024, 3, "A"), "off_epa"] == pytest.approx(projected, abs=1e-6)


def test_prior_with_few_teams_falls_back_and_ignores_returning_production():
    seasons = [2021, 2022, 2023, 2024]
    games = pd.concat([make_games(s) for s in seasons], ignore_index=True)
    plays = pd.concat([make_plays(make_games(s), seed=s) for s in seasons[:-1]], ignore_index=True)
    teams = ["A", "B", "C", "D"]
    high = pd.DataFrame({"season": 2024, "team": teams, "percent_ppa": 0.95})
    low = high.assign(percent_ppa=0.05)
    kw = {"alpha": 1.0, "shrink_plays": 10}
    hi = rolling_ratings(plays, games, returning=high, **kw).set_index(["season", "slate", "team"])
    lo = rolling_ratings(plays, games, returning=low, **kw).set_index(["season", "slate", "team"])
    assert np.isfinite(hi.loc[(2024, 1, "A"), "off_epa"])
    # Only 4 teams, so no season has MIN_PAIRS pairs: the fallback ignores returning production.
    assert hi.loc[(2024, 1, "A"), "off_epa"] == pytest.approx(lo.loc[(2024, 1, "A"), "off_epa"])


def test_rolling_ratings_fit_each_season_on_earlier_pairs_only(monkeypatch):
    import psu.features as features

    seen = []
    real = features.season_projections

    def spy(pairs, season):
        seen.append(season)
        return real(pairs, season)

    monkeypatch.setattr(features, "season_projections", spy)
    games = pd.concat([make_games(s) for s in (2022, 2023, 2024)], ignore_index=True)
    plays = make_plays(games)
    rolling_ratings(plays, games, alpha=1.0, shrink_plays=10)
    assert seen == [2023, 2024]  # 2022 has no previous season, so no prior to project
    # season_projections itself keeps only pairs before `season` (tested in test_priors)


def test_game_features_include_d_returning():
    g23, g24 = make_games(2023), make_games(2024)
    games = pd.concat([g23, g24], ignore_index=True)
    plays = make_plays(pd.concat([g23, g24], ignore_index=True))
    lines = pd.DataFrame({"game_id": [202400], "spread": [-3.5]})
    sp = pd.DataFrame({"year": [2023], "team": ["A"], "rating": [1.0]})  # non-empty: keeps merge dtypes numeric
    talent = pd.DataFrame({"year": [2024], "team": ["A"], "talent": [1.0]})
    returning = pd.DataFrame({"season": [2024, 2024], "team": ["A", "B"], "percent_ppa": [0.8, 0.3]})
    f = game_features(plays, games, lines, sp, talent, alpha=1.0, shrink_plays=10, returning=returning).set_index(
        "game_id"
    )
    assert FEATURES[-1] == "d_returning"
    assert f.loc[202400, "d_returning"] == pytest.approx(0.5)  # 2024 week 1: A (home) vs B
    assert np.isnan(f.loc[202401, "d_returning"])  # C vs D: no rows, left for the imputer


def test_game_features_are_home_minus_away_and_fbs_only():
    g23, g24 = make_games(2023), make_games(2024)
    g24.loc[0, "neutral_site"] = True
    g24.loc[5, ["completed", "home_points", "away_points"]] = [False, np.nan, np.nan]
    fcs = make_games(2024, schedule=[(4, "A", "F")], start_id=50)
    fcs["away_classification"] = "fcs"
    games = pd.concat([g23, g24, fcs], ignore_index=True)
    plays = make_plays(pd.concat([g23, g24], ignore_index=True))
    lines = pd.DataFrame({"game_id": [202400], "spread": [-3.5]})
    sp = pd.DataFrame({"year": [2023, 2023, 2024], "team": ["A", "B", "A"], "rating": [20.0, 5.0, 99.0]})
    talent = pd.DataFrame({"year": [2024, 2024], "team": ["A", "B"], "talent": [900.0, 700.0]})

    f = game_features(plays, games, lines, sp, talent, alpha=1.0, shrink_plays=10).set_index("game_id")
    assert set(FEATURES) <= set(f.columns)
    assert 202450 not in f.index  # FCS opponent excluded
    row = f.loc[202400]  # 2024 week 1: A (home) vs B, neutral site
    assert row["home_field"] == 0 and row["margin"] == 7 and row["vegas_margin"] == 3.5
    assert row["d_prior_sp"] == 15.0  # 2023 SP+, not 2024's 99
    assert row["d_talent"] == 200.0
    assert row["d_off_epa"] > 0  # A's 2023 offense beat B's
    assert f.loc[202401, "home_field"] == 1
    assert np.isnan(f.loc[202405, "margin"])
    assert np.isnan(f.loc[202401, "vegas_margin"])
