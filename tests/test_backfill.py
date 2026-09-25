import pandas as pd
from conftest import synthetic_league

from psu.backfill import replay_games, simulate_as_of

NOW = pd.Timestamp(2026, 9, 2)  # just after synthetic_league's played games; keeps every open week within the grace


def all_predictions(games):
    """A prediction for every synthetic-league game, including the two already played."""
    fixed = {1: 5.0, 2: 3.0, 9: 30.0}
    _, upcoming = synthetic_league()
    played = pd.DataFrame(
        [
            {
                "game_id": g.id,
                "home_team": g.home_team,
                "away_team": g.away_team,
                "neutral_site": False,
                "pred_margin": fixed[g.id],
            }
            for g in games.itertuples()
            if g.id in fixed
        ]
    )
    return pd.concat([upcoming, played], ignore_index=True)


def test_replay_games_reopens_the_as_of_week_and_later():
    games, _ = synthetic_league()
    replay = replay_games(games, 1)
    assert not replay["completed"].any()
    assert replay["home_points"].isna().all()
    assert replay_games(games, 2)["completed"].tolist() == games["completed"].tolist()  # week 1 stays played


def test_simulate_as_of_replays_each_finished_week():
    games, _ = synthetic_league()
    seen = []

    def predict(n):
        seen.append(n)
        return all_predictions(games)

    results = simulate_as_of(games, predict, season=2026, sigma=16.0, team="B", n_sims=500, seed=0, tau=0.1, now=NOW)
    assert seen == [1]  # week 1 is the only finished week in the synthetic league
    assert [r.as_of_slate for r in results] == [1]
    # As of week 1, B's loss to A hasn't happened yet, so B can still win all three of its games.
    totals = results[0].win_totals
    assert totals.loc[totals["wins"] == totals["wins"].max(), "prob"].item() > 0


def test_backfill_before_any_games_writes_nothing():
    games, _ = synthetic_league()
    unplayed = replay_games(games, 1)
    results = simulate_as_of(
        unplayed,
        lambda n: all_predictions(games),
        season=2026,
        sigma=16.0,
        team="A",
        n_sims=50,
        seed=0,
        tau=0.1,
        now=NOW,
    )
    assert results == []
