import numpy as np
import pandas as pd
import pytest

from psu.priors import (
    DEFAULT_RETURNING,
    FALLBACK_B0,
    MIN_PAIRS,
    Projection,
    empty_returning,
    fit_projection,
    project_prior,
    projection_pairs,
    returning_pct,
    season_projections,
)

MEANS = {s: {"epa": 0.0, "sr": 0.4} for s in (2022, 2023, 2024)}


def synthetic(b0=0.3, b1=0.5, d=0.55, n=200, seed=0, first=2022):
    """Final ratings for `first` and `first + 1` where next = (b0 + b1*ret)*last (offense) and d*last (defense)."""
    rng = np.random.default_rng(seed)
    teams = pd.Index([f"T{i}" for i in range(n)], name="team")
    ret = pd.Series(rng.uniform(0.2, 0.9, n), index=teams)
    last = pd.DataFrame({"off_epa": rng.normal(0, 0.1, n), "def_epa": rng.normal(0, 0.1, n)}, index=teams)
    nxt = pd.DataFrame(
        {
            "off_epa": (b0 + b1 * ret) * last["off_epa"] + rng.normal(0, 0.002, n),
            "def_epa": d * last["def_epa"] + rng.normal(0, 0.002, n),
        },
        index=teams,
    )
    returning = pd.DataFrame({"season": first + 1, "team": teams, "percent_ppa": ret.to_numpy()})
    return {first: last, first + 1: nxt}, returning


def test_fit_recovers_offense_and_defense_regression():
    finals, ret = synthetic()
    off = fit_projection(projection_pairs(finals, MEANS, ret, "off_epa"), "off_epa")
    de = fit_projection(projection_pairs(finals, MEANS, ret, "def_epa"), "def_epa")
    assert off.b0 == pytest.approx(0.3, abs=0.03) and off.b1 == pytest.approx(0.5, abs=0.05)
    assert de.b0 == pytest.approx(0.55, abs=0.02) and de.b1 == 0.0  # defense ignores returning production


def test_pairs_are_relative_to_each_season_mean():
    finals, ret = synthetic()
    means = {2022: {"epa": 0.1, "sr": 0.0}, 2023: {"epa": -0.1, "sr": 0.0}}
    pairs = projection_pairs(finals, means, ret, "off_epa")
    assert list(pairs.columns) == ["season", "team", "x", "y", "ret"]
    row = pairs.iloc[0]
    assert row["season"] == 2023
    assert row["x"] == pytest.approx(finals[2022].loc[row["team"], "off_epa"] - 0.1)
    assert row["y"] == pytest.approx(finals[2023].loc[row["team"], "off_epa"] + 0.1)


def test_too_few_pairs_fall_back():
    finals, ret = synthetic(n=MIN_PAIRS - 1)
    assert fit_projection(projection_pairs(finals, MEANS, ret, "off_epa"), "off_epa") == Projection(FALLBACK_B0)
    lone = {2023: finals[2023]}  # no previous season: no pairs at all
    assert projection_pairs(lone, MEANS, ret, "off_epa").empty
    assert fit_projection(projection_pairs(lone, MEANS, ret, "off_epa"), "off_epa") == Projection(FALLBACK_B0)


def test_season_projections_use_only_earlier_seasons():
    early, r1 = synthetic(b0=0.3, b1=0.0, d=0.3, first=2022)
    late, r2 = synthetic(b0=0.9, b1=0.0, d=0.9, first=2023, seed=1)
    # 2024's finals come from an unrelated draw, so 2024 pairs are noise: only the 2023 pairs carry b0 = 0.3.
    finals = {2022: early[2022], 2023: early[2023], 2024: late[2024]}
    returning = pd.concat([r1, r2], ignore_index=True)
    pairs = {c: projection_pairs(finals, MEANS, returning, c) for c in ("off_epa", "def_epa")}
    assert season_projections(pairs, 2023)["off_epa"] == Projection(FALLBACK_B0)  # nothing before 2023
    assert season_projections(pairs, 2024)["off_epa"].b0 == pytest.approx(0.3, abs=0.03)  # 2023 pairs only
    assert season_projections(pairs, 2024)["def_epa"].b0 == pytest.approx(0.3, abs=0.03)


def test_factor_is_clipped_to_unit_interval():
    np.testing.assert_allclose(Projection(0.9, 0.5).factor(np.array([0.0, 0.5, 1.0])), [0.9, 1.0, 1.0])
    np.testing.assert_allclose(Projection(-0.2).factor(np.array([0.3])), [0.0])


def test_returning_pct_fills_missing_teams_with_the_season_median():
    ret = pd.DataFrame(
        {"season": [2024, 2024, 2024, 2023], "team": ["A", "B", "C", "A"], "percent_ppa": [0.2, 0.4, 0.9, 0.99]}
    )
    out = returning_pct(ret, 2024, ["A", "Z"])
    assert list(out.index) == ["A", "Z"] and list(out) == [0.2, 0.4]
    assert list(returning_pct(empty_returning(), 2024, ["A"])) == [DEFAULT_RETURNING]


def test_project_prior_regresses_toward_the_mean():
    last = pd.DataFrame({"off_epa": [0.3, -0.1], "def_epa": [0.2, 0.0]}, index=pd.Index(["A", "B"], name="team"))
    ret = pd.Series([1.0, 0.0], index=["A", "B"])
    projections = {"off_epa": Projection(0.2, 0.6), "def_epa": Projection(0.5)}
    prior = project_prior(last, {"epa": 0.1, "sr": 0.4}, ret, projections)
    assert prior.loc["A", "off_epa"] == pytest.approx(0.1 + 0.8 * 0.2)
    assert prior.loc["B", "off_epa"] == pytest.approx(0.1 + 0.2 * -0.2)
    assert prior.loc["A", "def_epa"] == pytest.approx(0.1 + 0.5 * 0.1)
    assert list(prior.index) == ["A", "B"]
