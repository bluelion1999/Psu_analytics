import numpy as np
import pandas as pd
import pytest

from psu.adjust import opponent_adjust

OFFENSE = {"A": 0.3, "B": 0.0, "C": 0.0, "D": 0.0}
DEFENSE = {"A": 0.0, "B": 0.0, "C": 0.0, "D": -0.3}  # D is the elite defense
PAIRS = [("A", "B"), ("A", "D"), ("B", "D"), ("B", "C"), ("C", "A"), ("C", "B"), ("D", "A"), ("D", "C")]


def synthetic(seed=0, n=300, venue="neutral", home_boost=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for off, de in PAIRS:
        y = 0.1 + OFFENSE[off] + DEFENSE[de] + rng.normal(0, 0.05, n)
        v = venue if venue != "alternate" else None
        for i, value in enumerate(y):
            where = v or ("home" if i % 2 == 0 else "away")
            bump = home_boost if where == "home" else -home_boost if where == "away" else 0.0
            rows.append(
                {
                    "season": 2024,
                    "offense": off,
                    "defense": de,
                    "ppa": value + bump,
                    "success": value > 0.1,
                    "venue": where,
                    "garbage": False,
                }
            )
    return pd.DataFrame(rows)


def test_adjustment_removes_schedule_strength():
    out = opponent_adjust(synthetic(), "ppa", alpha=1.0).set_index("team")
    # B's offense faced elite defense D; C's didn't. Raw says C > B; adjusted says they're equal.
    assert out.loc["C", "off_raw"] - out.loc["B", "off_raw"] > 0.1
    assert abs(out.loc["C", "off_adj"] - out.loc["B", "off_adj"]) < 0.03
    assert out.loc["A", "off_adj"] - out.loc["B", "off_adj"] == pytest.approx(0.3, abs=0.03)
    assert out["def_adj"].idxmin() == "D"


def test_success_adjustment_and_output_shape():
    out = opponent_adjust(synthetic(), "success", alpha=1.0)
    assert list(out.columns) == ["season", "team", "off_raw", "off_adj", "def_raw", "def_adj", "off_plays", "def_plays"]
    assert set(out["team"]) == {"A", "B", "C", "D"}
    assert (out["off_plays"] == 600).all() and (out["def_plays"] == 600).all()


def test_home_field_term_does_not_bias_neutral_games():
    boosted = opponent_adjust(synthetic(venue="alternate", home_boost=0.2), "ppa", alpha=1.0).set_index("team")
    neutral = opponent_adjust(synthetic(), "ppa", alpha=1.0).set_index("team")
    diff = boosted["off_adj"] - boosted.loc["B", "off_adj"]
    base = neutral["off_adj"] - neutral.loc["B", "off_adj"]
    assert (diff - base).abs().max() < 0.03  # the home term absorbs home advantage


def test_adjusted_levels_are_centred_on_the_league_average():
    df = synthetic()
    out = opponent_adjust(df, "ppa", alpha=1.0)
    assert np.average(out["off_adj"], weights=out["off_plays"]) == pytest.approx(df["ppa"].mean(), abs=1e-9)
    assert np.average(out["def_adj"], weights=out["def_plays"]) == pytest.approx(df["ppa"].mean(), abs=1e-9)


def test_garbage_time_excluded_by_default_and_value_validated():
    df = pd.concat(
        [
            synthetic(),
            pd.DataFrame(
                [
                    {
                        "season": 2024,
                        "offense": "B",
                        "defense": "C",
                        "ppa": 50.0,
                        "success": True,
                        "venue": "neutral",
                        "garbage": True,
                    }
                ]
            ),
        ]
    )
    out = opponent_adjust(df, "ppa", alpha=1.0).set_index("team")
    assert out.loc["B", "off_plays"] == 600
    with pytest.raises(ValueError):
        opponent_adjust(df, "yards")
