import numpy as np
import pytest

from conftest import synthetic_features
from psu.models import game_predict as gp


def test_win_prob_is_a_normal_cdf_of_margin():
    assert gp.win_prob(0.0, 14.0) == pytest.approx(0.5)
    p = gp.win_prob(np.array([-14.0, 7.0, 14.0]), 14.0)
    assert p[0] == pytest.approx(0.1587, abs=1e-4) and p[2] == pytest.approx(0.8413, abs=1e-4)
    assert p[0] < p[1] < p[2]


def test_scores():
    assert gp.scores([7, -3], [3, -1], [0.6, 0.4]) == {"n": 2, "mae": pytest.approx(3.0), "brier": pytest.approx(0.16)}
    assert gp.scores([], [], []) == {"n": 0, "mae": None, "brier": None}


@pytest.mark.parametrize("kind", gp.MODEL_KINDS)
def test_fit_learns_signal_and_estimates_sigma(kind):
    f = synthetic_features()
    train = f[f["season"].isin([2023, 2024])].copy()
    test = f[f["season"] == 2025].copy()
    train.loc[train.index[:20], "d_prior_sp"] = np.nan  # missing values are imputed
    test.loc[test.index[:5], ["d_prior_sp", "d_talent"]] = np.nan
    model = gp.fit(train, kind)
    predicted = model.predict_margin(test)
    assert np.isfinite(predicted).all()
    assert np.mean(np.abs(test["margin"] - predicted)) < np.mean(np.abs(test["margin"]))
    assert 8 < model.sigma < 20


def test_backtest_folds_are_time_ordered(monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind):
        seen.append(set(train["season"]))
        return real_fit(train, kind)

    monkeypatch.setattr(gp, "fit", spy)
    report = gp.backtest(synthetic_features(), current_season=2026)
    assert (report["validation_season"], report["test_season"]) == (2024, 2025)
    assert report["train_seasons"] == [2023, 2024]
    assert seen[: len(gp.MODEL_KINDS)] == [{2023}] * len(gp.MODEL_KINDS)
    assert seen[-1] == {2023, 2024}
    assert report["model_kind"] in gp.MODEL_KINDS
    for block in ("all", "Penn State"):
        assert set(report["test"][block]) == {"model", "model_on_lined_games", "vegas"}
        assert report["test"][block]["model"]["n"] > 0
        assert report["test"][block]["vegas"]["n"] == report["test"][block]["model_on_lined_games"]["n"]
    assert report["test"]["all"]["vegas"]["n"] > 0 and report["test"]["all"]["vegas"]["mae"] is not None
    assert isinstance(report["test_season"], int) and isinstance(report["sigma"], float)


def test_backtest_needs_three_complete_seasons():
    f = synthetic_features()
    with pytest.raises(ValueError):
        gp.backtest(f[f["season"] >= 2024], current_season=2026)


def test_training_rows_skip_first_season_and_unplayed_games():
    rows = gp.training_rows(synthetic_features())
    assert set(rows["season"]) == {2023, 2024, 2025, 2026}
    assert rows["margin"].notna().all()
