import json

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_features

from psu.models import game_predict as gp


def test_phase_of():
    phases = gp.phase_of(["regular", "regular", "regular", "postseason"], [1, 4, 5, 16])
    assert list(phases) == ["early", "early", "mid", "post"]


def test_phase_sigmas_use_rms_and_fall_back_below_min_n():
    residuals = pd.DataFrame(
        {"season": 2024, "phase": ["early"] * 40 + ["mid"] * 40 + ["post"] * 5, "residual": [20.0] * 40 + [10.0] * 45}
    )
    sig = gp.phase_sigmas(residuals, fallback=99.0)
    assert sig["early"] == pytest.approx(20.0) and sig["mid"] == pytest.approx(10.0)
    pooled = np.sqrt((40 * 400 + 45 * 100) / 85)
    assert sig["post"] == pytest.approx(pooled)  # only 5 postseason residuals
    empty = pd.DataFrame(columns=["season", "phase", "residual"])
    assert gp.phase_sigmas(empty, fallback=15.0) == {"early": 15.0, "mid": 15.0, "post": 15.0}


def test_oos_residuals_never_train_on_the_season_they_score(monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind, **kwargs):
        seen.append(set(train["season"]))
        return real_fit(train, kind, **kwargs)

    monkeypatch.setattr(gp, "fit", spy)
    f = synthetic_features()
    played = f[f["margin"].notna()]
    res = gp.oos_residuals(played, [2023, 2024, 2025], "linear")
    assert seen == [{2023}, {2023, 2024}]
    assert set(res["season"]) == {2024, 2025}
    assert set(res["phase"]) <= {"early", "mid"}


def test_reliability_bins_and_ece():
    table = gp.reliability([0.05, 0.15, 0.95, 0.99, 1.0], [0, 1, 1, 1, 0])
    assert list(table.columns) == ["bin", "n", "mean_pred", "actual"]
    assert list(table["bin"]) == [0.0, 0.1, 0.9] and list(table["n"]) == [1, 1, 3]
    assert table.loc[2, "mean_pred"] == pytest.approx((0.95 + 0.99 + 1.0) / 3)
    assert table.loc[2, "actual"] == pytest.approx(2 / 3)
    expected = (1 * 0.05 + 1 * 0.85 + 3 * abs((0.95 + 0.99 + 1.0) / 3 - 2 / 3)) / 5
    assert gp.ece(table) == pytest.approx(expected)
    assert gp.ece(gp.reliability([], [])) is None


def test_model_win_prob_uses_phase_sigma():
    f = synthetic_features()
    model = gp.fit(f[f["season"].isin([2023, 2024])], "linear")
    rows = f[f["season"] == 2025].head(20)
    base = model.win_prob(rows)
    model.sigma_by_phase = {"early": 1e6, "mid": model.sigma, "post": model.sigma}
    wide = model.win_prob(rows)
    early = gp.phase_of(rows["season_type"], rows["slate"]) == "early"
    assert np.allclose(wide[early], 0.5, atol=1e-3)  # a huge sigma makes early games coin flips
    assert np.allclose(wide[~early], base[~early])


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

    def spy(train, kind, **kwargs):
        seen.append(set(train["season"]))
        return real_fit(train, kind, **kwargs)

    monkeypatch.setattr(gp, "fit", spy)
    report = gp.backtest(synthetic_features(), current_season=2026)
    assert (report["validation_season"], report["test_season"]) == (2024, 2025)
    assert report["train_seasons"] == [2023, 2024]
    assert seen[: len(gp.BASE_KINDS)] == [{2023}] * len(gp.BASE_KINDS)
    assert seen[-1] == {2023, 2024}
    assert report["model_kind"] in gp.BASE_KINDS
    for block in ("all", "early", "Penn State"):
        assert set(report["test"][block]) == {"model", "model_on_lined_games", "vegas"}
        assert report["test"][block]["model"]["n"] > 0
        assert report["test"][block]["vegas"]["n"] == report["test"][block]["model_on_lined_games"]["n"]
    assert report["test"]["all"]["vegas"]["n"] > 0 and report["test"]["all"]["vegas"]["mae"] is not None
    assert isinstance(report["test_season"], int) and isinstance(report["sigma"], float)
    assert set(report["sigma_by_phase"]) == {"early", "mid", "post"}
    for source in ("model", "vegas"):
        cal = report["calibration"][source]
        assert 0 <= cal["ece"] <= 1 and sum(b["n"] for b in cal["bins"]) > 0
    json.dumps(report)  # the report is written as JSON


def test_backtest_forwards_model_features_to_every_fit_call(monkeypatch):
    seen_features = []
    real_fit = gp.fit

    def spy(train, kind, *, features=gp.FEATURES, **kwargs):
        seen_features.append(tuple(features))
        return real_fit(train, kind, features=features, **kwargs)

    monkeypatch.setattr(gp, "fit", spy)
    custom_features = [c for c in gp.FEATURES if c != "d_talent"]
    gp.backtest(synthetic_features(), current_season=2026, model_features=custom_features)
    assert seen_features  # at least the validation and final-model calls happened
    assert all(f == tuple(custom_features) for f in seen_features)


def test_ensemble_predicts_the_mean_of_its_members():
    f = synthetic_features()
    train = f[f["season"].isin([2023, 2024])]
    test = f[f["season"] == 2025].head(20)
    model = gp.fit(train, "ensemble")
    expected = np.mean([m.predict_margin(test) for m in model.members], axis=0)
    assert np.allclose(model.predict_margin(test), expected)


def test_custom_features_ignore_other_columns():
    f = synthetic_features()
    train = f[f["season"].isin([2023, 2024])]
    test = f[f["season"] == 2025].head(20).drop(columns=["d_talent"])
    model = gp.fit(train, "linear", features=[c for c in gp.FEATURES if c != "d_talent"])
    predicted = model.predict_margin(test)
    assert np.isfinite(predicted).all()


def test_old_pickle_style_model_without_features_still_predicts():
    f = synthetic_features()
    model = gp.fit(f[f["season"].isin([2023, 2024])], "linear")
    del model.features
    assert "features" not in model.__dict__
    predicted = model.predict_margin(f[f["season"] == 2025].head(5))
    assert np.isfinite(predicted).all()


def test_zero_weight_season_matches_dropping_that_season():
    f = synthetic_features()
    train = f[f["season"].isin([2023, 2024, 2025])]
    weights = pd.Series(1.0, index=train.index)
    weights[train["season"] == 2023] = 0.0
    weighted_model = gp.fit(train, "linear", weights=weights)
    dropped_model = gp.fit(train[train["season"] != 2023], "linear")
    weighted_coef = weighted_model.pipeline.named_steps["model"].coef_
    dropped_coef = dropped_model.pipeline.named_steps["model"].coef_
    assert np.allclose(weighted_coef, dropped_coef)


def test_tune_never_sees_rows_from_seasons_it_is_scoring_on(monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind, **kwargs):
        seen.append(set(train["season"]))
        return real_fit(train, kind, **kwargs)

    monkeypatch.setattr(gp, "fit", spy)
    f = synthetic_features()
    train = f[f["season"].isin([2022, 2023, 2024, 2025])]
    gp.tune(train, "linear", gp.LINEAR_GRID[:2], features=gp.FEATURES)
    # inner seasons scored are 2024 and 2025; each fit must only see strictly earlier seasons
    assert seen == [{2022, 2023}, {2022, 2023, 2024}] * 2


def test_tune_picks_the_better_alpha_on_synthetic_data():
    rng = np.random.default_rng(0)
    n_per_season = 200
    rows = []
    for season in (2022, 2023, 2024, 2025):
        x = rng.normal(size=n_per_season)
        y = 10 * x + rng.normal(scale=0.5, size=n_per_season)
        for xi, yi in zip(x, y, strict=True):
            rows.append({"season": season, "d_off_epa": xi, "margin": yi})
    train = pd.DataFrame(rows)
    for c in gp.FEATURES:
        if c not in train.columns:
            train[c] = 0.0
    grid = [{"alpha": 0.01}, {"alpha": 1000.0}]
    best = gp.tune(train, "linear", grid, features=["d_off_epa"])
    assert best == {"alpha": 0.01}


def test_backtest_needs_three_complete_seasons():
    f = synthetic_features()
    with pytest.raises(ValueError):
        gp.backtest(f[f["season"] >= 2024], current_season=2026)


def test_training_rows_skip_first_season_and_unplayed_games():
    rows = gp.training_rows(synthetic_features())
    assert set(rows["season"]) == {2023, 2024, 2025, 2026}
    assert rows["margin"].notna().all()
