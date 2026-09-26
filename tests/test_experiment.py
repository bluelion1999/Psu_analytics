import json

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_features

from psu import experiment as ex
from psu.features import FEATURES
from psu.modelconfig import ModelConfig
from psu.models import game_predict as gp


def frame_2020_2025(games_per_season=120):
    """synthetic_features() relabelled to 2020-2025: 2022-2025 as-is, plus copies of 2022/2023 as 2020/2021."""
    base = synthetic_features(games_per_season=games_per_season)
    base = base[base["season"] <= 2025]
    extra = []
    for src, dst in ((2022, 2020), (2023, 2021)):
        part = base[base["season"] == src].copy()
        part["season"] = dst
        extra.append(part)
    out = pd.concat([*extra, base], ignore_index=True)
    out["game_id"] = np.arange(1, len(out) + 1)
    return out


@pytest.fixture
def fit_spy(monkeypatch):
    calls = []
    real_fit = gp.fit

    def spy(train, kind, **kw):
        calls.append({"seasons": sorted(int(s) for s in train["season"].unique()), "kind": kind, **kw})
        return real_fit(train, kind, **kw)

    monkeypatch.setattr(gp, "fit", spy)
    return calls


def test_walk_forward_never_trains_on_the_test_season_or_later(fit_spy):
    frame = frame_2020_2025()
    preds = ex.walk_forward(frame, ModelConfig(model="linear"))
    assert len(fit_spy) == len(ex.TEST_SEASONS)
    for call, season in zip(fit_spy, ex.TEST_SEASONS, strict=True):
        assert call["seasons"] == list(range(2020, season))
    assert sorted(preds["season"].unique()) == list(ex.TEST_SEASONS)
    assert {"game_id", "season", "slate", "season_type", "margin", "pred", "prob"} <= set(preds.columns)
    assert len(preds) == frame[frame["season"].isin(ex.TEST_SEASONS) & frame["margin"].notna()].shape[0]
    assert preds["prob"].between(0, 1).all()


def test_walk_forward_select_validates_on_the_last_training_season(fit_spy):
    frame = frame_2020_2025()
    ex.walk_forward(frame, ModelConfig(), test_seasons=(2023,))
    # linear and xgboost fitted on 2020-2021 (validated on 2022), then the winner refit on 2020-2022
    assert [c["seasons"] for c in fit_spy] == [[2020, 2021], [2020, 2021], [2020, 2021, 2022]]
    assert {c["kind"] for c in fit_spy[:2]} == set(gp.BASE_KINDS)


def test_season_weight_zero_drops_the_season(fit_spy):
    frame = frame_2020_2025()
    cfg = ModelConfig(model="linear", season_weights=((2020, 0.0), (2021, 0.5)))
    ex.walk_forward(frame, cfg)
    for call, season in zip(fit_spy, ex.TEST_SEASONS, strict=True):
        assert call["seasons"] == list(range(2021, season))
        weights = call["weights"]
        assert weights is not None and (weights > 0).all()
    assert cfg.weight_of(2021) == 0.5 and cfg.weight_of(2022) == 1.0


def test_train_from_limits_training_seasons(fit_spy):
    ex.walk_forward(frame_2020_2025(), ModelConfig(model="linear", train_from=2022))
    assert [c["seasons"] for c in fit_spy] == [[2022], [2022, 2023], [2022, 2023, 2024]]
    assert all(c["weights"] is None for c in fit_spy)


def test_tuning_sees_only_training_seasons_and_ensemble_tunes_each_member(monkeypatch, fit_spy):
    tuned = []

    def fake_tune(train, kind, grid, **kw):
        tuned.append((kind, int(train["season"].max()), grid))
        return grid[-1]

    monkeypatch.setattr(gp, "tune", fake_tune)
    ex.walk_forward(frame_2020_2025(), ModelConfig(model="ensemble", tune=True), test_seasons=(2023, 2024))
    assert [(k, s) for k, s, _ in tuned] == [("linear", 2022), ("xgboost", 2022), ("linear", 2023), ("xgboost", 2023)]
    assert tuned[0][2] is gp.LINEAR_GRID and tuned[1][2] is gp.XGB_GRID
    assert fit_spy[0]["params"] == {"linear": gp.LINEAR_GRID[-1], "xgboost": gp.XGB_GRID[-1]}


def test_score_reports_mae_early_mae_brier_and_n():
    preds = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "season": 2023,
            "slate": [1, 5, 15],
            "season_type": ["regular", "regular", "postseason"],
            "margin": [10.0, -3.0, 7.0],
            "pred": [7.0, 1.0, 7.0],
            "prob": [1.0, 0.5, 0.0],
        }
    )
    s = ex.score(preds)
    assert s["n"] == 3
    assert s["mae"] == pytest.approx(7 / 3)
    assert s["early_mae"] == pytest.approx(3.0)
    assert s["brier"] == pytest.approx((0 + 0.25 + 1) / 3)


def _preds(margin, pred):
    n = len(margin)
    return pd.DataFrame({"game_id": np.arange(n), "margin": margin, "pred": pred})


def test_paired_delta_recovers_a_known_difference_and_is_seeded():
    rng = np.random.default_rng(1)
    margin = rng.normal(0, 15, 500)
    err = rng.normal(0, 10, 500)
    err = np.where(np.abs(err) < 1.5, 3.0, err)  # keep |error| > 1 so shrinking by 1 point is exact
    ref = _preds(margin, margin + err)
    cand = _preds(margin, margin + err - np.sign(err))
    cand = cand.sample(frac=1, random_state=0)  # joined on game_id, not row order
    mean, lo, hi = ex.paired_delta(ref, cand)
    assert mean == pytest.approx(-1.0)
    assert lo == pytest.approx(-1.0) and hi == pytest.approx(-1.0)
    noisy = _preds(margin, margin + rng.normal(0, 10, 500))
    first = ex.paired_delta(ref, noisy, seed=3)
    assert first == ex.paired_delta(ref, noisy, seed=3)
    assert first[1] < first[0] < first[2]


def test_adopt_thresholds_and_brier_guard():
    ref = {"mae": 12.0, "brier": 0.180}
    assert ex.adopt(ref, {"mae": 11.9, "brier": 0.180}, (-0.02, -0.1, 0.05))
    assert ex.adopt(ref, {"mae": 11.9, "brier": 0.1805}, (-0.05, -0.1, 0.0))
    assert not ex.adopt(ref, {"mae": 11.99, "brier": 0.170}, (-0.019, -0.1, 0.05))
    assert not ex.adopt(ref, {"mae": 11.5, "brier": 0.1806}, (-0.5, -0.6, -0.4))


def test_validate_rejects_unknown_feature_and_missing_metric():
    ModelConfig().validate()
    with pytest.raises(ValueError, match="Unknown feature"):
        ModelConfig(features=(*FEATURES, "d_vegas")).validate()
    with pytest.raises(ValueError, match="expl"):
        ModelConfig(features=(*FEATURES, "d_off_expl")).validate()
    ModelConfig(features=(*FEATURES, "d_off_expl"), metrics=("epa", "sr", "expl")).validate()
    with pytest.raises(ValueError, match="model"):
        ModelConfig(model="forest").validate()
    with pytest.raises(ValueError, match="metric"):
        ModelConfig(metrics=("epa", "sr", "yards")).validate()


def test_validate_rejects_a_non_positive_half_life():
    ModelConfig(half_life=None).validate()
    ModelConfig(half_life=4.0).validate()
    with pytest.raises(ValueError, match="half_life"):
        ModelConfig(half_life=0).validate()
    with pytest.raises(ValueError, match="half_life"):
        ModelConfig(half_life=-1.0).validate()


def test_validate_rejects_train_from_at_or_before_the_first_season():
    from psu.modelconfig import FIRST_SEASON

    ModelConfig(train_from=FIRST_SEASON + 1).validate()
    with pytest.raises(ValueError, match="train_from"):
        ModelConfig(train_from=FIRST_SEASON).validate()
    with pytest.raises(ValueError, match="train_from"):
        ModelConfig(train_from=FIRST_SEASON - 1).validate()


def test_walk_forward_validates_before_fitting(fit_spy):
    with pytest.raises(ValueError, match="Unknown feature"):
        ex.walk_forward(frame_2020_2025(), ModelConfig(features=("nope",)))
    assert fit_spy == []


CANNED = {
    "B0": {"mae": 12.00, "early_mae": 13.0, "brier": 0.180, "n": 100},
    "D0": {"mae": 12.30, "early_mae": 13.3, "brier": 0.182, "n": 100},
    "D1": {"mae": 11.95, "early_mae": 12.9, "brier": 0.180, "n": 100},  # adopted
    "D2": {"mae": 11.90, "early_mae": 12.9, "brier": 0.1808, "n": 100},  # better MAE but fails the Brier guard
}


def _fake_evaluate(names):
    def evaluate(cfg, ref):
        cand_scores, ref_scores = CANNED[names[cfg]], CANNED[names[ref]]
        d = cand_scores["mae"] - ref_scores["mae"]
        return cand_scores, ref_scores, (d, d - 0.05, d + 0.05)

    return evaluate


def test_select_stage_adopts_the_best_candidate_that_passes_the_rule():
    b0 = ModelConfig()
    cands = [("D0", ModelConfig(train_from=2022)), ("D1", ModelConfig(season_weights=((2020, 0.0),)))]
    cands.append(("D2", ModelConfig(season_weights=((2020, 0.5),))))
    names = {b0: "B0", **{cfg: name for name, cfg in cands}}
    best, rows = ex.select_stage(("B0", b0), cands, _fake_evaluate(names), stage="D")
    assert best == ("D1", cands[1][1])
    assert [r["name"] for r in rows] == ["D0", "D1", "D2"]
    assert [r["adopted"] for r in rows] == [False, True, False]
    assert all(r["reference"] == "B0" and r["stage"] == "D" for r in rows)
    assert rows[1]["delta"] == pytest.approx([-0.05, -0.1, 0.0])


def test_select_stage_keeps_the_reference_when_nothing_is_adopted():
    b0 = ModelConfig()
    d0 = ModelConfig(train_from=2022)
    best, rows = ex.select_stage(("B0", b0), [("D0", d0)], _fake_evaluate({b0: "B0", d0: "D0"}), stage="D")
    assert best == ("B0", b0)
    assert rows[0]["adopted"] is False


def test_run_writes_reports_with_a_row_per_candidate(tmp_path, monkeypatch):
    frame = frame_2020_2025(games_per_season=40)
    built = []
    monkeypatch.setattr(ex, "_check_training_seasons", lambda con: None)
    monkeypatch.setattr(ex, "load_feature_inputs", lambda con: object())

    def fake_features(inputs, half_life):
        built.append(half_life)
        return frame

    monkeypatch.setattr(ex, "_build_frame", fake_features)

    def fake_walk_forward(frame, cfg, test_seasons=ex.TEST_SEASONS):
        cfg.validate()
        test = frame[frame["season"].isin(test_seasons) & frame["margin"].notna()]
        # F1 (d_elo) and M1 (linear) each help; everything else is neutral
        shrink = 0.1 + 0.1 * ("d_elo" in cfg.features) + 0.1 * (cfg.model == "linear")
        pred = test["margin"] * shrink
        return pd.DataFrame(
            {
                "game_id": test["game_id"],
                "season": test["season"],
                "slate": test["slate"],
                "season_type": test["season_type"],
                "margin": test["margin"],
                "pred": pred,
                "prob": gp.win_prob(pred, 15.0),
            }
        )

    monkeypatch.setattr(ex, "walk_forward", fake_walk_forward)
    report = ex.run(None, out_dir=tmp_path)
    names = [r["name"] for r in report["candidates"]]
    assert names[:4] == ["B0", "D0", "D1", "D2"]
    assert {"F1", "F2", "F3", "F4a", "F4b", "M1", "M2", "M3"} <= set(names)
    assert sorted(set(built), key=str) == sorted({None, 4, 8}, key=str)
    final = report["final"]
    assert final["name"] == "M1"
    assert "d_elo" in final["config"]["features"] and final["config"]["model"] == "linear"
    saved = json.loads((tmp_path / "reports" / "experiments.json").read_text(encoding="utf-8"))
    assert saved["final"] == json.loads(json.dumps(final))
    md = (tmp_path / "reports" / "experiments.md").read_text(encoding="utf-8")
    assert "| M1 |" in md and "Final config" in md


def test_folds_without_enough_seasons_to_tune_or_validate_are_counted(caplog):
    frame = frame_2020_2025()
    with caplog.at_level("WARNING", logger="psu.experiment"):
        tuned = ex.walk_forward(frame, ModelConfig(model="linear", tune=True, train_from=2022), test_seasons=(2023,))
    assert tuned.attrs["fallback_folds"] == 1
    assert ex.score(tuned)["fallback_folds"] == 1
    assert "falls back" in caplog.text
    selected = ex.walk_forward(frame, ModelConfig(train_from=2022), test_seasons=(2023,))
    assert selected.attrs["fallback_folds"] == 1
    enough = ex.walk_forward(frame, ModelConfig(model="linear"), test_seasons=(2023,))
    assert enough.attrs["fallback_folds"] == 0


def test_fallback_folds_reach_the_candidate_rows():
    b0, m1 = ModelConfig(), ModelConfig(model="linear", tune=True)

    def evaluate(cfg, ref):
        return {**CANNED["D1"], "fallback_folds": 2}, CANNED["B0"], (-0.05, -0.1, 0.0)

    _, rows = ex.select_stage(("B0", b0), [("M1", m1)], evaluate, stage="M")
    assert rows[0]["fallback_folds"] == 2
    report = {"test_seasons": [2023], "candidates": rows, "final": {"name": "M1", "config": ex.config_dict(m1)}}
    assert "| yes | 2 |" in ex.report_markdown(report)


def test_compare_requires_the_same_games():
    preds = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "season": 2023,
            "slate": [1, 2, 3],
            "season_type": "regular",
            "margin": [3.0, -7.0, 10.0],
            "pred": [1.0, -2.0, 4.0],
            "prob": [0.6, 0.4, 0.7],
        }
    )
    cand_scores, ref_scores, delta = ex.compare(preds, preds.assign(pred=preds["pred"] + 1))
    assert cand_scores["n"] == ref_scores["n"] == 3 and len(delta) == 3
    with pytest.raises(ValueError, match="different games"):
        ex.compare(preds, preds.iloc[:2])
