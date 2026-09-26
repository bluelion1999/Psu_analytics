import json

import joblib
import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_features

from psu.db import connect
from psu.features import FEATURES
from psu.modelconfig import ModelConfig
from psu.models import game_predict as gp
from psu.train import PREDICTION_COLUMNS, report_markdown, train_and_save


def test_train_and_save_writes_predictions_model_and_report(tmp_path):
    con = connect(":memory:")
    features = synthetic_features()
    report = train_and_save(con, features, current_season=2026, out_dir=tmp_path)

    preds = con.execute("SELECT * FROM game_predictions").df()
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == len(features)
    assert preds["home_win_prob"].between(0, 1).all()
    upcoming = preds[preds["margin"].isna()]
    assert len(upcoming) > 0 and upcoming["pred_margin"].notna().all()

    assert set(preds["split"]) == {"no_prior", "in_sample", "upcoming"}
    assert (preds.loc[preds["margin"].isna(), "split"] == "upcoming").all()
    assert (preds.loc[preds["season"] == 2022, "split"] == "no_prior").all()
    assert (preds.loc[preds["split"] == "in_sample", "season"] >= 2023).all()

    model = joblib.load(tmp_path / "models" / "game_model.joblib")
    assert model.kind == report["model_kind"]
    saved = json.loads((tmp_path / "reports" / "game_model.json").read_text(encoding="utf-8"))
    assert saved["test_season"] == 2025
    assert saved["alpha"] == pytest.approx(20.0) and saved["shrink_plays"] == 75
    assert saved["final_train_games"] == len(gp.training_rows(features))
    md = (tmp_path / "reports" / "game_model.md").read_text(encoding="utf-8")
    assert "Vegas" in md and "Penn State" in md
    report_markdown(report).encode("ascii")  # console-safe


def test_train_and_save_records_custom_alpha_and_shrink_plays(tmp_path):
    con = connect(":memory:")
    report = train_and_save(
        con, synthetic_features(), current_season=2026, out_dir=tmp_path, alpha=5.0, shrink_plays=10
    )
    assert report["alpha"] == pytest.approx(5.0) and report["shrink_plays"] == 10
    saved = json.loads((tmp_path / "reports" / "game_model.json").read_text(encoding="utf-8"))
    assert saved["alpha"] == pytest.approx(5.0) and saved["shrink_plays"] == 10


def test_saved_model_and_markdown_carry_phase_sigmas(tmp_path):
    con = connect(":memory:")
    report = train_and_save(con, synthetic_features(), current_season=2026, out_dir=tmp_path)
    model = joblib.load(tmp_path / "models" / "game_model.joblib")
    assert model.sigma_by_phase == report["sigma_by_phase"]
    text = report_markdown(report)
    assert "by phase" in text and "## Calibration" in text and "Before model v2" in text
    assert "| early |" in text


def test_train_and_save_follows_a_non_default_config(tmp_path):
    con = connect(":memory:")
    features = synthetic_features()
    features["d_elo"] = np.random.default_rng(1).normal(size=len(features))
    cfg = ModelConfig(features=(*FEATURES, "d_elo"), model="linear")

    report = train_and_save(con, features, current_season=2026, out_dir=tmp_path, cfg=cfg)

    preds = con.execute("SELECT * FROM game_predictions").df()
    assert len(preds) == len(features)

    model = joblib.load(tmp_path / "models" / "game_model.joblib")
    assert "d_elo" in model.features
    assert model.kind == "linear"

    assert tuple(report["config"]["features"]) == cfg.features
    saved = json.loads((tmp_path / "reports" / "game_model.json").read_text(encoding="utf-8"))
    assert saved["config"]["model"] == "linear"
    assert "d_elo" in saved["config"]["features"]


def test_row_weights_zeroes_seasons_below_train_from():
    from psu.train import _row_weights

    frame = pd.DataFrame({"season": [2021, 2022, 2023, 2024]})
    w = _row_weights(frame, ModelConfig(train_from=2023))
    assert w is not None
    assert w.tolist() == [0.0, 0.0, 1.0, 1.0]

    # No season below train_from and no season_weights: still collapses to None (today's default behaviour).
    assert _row_weights(frame, ModelConfig(train_from=2021)) is None

    # train_from combines with season_weights rather than overriding them.
    w2 = _row_weights(frame, ModelConfig(train_from=2022, season_weights=((2023, 0.5),)))
    assert w2.tolist() == [0.0, 1.0, 0.5, 1.0]


def test_backtest_honours_train_from(tmp_path, monkeypatch):
    """train_and_save must forward weights honouring cfg.train_from to gp.backtest.

    The real gp.backtest is still exercised (for report_markdown's sake) but with weights=None once captured,
    since a training season entirely zeroed out empties a fold's fit -- a pre-existing, unrelated limitation of
    gp.backtest's walk-forward residuals that this fix doesn't need to solve.
    """
    real_backtest = gp.backtest
    captured = {}

    def spy(features, current_season, team="Penn State", **kw):
        captured["weights"] = kw.get("weights")
        return real_backtest(features, current_season, team=team, **{**kw, "weights": None})

    monkeypatch.setattr(gp, "backtest", spy)
    features = synthetic_features()  # seasons 2022-2026
    cfg = ModelConfig(train_from=2024)

    train_and_save(connect(":memory:"), features, current_season=2026, out_dir=tmp_path, cfg=cfg)

    weights = captured["weights"]
    assert weights is not None
    below = features["season"] < 2024
    assert below.any()
    assert (weights[below] == 0).all()
    assert (weights[~below] == 1.0).all()


def test_final_model_trains_on_every_completed_game_after_the_first_season(tmp_path, monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind, **kwargs):
        seen.append(set(train["season"]))
        return real_fit(train, kind, **kwargs)

    monkeypatch.setattr(gp, "fit", spy)
    train_and_save(connect(":memory:"), synthetic_features(), current_season=2026, out_dir=tmp_path)
    assert seen[-1] == {2023, 2024, 2025, 2026}
