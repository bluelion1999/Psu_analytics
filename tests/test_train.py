import json

import joblib
from conftest import synthetic_features

from psu.db import connect
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
    assert saved["final_train_games"] == len(gp.training_rows(features))
    md = (tmp_path / "reports" / "game_model.md").read_text(encoding="utf-8")
    assert "Vegas" in md and "Penn State" in md
    report_markdown(report).encode("ascii")  # console-safe


def test_final_model_trains_on_every_completed_game_after_the_first_season(tmp_path, monkeypatch):
    seen = []
    real_fit = gp.fit

    def spy(train, kind):
        seen.append(set(train["season"]))
        return real_fit(train, kind)

    monkeypatch.setattr(gp, "fit", spy)
    train_and_save(connect(":memory:"), synthetic_features(), current_season=2026, out_dir=tmp_path)
    assert seen[-1] == {2023, 2024, 2025, 2026}
