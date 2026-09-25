"""Train the game model from DuckDB: features, backtest, final fit, predictions table, saved model and report."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from psu.build import _PLAY_COLUMNS
from psu.config import TEAM
from psu.features import game_features
from psu.models import game_predict as gp
from psu.transform import enrich_plays

PREDICTION_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "neutral_site",
    "home_team",
    "away_team",
    "completed",
    "margin",
    "vegas_margin",
    "pred_margin",
    "home_win_prob",
    "split",
]


def load_features(con: duckdb.DuckDBPyConnection, *, alpha: float = 20.0, shrink_plays: int = 75) -> pd.DataFrame:
    plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays").df()
    games = con.execute(
        "SELECT id, season, week, season_type, start_date, neutral_site, completed, home_team, away_team, "
        "home_classification, away_classification, home_points, away_points FROM games"
    ).df()
    drives = con.execute("SELECT id, offense, start_offense_score, start_defense_score FROM drives").df()
    lines = con.execute("SELECT game_id, spread FROM lines").df()
    sp = con.execute("SELECT year, team, rating FROM ratings_sp").df()
    talent = con.execute("SELECT year, team, talent FROM talent").df()
    enriched = enrich_plays(plays, games, drives)
    return game_features(enriched, games, lines, sp, talent, alpha=alpha, shrink_plays=shrink_plays)


def _fmt(value, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def report_markdown(report: dict) -> str:
    validation = ", ".join(f"{kind} {_fmt(s['mae'], 2)}" for kind, s in report["validation"].items())
    seasons = "-".join(str(s) for s in (report["train_seasons"][0], report["train_seasons"][-1]))
    lines = [
        "# Game model report",
        "",
        f"Model: {report['model_kind']} (chosen on {report['validation_season']} validation MAE: {validation}).",
        f"Trained on {seasons}, tested on {report['test_season']}; final model refit on "
        f"{report.get('final_train_games', '?')} completed games.",
        f"Win probability = NormalCDF(margin / {report['sigma']:.1f}); Vegas uses sigma {report['vegas_sigma']:.1f}.",
        "",
        "| Games | N (lined) | Model MAE | Vegas MAE | Model Brier | Vegas Brier | Model MAE (all games) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, block in report["test"].items():
        lined, vegas, every = block["model_on_lined_games"], block["vegas"], block["model"]
        lines.append(
            f"| {name} | {lined['n']} | {_fmt(lined['mae'], 2)} | {_fmt(vegas['mae'], 2)} | "
            f"{_fmt(lined['brier'], 3)} | {_fmt(vegas['brier'], 3)} | {_fmt(every['mae'], 2)} (n={every['n']}) |"
        )
    return "\n".join(lines) + "\n"


def train_and_save(
    con: duckdb.DuckDBPyConnection,
    features: pd.DataFrame,
    *,
    current_season: int,
    out_dir: Path,
    team: str = TEAM,
) -> dict:
    report = gp.backtest(features, current_season, team=team)
    train = gp.training_rows(features)
    model = gp.fit(train, report["model_kind"])
    report["final_train_games"] = int(len(train))

    predictions = features.copy()
    predictions["pred_margin"] = model.predict_margin(features)
    predictions["home_win_prob"] = model.win_prob(features)

    min_season = features["season"].min()
    predictions["split"] = np.select(
        [
            predictions["margin"].isna(),
            predictions["season"] == min_season,
        ],
        [
            "upcoming",
            "no_prior",
        ],
        default="in_sample",
    )

    predictions = predictions[PREDICTION_COLUMNS]
    con.register("_predictions", predictions)
    try:
        con.execute("CREATE OR REPLACE TABLE game_predictions AS SELECT * FROM _predictions")
    finally:
        con.unregister("_predictions")

    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "models" / "game_model.joblib")
    (out_dir / "reports" / "game_model.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "reports" / "game_model.md").write_text(report_markdown(report), encoding="utf-8")
    return report
