"""Train the game model from DuckDB: features, backtest, final fit, predictions table, saved model and report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from psu.build import _PLAY_COLUMNS
from psu.config import MODEL_CONFIG, SHRINK_PLAYS, TEAM, TRAIN_ALPHA
from psu.features import game_features
from psu.modelconfig import ModelConfig
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

# 2025 test-season scores before model v2 (preseason priors, phase sigmas), kept to show the change.
BASELINE = {"season": 2025, "model_mae": 12.52, "model_brier": 0.185, "vegas_mae": 11.82, "vegas_brier": 0.175}


@dataclass(frozen=True)
class FeatureInputs:
    enriched: pd.DataFrame
    games: pd.DataFrame
    lines: pd.DataFrame
    sp: pd.DataFrame
    talent: pd.DataFrame


_ELO_COLUMNS = ("home_pregame_elo", "away_pregame_elo")


def _elo_select(con: duckdb.DuckDBPyConnection) -> str:
    """Select the pregame Elo columns, or typed NULLs when an older database's games table lacks them."""
    present = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'games'"
        ).fetchall()
    }
    return ", ".join(c if c in present else f"CAST(NULL AS DOUBLE) AS {c}" for c in _ELO_COLUMNS)


def load_feature_inputs(con: duckdb.DuckDBPyConnection) -> FeatureInputs:
    plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays").df()
    games = con.execute(
        "SELECT id, season, week, season_type, start_date, neutral_site, completed, home_team, away_team, "
        f"home_classification, away_classification, home_points, away_points, {_elo_select(con)} FROM games"
    ).df()
    drives = con.execute("SELECT id, offense, start_offense_score, start_defense_score FROM drives").df()
    return FeatureInputs(
        enriched=enrich_plays(plays, games, drives),
        games=games,
        lines=con.execute("SELECT game_id, spread FROM lines").df(),
        sp=con.execute("SELECT year, team, rating FROM ratings_sp").df(),
        talent=con.execute("SELECT year, team, talent FROM talent").df(),
    )


def load_features(
    con: duckdb.DuckDBPyConnection, *, alpha: float = 20.0, shrink_plays: int = 75, cfg: ModelConfig = MODEL_CONFIG
) -> pd.DataFrame:
    i = load_feature_inputs(con)
    return game_features(
        i.enriched,
        i.games,
        i.lines,
        i.sp,
        i.talent,
        alpha=alpha,
        shrink_plays=shrink_plays,
        metrics=cfg.metrics,
        half_life=cfg.half_life,
    )


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
    ]
    phases = report.get("sigma_by_phase")
    sigma_text = (
        "by phase: " + ", ".join(f"{p} {phases[p]:.1f}" for p in ("early", "mid", "post"))
        if phases
        else f"{report['sigma']:.1f}"
    )
    lines += [
        f"Win probability = NormalCDF(margin / sigma), sigma {sigma_text}; "
        f"Vegas uses sigma {report['vegas_sigma']:.1f}.",
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
    lines += [
        "",
        f"Before model v2 ({BASELINE['season']} test, all games): model MAE {BASELINE['model_mae']:.2f}, "
        f"Brier {BASELINE['model_brier']:.3f}; Vegas MAE {BASELINE['vegas_mae']:.2f}, "
        f"Brier {BASELINE['vegas_brier']:.3f}.",
    ]
    cal = report.get("calibration")
    if cal:
        lines += [
            "",
            "## Calibration (test season, games with a Vegas line)",
            "",
            f"Expected calibration error: model {_fmt(cal['model']['ece'], 3)}, Vegas {_fmt(cal['vegas']['ece'], 3)}.",
            "",
            "| Source | Bin | N | Mean predicted | Actual |",
            "|---|---|---|---|---|",
        ]
        for source in ("model", "vegas"):
            lines += [
                f"| {source} | {b['bin']:.1f} | {b['n']} | {b['mean_pred']:.3f} | {b['actual']:.3f} |"
                for b in cal[source]["bins"]
            ]
    return "\n".join(lines) + "\n"


def _row_weights(frame: pd.DataFrame, cfg: ModelConfig) -> pd.Series | None:
    """Per-row weight from cfg.season_weights, zeroed below cfg.train_from; None when every weight is 1
    (today's behaviour, when there's no season below train_from and no season_weights).
    """
    w = frame["season"].map(cfg.weight_of).astype(float)
    w = w.where(frame["season"] >= cfg.train_from, 0.0)
    return None if (w == 1.0).all() else w


def train_and_save(
    con: duckdb.DuckDBPyConnection,
    features: pd.DataFrame,
    *,
    current_season: int,
    out_dir: Path,
    team: str = TEAM,
    alpha: float = TRAIN_ALPHA,
    shrink_plays: int = SHRINK_PLAYS,
    cfg: ModelConfig = MODEL_CONFIG,
) -> dict:
    from psu.experiment import fit_config, training  # local import: psu.experiment imports psu.train

    cfg.validate()
    report = gp.backtest(
        features, current_season, team=team, model_features=cfg.features, weights=_row_weights(features, cfg)
    )

    played = gp.training_rows(features)
    train, weights = training(played, cfg)
    kind = report["model_kind"] if cfg.model == "select" else cfg.model
    model = fit_config(train, replace(cfg, model=kind), weights)
    model.sigma_by_phase = report["sigma_by_phase"]
    report["final_train_games"] = int(len(train))
    report["alpha"] = alpha
    report["shrink_plays"] = shrink_plays
    report["config"] = asdict(cfg)

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
