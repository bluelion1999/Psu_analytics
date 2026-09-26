"""`psu experiment`: score model configurations with a walk-forward backtest and pick one in greedy stages.

Each candidate is trained on its seasons before each test season T (2023-2025) and scored on T's completed
games. It is compared with its reference through a paired bootstrap of the per-game absolute-error difference,
and adopted only when it lowers MAE by at least 0.02 without worsening the Brier score by more than 0.0005.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from psu import config
from psu.features import METRICS, game_features
from psu.modelconfig import ModelConfig
from psu.models import game_predict as gp
from psu.train import FeatureInputs, load_feature_inputs

log = logging.getLogger(__name__)

TEST_SEASONS = (2023, 2024, 2025)
ALL_METRICS = tuple(METRICS)  # every experiment frame carries all five metrics; configs pick their columns
HALF_LIVES = (None, 4.0, 8.0)
ADOPT_DELTA = -0.02  # mean delta MAE (candidate - reference) must be at most this
BRIER_GUARD = 0.0005  # candidate Brier may be at most this much worse than the reference's
MIN_TRAIN_SEASONS = 2
_EPS = 1e-12  # float slack so values exactly on a threshold count as meeting it

Candidate = tuple[str, ModelConfig]
Evaluate = Callable[[ModelConfig, ModelConfig], tuple[dict, dict, tuple[float, float, float]]]


# --- walk-forward -------------------------------------------------------------------------------------------


def _training(played: pd.DataFrame, cfg: ModelConfig, season: int) -> tuple[pd.DataFrame, pd.Series | None]:
    rows = played[(played["season"] >= cfg.train_from) & (played["season"] < season)]
    w = rows["season"].map(cfg.weight_of).astype(float)
    keep = w > 0
    rows, w = rows[keep], w[keep]
    return rows, (None if (w == 1.0).all() else w)


def _params(train: pd.DataFrame, kind: str, cfg: ModelConfig, weights: pd.Series | None):
    if cfg.tune:
        if kind == "ensemble":
            return {k: _params(train, k, cfg, weights) for k in gp.BASE_KINDS}
        grid = gp.LINEAR_GRID if kind == "linear" else gp.XGB_GRID
        return gp.tune(train, kind, grid, features=cfg.features, weights=weights)
    if cfg.params is None:
        return None
    return cfg.params.get(kind) if cfg.model == "select" else cfg.params


def _mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def _choose_kind(train: pd.DataFrame, cfg: ModelConfig, weights: pd.Series | None) -> str:
    """Today's rule: fit linear and xgboost on all but the last training season and keep the lower-MAE one."""
    seasons = sorted(train["season"].unique())
    if len(seasons) < 2:
        return "linear"
    inner, valid = train[train["season"] < seasons[-1]], train[train["season"] == seasons[-1]]
    maes = {}
    for kind in gp.BASE_KINDS:
        model = gp.fit(inner, kind, features=cfg.features, params=_params(inner, kind, cfg, weights), weights=weights)
        maes[kind] = _mae(valid["margin"], model.predict_margin(valid))
    return min(gp.BASE_KINDS, key=lambda k: maes[k])


def fit_config(train: pd.DataFrame, cfg: ModelConfig, weights: pd.Series | None = None) -> gp.GameModel:
    kind = _choose_kind(train, cfg, weights) if cfg.model == "select" else cfg.model
    return gp.fit(train, kind, features=cfg.features, params=_params(train, kind, cfg, weights), weights=weights)


def walk_forward(frame: pd.DataFrame, cfg: ModelConfig, test_seasons=TEST_SEASONS) -> pd.DataFrame:
    """One row per completed test game, predicted by a model trained only on the config's seasons before it."""
    cfg.validate()
    played = frame[frame["margin"].notna()]
    out = []
    for season in test_seasons:
        test = played[played["season"] == season]
        if test.empty:
            continue
        train, weights = _training(played, cfg, season)
        if train.empty:
            raise ValueError(f"No training seasons before {season} for {cfg}")
        model = fit_config(train, cfg, weights)
        pred = model.predict_margin(test)
        out.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "slate": test["slate"].to_numpy(),
                    "season_type": test["season_type"].to_numpy(),
                    "margin": test["margin"].to_numpy(dtype=float),
                    "pred": pred,
                    "prob": gp.win_prob(pred, model.sigma),
                }
            )
        )
    columns = ["game_id", "season", "slate", "season_type", "margin", "pred", "prob"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


# --- scoring and comparison ---------------------------------------------------------------------------------


def score(pred_frame: pd.DataFrame) -> dict:
    s = gp.scores(pred_frame["margin"], pred_frame["pred"], pred_frame["prob"])
    early = pred_frame[(pred_frame["season_type"] == "regular") & (pred_frame["slate"] <= gp.EARLY_SLATES)]
    return {
        "mae": s["mae"],
        "early_mae": _mae(early["margin"], early["pred"]) if len(early) else None,
        "brier": s["brier"],
        "n": s["n"],
    }


def paired_delta(ref: pd.DataFrame, cand: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0):
    """Mean, 5th and 95th percentile of the bootstrapped mean per-game |error| difference (cand - ref)."""
    joined = ref[["game_id", "margin", "pred"]].merge(cand[["game_id", "pred"]], on="game_id", suffixes=("_r", "_c"))
    if joined.empty:
        raise ValueError("The two prediction frames share no game_id")
    margin = joined["margin"].to_numpy(dtype=float)
    d = np.abs(joined["pred_c"].to_numpy(float) - margin) - np.abs(joined["pred_r"].to_numpy(float) - margin)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_boot)
    chunk = 250  # bounded memory: chunk x n indices at a time
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        boots[start : start + size] = d[rng.integers(0, n, size=(size, n))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def adopt(ref_scores: dict, cand_scores: dict, delta) -> bool:
    delta_mean = delta[0] if isinstance(delta, (tuple, list)) else delta
    return bool(delta_mean <= ADOPT_DELTA + _EPS and cand_scores["brier"] <= ref_scores["brier"] + BRIER_GUARD + _EPS)


# --- staged selection ---------------------------------------------------------------------------------------


def _row(name: str, stage: str, reference: str | None, scores: dict, delta, adopted) -> dict:
    return {
        "name": name,
        "stage": stage,
        "reference": reference,
        "mae": scores["mae"],
        "early_mae": scores["early_mae"],
        "brier": scores["brier"],
        "n": scores["n"],
        "delta": None if delta is None else [float(x) for x in delta],
        "adopted": adopted,
    }


def select_stage(
    reference: Candidate, candidates: list[Candidate], evaluate: Evaluate, *, stage: str = ""
) -> tuple[Candidate, list[dict]]:
    """Compare each candidate with the reference; the adopted one with the lowest mean delta wins, else the reference.

    `evaluate(cfg, ref_cfg)` returns (candidate scores, reference scores, (delta mean, lo, hi)).
    """
    ref_name, ref_cfg = reference
    rows, adopted = [], []
    for name, cfg in candidates:
        start = time.monotonic()
        cand_scores, ref_scores, delta = evaluate(cfg, ref_cfg)
        ok = adopt(ref_scores, cand_scores, delta)
        rows.append(_row(name, stage, ref_name, cand_scores, delta, ok))
        log.info(
            "%s (stage %s vs %s): MAE %.3f, Brier %.4f, delta %+.3f [%+.3f, %+.3f] -> %s (%.0fs)",
            name,
            stage,
            ref_name,
            cand_scores["mae"],
            cand_scores["brier"],
            *delta,
            "adopted" if ok else "dropped",
            time.monotonic() - start,
        )
        if ok:
            adopted.append((delta[0], name, cfg))
    if not adopted:
        return reference, rows
    _, name, cfg = min(adopted, key=lambda a: a[0])
    return (name, cfg), rows


def _extend(cfg: ModelConfig, features=(), metrics=(), half_life=None) -> ModelConfig:
    return replace(
        cfg,
        features=cfg.features + tuple(f for f in features if f not in cfg.features),
        metrics=cfg.metrics + tuple(m for m in metrics if m not in cfg.metrics),
        half_life=cfg.half_life if half_life is None else half_life,
    )


F_ADDITIONS: dict[str, dict] = {
    "F1": {"features": ("d_elo",)},
    "F2": {"features": ("d_off_expl", "d_def_expl"), "metrics": ("expl",)},
    "F3": {
        "features": ("d_off_rush_epa", "d_def_rush_epa", "d_off_pass_epa", "d_def_pass_epa"),
        "metrics": ("rush_epa", "pass_epa"),
    },
    "F4a": {"half_life": 4.0},
    "F4b": {"half_life": 8.0},
}


def _union(reference: ModelConfig, adopted: list[str], mae_of: dict[str, float]) -> ModelConfig:
    """The reference plus every adopted addition; of competing half-lives, the lower-MAE one."""
    cfg = reference
    half_lives = [n for n in adopted if "half_life" in F_ADDITIONS[n]]
    for name in adopted:
        if name in half_lives and name != min(half_lives, key=lambda n: mae_of[n]):
            continue
        cfg = _extend(cfg, **F_ADDITIONS[name])
    return cfg


def _check_training_seasons(con: duckdb.DuckDBPyConnection) -> None:
    first = con.execute(
        "SELECT MIN(season) FROM games WHERE completed OR (home_points IS NOT NULL AND away_points IS NOT NULL)"
    ).fetchone()[0]
    start = ModelConfig().train_from if first is None else max(int(first), ModelConfig().train_from)
    n = TEST_SEASONS[0] - start
    if first is None or n < MIN_TRAIN_SEASONS:
        raise ValueError(
            f"psu experiment needs at least {MIN_TRAIN_SEASONS} training seasons before {TEST_SEASONS[0]}, "
            f"counted from max(first completed season {first}, train_from {ModelConfig().train_from}); "
            f"have {max(n, 0)}. Ingest earlier seasons with `psu ingest`."
        )


def _build_frame(inputs: FeatureInputs, half_life: float | None) -> pd.DataFrame:
    return game_features(
        inputs.enriched,
        inputs.games,
        inputs.lines,
        inputs.sp,
        inputs.talent,
        alpha=config.TRAIN_ALPHA,
        shrink_plays=config.SHRINK_PLAYS,
        metrics=ALL_METRICS,
        half_life=half_life,
    )


def config_dict(cfg: ModelConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg)))


def run(con: duckdb.DuckDBPyConnection, *, out_dir: Path) -> dict:
    """Stages D (data window), F (features) and M (model); writes reports/experiments.{md,json}."""
    _check_training_seasons(con)
    inputs = load_feature_inputs(con)
    frames: dict[float | None, pd.DataFrame] = {}
    predictions: dict[str, pd.DataFrame] = {}

    def frame_for(half_life: float | None) -> pd.DataFrame:
        if half_life not in frames:
            start = time.monotonic()
            frames[half_life] = _build_frame(inputs, half_life)
            log.info("features (half-life %s) built in %.0fs", half_life, time.monotonic() - start)
        return frames[half_life]

    def predict(cfg: ModelConfig) -> pd.DataFrame:
        key = repr(cfg)
        if key not in predictions:
            predictions[key] = walk_forward(frame_for(cfg.half_life), cfg)
        return predictions[key]

    def evaluate(cfg: ModelConfig, ref: ModelConfig):
        cand, base = predict(cfg), predict(ref)
        return score(cand), score(base), paired_delta(base, cand)

    b0 = ModelConfig()
    start = time.monotonic()
    rows = [_row("B0", "B", None, score(predict(b0)), None, None)]
    log.info("B0 (baseline): MAE %.3f (%.0fs)", rows[0]["mae"], time.monotonic() - start)

    d_cands = [
        ("D0", replace(b0, train_from=2022)),
        ("D1", replace(b0, season_weights=((2020, 0.0),))),
        ("D2", replace(b0, season_weights=((2020, 0.5),))),
    ]
    best, stage_rows = select_stage(("B0", b0), d_cands, evaluate, stage="D")
    rows += stage_rows

    ref_name, ref_cfg = best
    f_cands = [(name, _extend(ref_cfg, **add)) for name, add in F_ADDITIONS.items()]
    best, stage_rows = select_stage(best, f_cands, evaluate, stage="F")
    rows += stage_rows
    adopted = [r["name"] for r in stage_rows if r["adopted"]]
    if len(adopted) >= 2:
        union = _union(ref_cfg, adopted, {r["name"]: r["mae"] for r in stage_rows})
        if union != best[1]:
            best, union_rows = select_stage(best, [("F-union", union)], evaluate, stage="F")
            rows += union_rows

    m_cands = [
        (name, replace(best[1], model=model, tune=True, params=None))
        for name, model in (("M1", "linear"), ("M2", "xgboost"), ("M3", "ensemble"))
    ]
    best, stage_rows = select_stage(best, m_cands, evaluate, stage="M")
    rows += stage_rows

    report = {
        "test_seasons": list(TEST_SEASONS),
        "candidates": rows,
        "final": {"name": best[0], "config": config_dict(best[1])},
    }
    reports = Path(out_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "experiments.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "experiments.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def _fmt(value, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def report_markdown(report: dict) -> str:
    seasons = ", ".join(str(s) for s in report["test_seasons"])
    lines = [
        "# Model experiments",
        "",
        f"Walk-forward test seasons: {seasons}. Delta MAE is candidate minus reference (negative is better), "
        "with a 90% paired-bootstrap interval. A candidate is adopted when delta MAE <= "
        f"{ADOPT_DELTA} and its Brier is at most {BRIER_GUARD} worse than the reference's.",
        "",
        "| Candidate | Stage | Reference | N | MAE | Early MAE | Brier | Delta MAE [90% CI] | Adopted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["candidates"]:
        delta = "-" if r["delta"] is None else f"{r['delta'][0]:+.3f} [{r['delta'][1]:+.3f}, {r['delta'][2]:+.3f}]"
        adopted = "baseline" if r["adopted"] is None else ("yes" if r["adopted"] else "no")
        lines.append(
            f"| {r['name']} | {r['stage']} | {r['reference'] or '-'} | {r['n']} | {_fmt(r['mae'], 3)} | "
            f"{_fmt(r['early_mae'], 3)} | {_fmt(r['brier'], 4)} | {delta} | {adopted} |"
        )
    final = report["final"]
    lines += ["", f"Final config: {final['name']}", "", "```json", json.dumps(final["config"], indent=2), "```"]
    return "\n".join(lines) + "\n"
