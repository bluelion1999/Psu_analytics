"""A game-model configuration: features, model kind, hyperparameters, training window and season weights.

Kept light (imports only psu.features) so psu.config can import it.
"""

from __future__ import annotations

from dataclasses import dataclass

from psu.features import ALL_FEATURES, DEFAULT_METRICS, FEATURES, METRICS

FIRST_SEASON = 2019  # 2019 on (2020 is the COVID season); psu.config re-exports this for the rest of the project
MODELS = ("select", "linear", "xgboost", "ensemble")
# d_off_<m> / d_def_<m> -> the metric it needs
_FEATURE_METRIC = {f"d_{side}_{m}": m for m in METRICS for side in ("off", "def")}


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[str, ...] = tuple(FEATURES)
    model: str = "select"  # "select" = today's rule: linear vs xgboost on validation MAE
    params: dict | None = None  # None = defaults; "tuned" handled by experiment/train via tune()
    tune: bool = False
    train_from: int = 2020
    season_weights: tuple[tuple[int, float], ...] = ()  # e.g. ((2020, 0.5),); weight 0 drops a season
    metrics: tuple[str, ...] = DEFAULT_METRICS
    half_life: float | None = None

    def validate(self) -> None:
        unknown = [f for f in self.features if f not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"Unknown feature(s) {unknown}; choose from {ALL_FEATURES}")
        bad_metrics = [m for m in self.metrics if m not in METRICS]
        if bad_metrics:
            raise ValueError(f"Unknown metric(s) {bad_metrics}; choose from {list(METRICS)}")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        missing = sorted({_FEATURE_METRIC[f] for f in self.features if f in _FEATURE_METRIC} - set(self.metrics))
        if missing:
            raise ValueError(f"Feature(s) need metric(s) {missing} in metrics {self.metrics}")
        if not self.features:
            raise ValueError("A model config needs at least one feature")
        if any(w < 0 for _, w in self.season_weights):
            raise ValueError(f"Season weights must be >= 0, got {self.season_weights}")
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError(f"half_life must be > 0 or None, got {self.half_life}")
        if self.train_from <= FIRST_SEASON:
            raise ValueError(
                f"train_from must be after the first ingested season {FIRST_SEASON}, got {self.train_from}"
            )

    def weight_of(self, season: int) -> float:
        return float(dict(self.season_weights).get(int(season), 1.0))
