# Model v2: more history, an experiment harness, and earned upgrades

This is a follow-up to the model upgrade (PR #8), requested as "go ham on the predictive models".

## Goal and success criteria

The goal is better pregame game predictions: lower margin MAE and lower Brier score on held-out seasons, with every change backed by a measured gain.

Success means:
- A `psu experiment` command that scores any model configuration with a multi-season walk-forward backtest, and reports each candidate's gain over the baseline with a confidence interval.
- A production model (`psu train`) that uses the best configuration the harness found.
- A PR that reports honest before-and-after numbers, including the candidates that did not help.

Constraints:
- **No Vegas line as a feature.** The model must stay independent, so "model vs Vegas" keeps its meaning.
- **No new dependencies.** Use Ridge (scikit-learn) and XGBoost only; a new compiled package risks Windows Smart App Control blocks.
- **Pregame information only.** Every feature for a game uses only information from before its slate.
- **Refresh stays manual.**
- **Seasons now start in 2019** (`config.FIRST_SEASON = 2019`). 2020 is the short COVID season.

## 1. Data

- 2019–2021 are ingested with the existing `psu ingest` (about 180 API calls, approved).
- `psu build` then computes metrics for them.
- The first season with a prior is 2020, so the `no_prior` split now applies only to 2019.

## 2. Experiment harness (`psu experiment`)

New module `src/psu/experiment.py`, plus a CLI command.

### Walk-forward
- The test seasons are **2023, 2024 and 2025**, about 2,400 games.
- For test season T, a candidate trains on its training seasons before T.
- The default training seasons are every season from 2020 (the first with a prior) up to T−1.
- The final model is fitted on all of those, then scores every completed FBS-vs-FBS game in T.

### Metrics
Each is computed per candidate over all test games, and separately for the early season (regular-season slates 1–4):
- **MAE** of the predicted margin.
- **Brier score**, with win probability = NormalCDF(margin / σ). σ is the pooled out-of-fold residual SD from the candidate's training fit (as `GameModel.fit` does today), so the comparison is like for like.

### Comparison
- Paired bootstrap over test games (2,000 resamples, fixed seed) gives the mean ΔMAE against the reference and a 90% interval.
- The reference is the current best configuration (see Selection).

### Output
- `data/reports/experiments.md`: a table with one row per candidate, showing MAE, early MAE, Brier, and ΔMAE [90% CI] against its reference, plus whether it was adopted.
- `data/reports/experiments.json`: the same content as JSON.

### Configuration object
`ModelConfig` is a frozen dataclass with these fields:
- `features: tuple[str, ...]`
- `model: str`: `"linear"`, `"xgboost"` or `"ensemble"`
- `params: dict`: Ridge alpha, or XGBoost hyperparameters
- `train_from: int`: default 2020
- `season_weights: dict[int, float]`: for example `{2020: 0.5}`; a weight of 0 drops the season
- `recency_half_life: float | None`: in slates; see F4

The rating parameters (`alpha` and `shrink_plays`) stay at the tuned `config.TRAIN_ALPHA` and `config.SHRINK_PLAYS`.

### Feature cache
Rolling ratings are expensive, so the harness computes features once per distinct feature-engineering setting (for example each recency half-life) and reuses them across model candidates.

## 3. Candidates

Candidates are evaluated in stages, and each stage's winner becomes the reference for the next (greedy forward selection).

**Baseline B0:** today's features and model (the current `FEATURES`, with Ridge and XGBoost chosen on validation MAE as today), trained from 2020.

**Stage D, data window:**
- D0: train only from 2022 (what the old data window allowed).
- D1: drop 2020.
- D2: weight 2020 at 0.5.
- D0 answers "did the extra history help?", with B0 as its reference.

**Stage F, features.** Each is added to the reference on its own, then winners are combined.
- **F1:** `d_elo`, the home minus away CFBD pregame Elo (`games.home_pregame_elo` / `away_pregame_elo`).
- **F2:** rolling opponent-adjusted explosive-play rate for offense and defense, giving `d_off_expl` and `d_def_expl`. It is computed like the existing rolling ratings, with `explosive` as the adjusted value.
- **F3:** rolling opponent-adjusted rush and pass EPA for offense and defense, giving `d_off_rush_epa`, `d_off_pass_epa`, `d_def_rush_epa` and `d_def_pass_epa`. Plays are split by `play_class`.
- **F4:** recency weighting within a season. Plays get the weight 0.5^((slate_now − slate_play) / H), for H ∈ {4, 8} slates. This needs a `weights` argument on `opponent_adjust`, applied as the ridge fit's sample weights and in the play counts that drive shrinkage.

**Stage M, model:**
- **M1:** Ridge with alpha chosen from {0.3, 1, 3, 10} by walk-forward inside the training seasons.
- **M2:** XGBoost tuned over n_estimators {300, 600}, max_depth {2, 3, 4}, learning_rate {0.03, 0.06} and min_child_weight {1, 5}, chosen the same way.
- **M3:** an ensemble averaging the tuned Ridge and the tuned XGBoost margins.

### Selection rule
A candidate is adopted when both of these hold:
- Its mean ΔMAE against the reference is ≤ −0.02.
- Its Brier score is no worse than the reference's by more than 0.0005.

Otherwise it is reported and dropped. Stage F runs each feature alone first. It then tries the union of the individually adopted features, and adopts the union only if it beats the best single adopted feature by the same rule.

## 4. Production

- The adopted configuration becomes `config.MODEL_CONFIG`.
- `psu train` builds features for that configuration and fits the chosen model. `backtest()` keeps its validation and test report (now with more seasons), plus phase sigmas and calibration as before.
- `game_predictions`, the simulator, backfill and the dashboard keep their interfaces.
  - Backfill must build the same features as the production configuration, including Elo and recency.
  - The saved-model feature-count check keeps catching stale models.
- The README's model section and its "2025 test season" table are updated from the real run.

## Error handling
- A missing Elo value is left as NaN for the median imputer.
- `psu experiment` refuses to run with fewer than two training seasons before the first test season, and says why.
- Unknown feature names in a `ModelConfig` raise `ValueError` before any fitting.

## Testing
All tests use synthetic data; none call the API.
- **Walk-forward:** training never includes the test season or later; season weights of 0 drop that season.
- **Bootstrap:** a known difference is recovered, and the same seed gives the same interval.
- **Selection rule:** adopt and reject cases, including the Brier guard.
- **Features:**
  - `d_elo` is home minus away.
  - F2 and F3 rolling values never use plays from the game's own slate or later, the same guarantee as the existing ratings.
  - With H = ∞, recency weighting reproduces the unweighted ratings exactly.
- **Weighted opponent adjustment:** all-equal weights match the unweighted fit.
- **Tuning:** inner selection never sees the outer test season.
- **Production:** `psu train` with a non-default `MODEL_CONFIG` writes predictions, and backfill builds matching features.
