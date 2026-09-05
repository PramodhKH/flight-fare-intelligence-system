# Phase 3 — Reproducible Baseline Regression System

## Objective

Establish the first official modeling benchmark using a deployment-aligned Linear Regression pipeline while preserving the leakage-safe evaluation design created in Phase 2.

## Evaluation policy

Phase 3 deliberately separates development evaluation from the final held-out test set:

- **Train split:** 210,190 rows — used to fit preprocessing and Linear Regression.
- **Validation split:** 44,954 rows — used for the official Phase 3 baseline benchmark and residual diagnostics.
- **Test split:** 45,009 rows — remains sealed and is not scored in Phase 3.

The test set stays untouched during baseline development and Phase 4 model selection. This avoids repeatedly adapting modeling choices to the final evaluation data.

## Production-aligned preprocessing

The model uses exactly the eight deployed features locked in Phase 1:

`airline, source_city, destination_city, departure_time, stops, class, duration, days_left`

Preprocessing is fitted only on the training split:

- categorical features: one-hot encoding with `drop="first"` and `handle_unknown="ignore"`;
- numeric features: standard scaling;
- target: raw ticket price in INR;
- estimator: ordinary least-squares `LinearRegression`.

No `flight`, `arrival_time`, export index, target-derived feature, or test-set statistic enters the pipeline.

## Canonical baseline results

### Overall validation performance

| Metric | Train | Validation |
| --- | ---: | ---: |
| RMSE | ₹6,769.12 | **₹6,744.07** |
| MAE | ₹4,576.45 | **₹4,554.28** |
| R² | 0.9110 | **0.9116** |
| MAPE | 46.37% | **46.25%** |
| Median absolute error | ₹3,128.53 | **₹3,101.11** |
| 90th-percentile absolute error | ₹10,539.56 | **₹10,472.34** |

The close train/validation metrics indicate that the baseline is not materially overfitting. Its main limitation is model form: a single additive linear relationship cannot capture the nonlinear route, cabin-class, booking-horizon, stop, duration, and airline interactions present in the data.

### Why the headline R² is deceptive

The overall R² is high largely because cabin class separates two very different fare scales. Within each class, Linear Regression explains much less of the variation:

| Validation segment | RMSE | MAE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Economy | ₹3,671.34 | ₹2,884.64 | **0.0500** | 58.69% |
| Business | ₹10,775.81 | ₹8,239.35 | **0.3063** | 18.79% |

This is an important modeling finding: aggregate R² alone would make the baseline appear far stronger than it is for within-market fare estimation.

### Residual and production-suitability findings

On the validation split:

- 17.21% of predictions are within ₹1,000 of the observed fare;
- 33.83% are within ₹2,000;
- 71.43% are within ₹5,000;
- 90% of absolute errors are below approximately ₹10,472;
- **2,672 predictions (5.94%) are negative fares**, which are impossible in the product domain.

The negative predictions are a particularly clear reason not to deploy the Linear Regression baseline. They demonstrate a structural limitation of unconstrained linear regression rather than a data-validation issue.

The new validation RMSE is numerically lower than the roughly ₹7,260 Linear Regression RMSE reported in the legacy coursework, but the comparison is not apples-to-apples because the rebuilt system uses a different production feature contract and a scenario-grouped leakage-safe split.

## Why Linear Regression remains useful

The purpose of this model is not to win the project. It creates a simple, reproducible lower bound against which Random Forest, XGBoost, and CatBoost can be measured. Its residual structure provides concrete hypotheses for Phase 4: the stronger models should improve within-class fit, capture nonlinear booking-horizon behavior, reduce tail errors, and eliminate impossible negative-fare behavior.

## Metrics

The canonical comparison metrics are:

- RMSE — primary model-selection metric;
- MAE;
- R²;
- MAPE;
- median absolute error;
- 90th-percentile absolute error.

The last two diagnostics make the typical error and tail error visible rather than relying only on an average.

## Generated outputs

Running `make phase3` reproduces:

- `reports/metrics/phase3_linear_regression_metrics.json`
- `reports/predictions/phase3_linear_regression_validation.csv`
- `reports/predictions/phase3_linear_regression_coefficients.csv`
- `reports/figures/phase3_actual_vs_predicted.png`
- `reports/figures/phase3_residual_distribution.png`
- `reports/figures/phase3_residuals_vs_predicted.png`
- `models/phase3_linear_regression.joblib`

Generated artifacts remain ignored by Git. The code, tests, configuration, and this phase record are version-controlled.

Timing values written by the script are machine-dependent diagnostics and are not treated as portable benchmark results. Phase 4 will compare model latency under the same execution environment.

## Validation gate

Phase 3 is complete only when:

1. strict raw-data validation passes;
2. Phase 2 split assignments are reproduced successfully;
3. exact scenario overlap remains zero;
4. the Linear Regression pipeline trains and evaluates successfully;
5. the test split remains unscored;
6. all automated tests pass; and
7. Ruff reports no lint violations.
