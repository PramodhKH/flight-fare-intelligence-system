# Phase 4 — Large-Scale Tree Model Benchmarking

## Objective

Benchmark Random Forest, XGBoost, and CatBoost against the Phase 3 Linear Regression baseline using the **same leakage-safe Phase 2 train/validation split**, while keeping the 45,009-row test set sealed.

This phase evaluates not only predictive accuracy, but also training cost, batch inference latency, and serialized model size so the champion is defensible from both ML and production perspectives.

## Evaluation policy

- **Train:** 210,190 rows — used to fit every candidate.
- **Validation:** 44,954 rows — used for bounded hyperparameter selection and the official Phase 4 benchmark.
- **Test:** 45,009 rows — remains sealed and is not scored.

No Phase 4 modeling decision is informed by the held-out test set.

## Why model families run in separate processes

Random Forest, XGBoost, and CatBoost use different native runtimes and threading implementations. The benchmark therefore trains each family in its own Python process. This design:

- prevents cross-library native-thread/runtime interference;
- releases each family’s peak memory before the next benchmark starts;
- makes the workflow more stable on a local laptop;
- preserves a single `make phase4` entry point.

This is an engineering decision, not a modeling shortcut. Every family still sees the exact same train and validation records.

## Bounded candidate search

Phase 4 deliberately uses a small, reproducible candidate search rather than an expensive broad sweep.

### Random Forest

Two candidates vary tree count, depth, sampling, and feature subsampling. The selected configuration is:

```text
n_estimators=55
max_depth=20
min_samples_leaf=1
max_features=0.70
max_samples=0.60
```

### XGBoost

Two histogram-tree candidates vary estimator count, depth, and learning rate. The selected configuration is:

```text
n_estimators=650
max_depth=9
learning_rate=0.045
min_child_weight=2
subsample=0.90
colsample_bytree=0.95
reg_lambda=3.0
```

### CatBoost

CatBoost uses the six categorical deployment features natively rather than one-hot encoding them. The selected bounded candidate is:

```text
iterations=180
depth=8
learning_rate=0.10
l2_leaf_reg=5.0
```

The search is intentionally bounded because Phase 4 is a model-family benchmark, not an unlimited hyperparameter-optimization exercise. More tuning is justified only if later reliability analysis shows a reason to revisit the model family decision.

## Canonical validation results

| Model | RMSE | MAE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| **XGBoost** | **₹2,677.55** | ₹1,401.91 | **0.9861** | 10.39% |
| Random Forest | ₹2,815.82 | **₹1,366.27** | 0.9846 | **9.76%** |
| CatBoost | ₹4,227.89 | ₹2,458.36 | 0.9653 | 17.35% |
| Linear Regression | ₹6,744.07 | ₹4,554.28 | 0.9116 | 46.25% |

XGBoost reduces validation RMSE by approximately **60.3% versus Linear Regression** and **4.9% versus the best Random Forest candidate**.

Random Forest has slightly better MAE and MAPE than XGBoost, which is useful evidence that model selection cannot be reduced to one universal metric. The project keeps RMSE as the primary metric because large fare misses are especially costly for the later fare-intelligence layer.

## Production tradeoffs

Reference-run system measurements were:

| Model | Fit time | Median 5k-row batch latency | Serialized size |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.18 s | 4.51 ms | 0.005 MiB |
| Random Forest | 2.20 s | 29.45 ms | 169.62 MiB |
| XGBoost | 2.30 s | 11.18 ms | 16.66 MiB |
| CatBoost | 3.20 s | **2.03 ms** | **0.67 MiB** |

Timing numbers are machine-dependent and should be regenerated locally. Model-size differences are still useful architectural signals: Random Forest is substantially larger than XGBoost, while CatBoost is extremely compact and fast in this bounded configuration but materially less accurate.

## Champion selection

The deterministic policy is:

1. lowest validation RMSE;
2. MAE as the first tie-breaker;
3. median 5,000-row batch latency;
4. serialized model size.

Under that policy, the Phase 4 champion is:

> **XGBoost**

The champion is copied to:

```text
models/phase4_champion.joblib
```

Phase 5 will challenge this choice with segment-level reliability and robustness analysis before the model is treated as production-ready.

## Generated outputs

Running `make phase4` reproduces:

```text
models/phase4_random_forest.joblib
models/phase4_xgboost.joblib
models/phase4_catboost.joblib
models/phase4_champion.joblib

reports/metrics/phase4_random_forest_benchmark.json
reports/metrics/phase4_xgboost_benchmark.json
reports/metrics/phase4_catboost_benchmark.json
reports/metrics/phase4_model_benchmark.json

reports/predictions/phase4_random_forest_validation.csv
reports/predictions/phase4_xgboost_validation.csv
reports/predictions/phase4_catboost_validation.csv
reports/predictions/phase4_model_comparison.csv
reports/predictions/phase4_validation_predictions.csv

reports/figures/phase4_validation_rmse.png
reports/figures/phase4_accuracy_vs_latency.png
reports/figures/phase4_champion_actual_vs_predicted.png
```

Generated artifacts remain ignored by Git. Source code, tests, configuration, phase documentation, and reproducibility settings are version-controlled.

## IDE and environment reliability

Phase 4 adds an explicit environment check for:

- scikit-learn;
- XGBoost;
- CatBoost;
- NumPy;
- pandas;
- the active Python interpreter.

Tracked VS Code workspace settings point Pylance at:

```text
${workspaceFolder}/.venv/bin/python
```

and add `${workspaceFolder}/src` to the analysis path. This directly addresses the earlier `sklearn.model_selection could not be resolved from source` Pylance warning when VS Code was using a different interpreter than the terminal virtual environment.

## Validation gate

Phase 4 is complete only when:

1. the Phase 4 environment/import check passes;
2. strict raw-data validation passes;
3. Phase 2 split assignments are reproduced;
4. Phase 3 baseline is reproduced;
5. all three tree families complete their bounded candidate search;
6. the test set remains sealed;
7. the comparison report selects a deterministic champion;
8. all automated tests pass; and
9. Ruff reports no lint violations.
