# Phase 10 — Final Results, Architecture & Portfolio Release

## Objective

Phase 10 closes the Flight Fare Intelligence System as a recruiter-facing, reproducible ML engineering project. It does not alter the locked XGBoost champion, uncertainty calibrator, feature contract, API behavior, or Streamlit decision logic.

The only model evaluation introduced in this phase is the **first score of the 45,009-row held-out test set**. The test split remained sealed during Phases 2–9 and was never used for model selection, hyperparameter tuning, reliability-policy design, SHAP analysis, uncertainty calibration, API development, or dashboard development.

**No post-test retuning is allowed.**

## Final held-out evaluation

| Metric | Result |
| --- | ---: |
| Test rows | 45,009 |
| RMSE | **₹2,670.07** |
| MAE | **₹1,403.75** |
| R² | **0.9862** |
| MAPE | **10.38%** |
| Median absolute error | ₹640.47 |
| P90 absolute error | ₹3,488.67 |
| 90% interval coverage | **90.28%** |
| Median interval width | ₹3,706.09 |

The validation champion benchmark was ₹2,677.55 RMSE / 0.9861 R², so the held-out test result is closely aligned with validation performance.

## Final segment checks

### Cabin class

- Economy: RMSE **₹1,461.93**, MAE ₹834.71, R² 0.8477.
- Business: RMSE **₹4,261.72**, MAE ₹2,661.41, R² 0.8924.

### Booking horizon

- 1–7 days: RMSE **₹4,426.26**.
- 8–14 days: RMSE ₹2,896.31.
- 15–21 days: RMSE ₹2,616.01.
- 22–35 days: RMSE **₹2,104.97**.
- 36–49 days: RMSE ₹2,223.20.

### High-fare tail

The known validation weakness generalized to unseen test data rather than disappearing:

- ₹80K+ test rows: 325.
- ₹80K+ RMSE: **₹13,047.48**.
- Mean bias: **-₹10,337.38**.
- Underprediction rate: **93.54%**.
- ₹100K+ underprediction rate: **100%** across 18 test records.

This remains a documented limitation and is one reason the product exposes uncertainty rather than returning point predictions as certainty.

## Final uncertainty check

The 90% conformal layer achieved **90.28% empirical coverage** on the untouched test set.

- Business coverage: 90.94%.
- Economy coverage: 89.99%.
- 1–7 day coverage: 89.89%.
- 8–14 day coverage: 90.58%.
- 15–21 day coverage: 90.82%.
- 22–35 day coverage: 90.24%.
- 36–49 day coverage: 90.06%.

This supports the calibration design without implying that every individual interval is guaranteed to contain the actual fare.

## Portfolio release assets

Phase 10 adds tracked recruiter-facing assets under `docs/assets/`:

- dashboard overview;
- validation accuracy-versus-latency benchmark;
- high-fare reliability/bias visualization;
- final held-out actual-vs-predicted visualization;
- final held-out interval-coverage visualization.

It also adds:

- final architecture and results narrative in `README.md`;
- resume, LinkedIn, and GitHub snippets in `docs/portfolio_snippets.md`;
- the Phase 10 held-out evaluation runner;
- Phase 10 documentation tests;
- project release version `1.0.0`.

## Final architecture

The deployed system remains:

```text
300,153 records
    ↓
validation + leakage-safe split
    ↓
LR / RF / XGBoost / CatBoost benchmark
    ↓
XGBoost champion
    ├── SHAP explanation engine
    ├── conformal uncertainty engine
    ├── Fare Opportunity Score
    ├── reliability score
    └── booking-horizon / what-if guidance
            ↓
         FastAPI
            ↓
         Streamlit
```

The 45,009-row test set enters only at the final reporting step after all decisions are frozen.

## Final quality gate

Run:

```bash
make phase10
```

The gate performs:

1. environment verification;
2. strict raw-data validation;
3. Phase 9 real-artifact API/frontend smoke test;
4. final held-out test evaluation;
5. safe Ruff autofix and formatting;
6. strict Ruff validation;
7. full automated regression suite.

Expected result:

```text
72 passed
All checks passed!
```

## Release status

- Model selection: locked.
- Production feature contract: locked.
- XGBoost champion: locked.
- SHAP layer: locked.
- Conformal uncertainty: locked.
- Fare Opportunity Score: locked.
- Reliability policy: locked.
- Booking guidance policy: locked.
- FastAPI contract: locked.
- Streamlit dashboard: locked.
- Final test: first scored in Phase 10.
- **No post-test retuning.**

Phase 10 is documentation, final evaluation, and release hardening only.
