# Flight Fare Intelligence System

An explainable airfare decision-intelligence platform built on **300,153 flight records**. The project progresses from leakage-aware large-tabular regression through reliability analysis, explainability, uncertainty estimation, contextual fare scoring, route intelligence, counterfactual simulation, FastAPI inference, and a Streamlit decision-support dashboard.

## Project status

- ✅ Phase 1 — Engineering Foundation, Legacy Audit & Data Contract
- ✅ Phase 2 — Flight Market Intelligence, EDA & Leakage-Safe Splitting
- ✅ Phase 3 — Reproducible Baseline Regression System
- ✅ Phase 4 — Large-Scale Tree Model Benchmarking
- ✅ Phase 5 — Model Reliability, Segment Error Analysis & Robustness
- ✅ Phase 6 — Explainable Fare Engine
- ✅ Phase 7 — Fare Intelligence, Uncertainty & Decision Engine
- ⏳ Phase 8 — Production ML & Analytics API
- ⏳ Phase 9 — Interactive Dashboard & Engineering Hardening
- ⏳ Phase 10 — Final Results, Architecture & Portfolio Story

## Dataset

The canonical dataset contains **300,153 records** spanning six airlines, six cities, 30 directed routes, Economy and Business cabin classes, trip duration, stops, departure/arrival periods, and booking horizons from 1–49 days before departure.

The raw CSV is intentionally ignored by Git. Place it at:

```text
data/raw/Flight_Booking.csv
```

## Production feature contract

The deployed fare model uses exactly:

```text
airline
source_city
destination_city
departure_time
stops
class
duration
days_left
```

Target: `price`

`flight` and `arrival_time` remain analysis-only. `Unnamed: 0` is an export index and is ignored by the model.

## Leakage-safe evaluation design

Phase 2 found **10,434 rows** that repeat an already-observed deployed feature vector. A conventional row-random split can therefore place the exact same booking scenario in both training and evaluation.

The project instead uses a **scenario-grouped, stratified 70/15/15 split**:

- Train: 210,190 rows
- Validation: 44,954 rows
- Test: 45,009 rows
- Exact deployed-scenario overlap across splits: **0**

All later phases reuse these assignments. The test set remains sealed through the model-development and reliability phases.

## Model progression

```text
Linear Regression → Random Forest → XGBoost → CatBoost
```

### Phase 3 baseline

Linear Regression established the deployment-aligned baseline at approximately:

```text
Validation RMSE: ₹6,744
Validation MAE:  ₹4,554
Overall R²:      0.9116
```

However, 5.94% of its validation predictions were negative fares and within-class R² dropped sharply, establishing the need for nonlinear modeling.

### Phase 4 model benchmark

The locked Phase 4 run produced:

| Model | RMSE | MAE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| **XGBoost** | **₹2,677.55** | ₹1,401.91 | **0.9861** | 10.39% |
| Random Forest | ₹2,815.82 | **₹1,366.27** | 0.9846 | **9.76%** |
| CatBoost | ₹4,227.89 | ₹2,458.36 | 0.9653 | 17.35% |
| Linear Regression | ₹6,744.07 | ₹4,554.28 | 0.9116 | 46.25% |

XGBoost is the Phase 4 champion under the RMSE-first selection policy, with Random Forest retained as the strongest robustness comparator.

## Phase 5 reliability findings

Phase 5 kept XGBoost locked and showed that the excellent aggregate **0.9861 R²** is not uniformly reliable. Business fares and last-minute bookings carry materially larger absolute error, while the extreme ₹80k+ tail is systematically underpredicted. Raw training-context frequency was also shown to be insufficient as a standalone confidence proxy.

See [`PHASE_5.md`](PHASE_5.md) for the full reliability methodology and diagnostics.

## Phase 6 explainability

Phase 6 adds an additive SHAP explanation layer without retuning the champion or touching the sealed test split. It explains the full validation set, aggregates one-hot SHAP values back to the eight deployed inputs, compares model drivers across Economy, Business, last-minute, and high-fare segments, and generates exact local explanations for deterministic representative cases.

The locked run confirmed that cabin class dominates the global model explanation, while booking horizon becomes substantially more influential for last-minute Economy cases and duration/airline effects become more prominent within Business pricing. SHAP reconstruction remains within sub-rupee numerical tolerance.

See [`PHASE_6.md`](PHASE_6.md) for the methodology and generated artifacts.

## Phase 7 fare intelligence and uncertainty

Phase 7 turns the point estimator into a decision-intelligence engine. The existing validation split is scenario-preservingly divided into an uncertainty calibration subset and a separate uncertainty-evaluation subset while all **45,009 test rows remain sealed**.

The engine adds:

- tail-aware asymmetric conformal prediction intervals;
- an explicit conservative high-fare Business guardrail motivated by Phase 5;
- Fare Opportunity Scores based on training-only route/class/booking-horizon comparables;
- comparative reliability scoring based on relative interval width rather than raw support count;
- route and exact booking-horizon market intelligence;
- one-feature-at-a-time counterfactual simulation; and
- cautious `BUY_NOW`, `MONITOR`, or `WAIT_OR_MONITOR` model-based guidance.

A reference execution achieved roughly **90.5% empirical coverage** for nominal 90% intervals while bringing the known ₹80k+ high-fare region to roughly **91% coverage** through deliberately wider risk-aware intervals. The guidance layer is explicitly historical/model-based and does not claim access to live airline inventory or guaranteed future fare movement.

See [`PHASE_7.md`](PHASE_7.md) for the full uncertainty, scoring, and decision policy.

## Run

Create a Python 3.11 or 3.12 virtual environment and install the project:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

Run individual phase gates:

```bash
make phase1
make phase2
make phase3
make phase4
make phase5
make phase6
make phase7
```

`make phase7` expects the local Phase 4 champion and Phase 6 explainability report to exist. If generated artifacts were removed, rerun the earlier phase gates first.

Generated data, metrics, predictions, figures, and model binaries are intentionally ignored by Git; the code that reproduces them is version-controlled.

## Final product direction

The completed system will provide:

- fare prediction with uncertainty bounds;
- Fare Opportunity Score and contextual cheap/fair/expensive positioning;
- route and booking-horizon intelligence;
- Buy Now / Wait model-based guidance;
- reliability scoring;
- counterfactual what-if simulation;
- SHAP explanations;
- FastAPI single and batch inference;
- Streamlit decision-support dashboard;
- Docker, CI, testing, and lightweight monitoring.

The system is historical/model-based decision support and will not claim access to live airline inventory or guaranteed future fare movements.
