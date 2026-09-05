# Flight Fare Intelligence System

An explainable airfare decision-intelligence platform built on **300,153 flight records**. The project progresses from leakage-aware large-tabular regression through uncertainty estimation, contextual fare scoring, route intelligence, SHAP explainability, counterfactual simulation, FastAPI inference, and a Streamlit decision-support dashboard.

## Project status

- ✅ Phase 1 — Engineering Foundation, Legacy Audit & Data Contract
- ✅ Phase 2 — Flight Market Intelligence, EDA & Leakage-Safe Splitting
- ⏳ Phase 3 — Reproducible Baseline Regression System
- ⏳ Phase 4 — Large-Scale Tree Model Benchmarking
- ⏳ Phase 5 — Model Reliability, Segment Error Analysis & Robustness
- ⏳ Phase 6 — Explainable Fare Engine
- ⏳ Phase 7 — Fare Intelligence, Uncertainty & Decision Engine
- ⏳ Phase 8 — Production ML & Analytics API
- ⏳ Phase 9 — Interactive Dashboard & Engineering Hardening
- ⏳ Phase 10 — Final Results, Architecture & Portfolio Story

## Dataset

The canonical dataset contains 300,153 records spanning six airlines, six cities, 30 directed routes, Economy and Business cabin classes, trip duration, stops, departure/arrival periods, and booking horizons from 1–49 days before departure.

The raw CSV is intentionally ignored by Git. Place it at:

```text
data/raw/Flight_Booking.csv
```

## Production feature contract

The deployed fare model will use exactly:

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

`flight` and `arrival_time` remain available for analysis only. `Unnamed: 0` is an export index and is ignored by the model.

## Phase 2 evaluation design

Phase 2 discovered 10,434 rows that repeat an already-observed deployed feature vector. A naive row-random split can therefore place the exact same production-input scenario in both training and evaluation.

The project instead uses a **scenario-grouped, stratified 70/15/15 split**:

- identical deployed feature vectors remain together;
- stratification preserves directed route × cabin class × booking-horizon coverage;
- booking-horizon buckets are 1–7, 8–14, 15–21, 22–35, and 36–49 days;
- random seed is 42;
- scenario overlap across train/validation/test must be exactly zero.

See [`PHASE_2.md`](PHASE_2.md) for the canonical findings and split rationale.

## Run

Create a Python 3.11 virtual environment and install the project:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

Validate Phase 1:

```bash
make phase1
```

Run Phase 2 analytics and split generation:

```bash
make phase2
```

Phase 2 produces local, reproducible outputs under:

```text
reports/metrics/
reports/figures/
data/processed/
```

Generated data and figures are intentionally ignored by Git; the code that reproduces them is version-controlled.

## Planned model progression

```text
Linear Regression → Random Forest → XGBoost → CatBoost
```

The champion model will be selected using predictive quality plus training cost, inference latency, model size, segment reliability, and robustness—not RMSE alone.

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

The system is designed as historical/model-based decision support and will not claim access to live airline inventory or guaranteed future fare movements.
