# Flight Fare Intelligence System

Production-oriented machine-learning system for flight fare prediction and travel-price intelligence using 300,153 historical booking records.

> Current status: **Phase 1 - repository foundation and strict data contract**.

## Portfolio objective

This rebuild turns a notebook-style flight-price regression exercise into an end-to-end ML system with reproducible data validation, benchmarked regression models, route/fare analytics, SHAP explanations, a FastAPI prediction service, and a Streamlit what-if application.

## Dataset snapshot

- 300,153 flight records
- 6 airlines
- 6 source cities and 6 destination cities
- 30 observed directed routes
- 2 cabin classes: Economy and Business
- booking horizon: 1-49 days before departure
- ticket price range: INR 1,105-INR 123,071
- no missing values in the supplied dataset

## Deployment feature contract

The production model will accept exactly these user-facing features:

1. `airline`
2. `source_city`
3. `destination_city`
4. `departure_time`
5. `stops`
6. `class`
7. `duration`
8. `days_left`

Target: `price`.

`flight` and `arrival_time` are retained for analytics but are not part of the deployment input contract. `Unnamed: 0` is an export index and is never a model feature.

## Planned model progression

1. Linear Regression - interpretable baseline
2. Random Forest Regressor
3. XGBoost Regressor
4. CatBoost Regressor

Primary metric: **RMSE**. Secondary metrics: **MAE, R2, and MAPE**, with subgroup error analysis by class, airline, route, stops, and booking horizon.

## Planned intelligence layer

Beyond point prediction, the final system will provide:

- SHAP explanations for individual fare predictions
- route-level fare analytics
- price-vs-days-left analysis
- scenario/what-if fare prediction
- FastAPI inference endpoints
- Streamlit frontend
- Docker packaging and CI tests

## Legacy baseline

The original Intellipaat reference reported approximately:

- Linear Regression RMSE: INR 7,259.93; MAPE ~34%
- Decision Tree RMSE: INR 3,620; MAPE ~7.7%
- Random Forest RMSE: INR 2,824; MAPE ~7.3%

These figures are historical references only. The rebuilt project will generate its own leakage-safe held-out benchmarks using a reproducible pipeline.

## Repository layout

```text
api/                       FastAPI layer
configs/                   project/model configuration
data/{raw,interim,processed}/
docs/assets/               README/application assets
legacy/                    original coursework/reference material
models/                    trained artifacts (not committed)
notebooks/                 analysis-only notebooks
reports/{figures,metrics,predictions}/
scripts/                   CLI entry points
src/flight_fare_intelligence/ reusable package code
tests/                     automated tests
.github/workflows/          CI
```

## Phase 1 setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python scripts/validate_data.py data/raw/Flight_Booking.csv --strict
pytest
ruff check src scripts tests
```

Or run:

```bash
make phase1
```

## Engineering principles

- no model logic hidden only in notebooks
- deterministic/config-driven workflows
- strict schema checks before training
- train/validation/test evaluation with no preprocessing leakage
- saved preprocessing + model artifacts as one inference contract
- API/UI input schema aligned with training features
- automated tests before every phase commit
