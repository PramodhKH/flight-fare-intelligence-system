# Phase 8 — Production ML & Analytics API

## Objective

Expose the locked XGBoost model and Phase 7 intelligence bundle through a versioned FastAPI service without retraining the model or scoring the sealed test set.

Phase 8 turns the offline ML system into a reusable application service for Phase 9's Streamlit frontend and for direct API clients.

## Production artifacts

The API loads two generated local artifacts:

```text
models/phase4_champion.joblib
models/phase7_intelligence_bundle.joblib
```

The first contains the deployment-aligned XGBoost pipeline. The second contains the hierarchical conformal calibrator, training-only comparable-fare index, reliability reference distribution, and calibration metadata.

Model artifacts remain ignored by Git. A fresh clone must reproduce Phases 4 and 7 before starting the API.

## API contract

The service is versioned under `/v1` and exposes:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/v1/health` | readiness and sealed-test status |
| GET | `/v1/model` | champion, feature contract, uncertainty metadata |
| GET | `/v1/telemetry` | bounded in-memory request/latency telemetry added in Phase 9 |
| POST | `/v1/predict` | complete fare-intelligence prediction |
| POST | `/v1/predict/batch` | bounded batch inference, maximum 100 scenarios |
| POST | `/v1/what-if` | one-feature-at-a-time counterfactual analysis |
| POST | `/v1/booking-horizon` | 1–49 day model curve plus historical context |
| GET | `/v1/routes` | supported directed routes |
| GET | `/v1/route-analytics` | training-only historical route/class summaries |

OpenAPI documentation is available at `/docs`.

## Prediction response

A single prediction returns the production feature scenario plus:

- XGBoost point estimate;
- hierarchical 90% prediction interval;
- interval calibration level and support rows;
- Fare Opportunity Score and contextual fare position;
- comparative reliability score;
- historical/model-based booking guidance;
- raw-feature SHAP explanation; and
- explicit limitations/warnings.

The API deliberately separates explainability from reliability. SHAP explains why the model produced its estimate, while conformal intervals and the reliability score communicate uncertainty.

## Batch inference

Batch requests accept at most 100 scenarios. To keep batch inference practical, booking-guidance curves are omitted and SHAP explanations are disabled by default. Clients can explicitly request per-row explanations when needed.

## Input validation

Pydantic enforces the eight deployed features and canonical categorical values. It also validates:

- source and destination are different;
- `days_left` is between 1 and 49;
- duration is positive and at most 50 hours;
- counterfactual scenarios remain valid after the changed value is applied; and
- batch size remains bounded.

The external JSON contract uses the key `class`, while the Python schema safely stores it internally as `cabin_class`.

## What-if and booking-horizon semantics

Counterfactual endpoints vary one deployed model input while holding all others fixed. They are not causal estimates and are not guaranteed future-price forecasts.

The booking-horizon endpoint therefore returns both:

1. the raw XGBoost 1–49 day counterfactual curve; and
2. the historical median for the corresponding route/class booking-horizon bucket.

This is intentional because Phase 7 showed that tree-model counterfactual curves may contain sharp split-driven discontinuities.

## Observability and failure behavior

Every HTTP response receives:

```text
X-Request-ID
X-Process-Time-Ms
```

Requests are logged with method, path, response status, latency, and request ID. The production application loads model artifacts during FastAPI lifespan startup so a missing or corrupt artifact fails fast rather than serving partial predictions.

## Docker

After the Phase 7 artifacts exist locally:

```bash
docker compose up --build
```

The image copies the local generated model artifacts, starts Uvicorn on port 8000, and includes an HTTP health check against `/v1/health`.

## Reproducibility gate

Run:

```bash
make phase8
```

The gate performs:

1. environment/import verification;
2. strict canonical dataset validation;
3. real-artifact API smoke testing through warning-free HTTPX ASGI transport;
4. safe Ruff auto-fixes;
5. Ruff formatting;
6. strict Ruff validation; and
7. the complete automated test suite.

The Phase 8 smoke test exercises all eight API capabilities using the deterministic Phase 7 demo scenario. It does not score any Phase 2 test rows.

## Deployment limitations

The service is based on a static historical dataset. It does not have live inventory, current airline pricing, booking availability, or a true future price feed. `BUY_NOW`, `MONITOR`, and `WAIT_OR_MONITOR` are model-based decision-support labels, not promises about future airline behavior.

Phase 9 uses this API as the backend for the interactive Streamlit Flight Fare Intelligence dashboard.
