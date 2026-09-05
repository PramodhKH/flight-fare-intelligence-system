# Phase 9 — Interactive Dashboard & Engineering Hardening

## Objective

Turn the Phase 8 production API into the user-facing **Flight Fare Intelligence System** while hardening the local deployment, observability, and developer workflow.

Phase 9 does not retrain XGBoost and does not score the sealed **45,009-row test split**. It consumes the locked FastAPI service and Phase 7 intelligence contract.

## Product architecture

```text
Browser
   ↓
Streamlit dashboard :8501
   ↓ HTTP /v1
FastAPI service :8000
   ↓
XGBoost champion
   + SHAP
   + conformal uncertainty
   + Fare Opportunity Score
   + reliability scoring
   + route/booking-horizon intelligence
```

The frontend intentionally calls the API rather than loading model binaries directly. This keeps prediction logic, validation, uncertainty, and explainability in one backend contract.

## Dashboard capabilities

The main dashboard renders the exact production scenario controls:

- source and destination;
- airline;
- cabin class;
- departure period;
- stops;
- duration; and
- days before departure.

A prediction surfaces five headline decisions:

1. predicted fare;
2. 90% expected range;
3. Fare Opportunity Score and contextual label;
4. comparative reliability score; and
5. `BUY_NOW`, `MONITOR`, or `WAIT_OR_MONITOR` guidance.

The dashboard then exposes:

- **Why this fare?** — raw-feature SHAP contributions;
- **Booking-Horizon Intelligence** — XGBoost days-left counterfactual plus historical route/class median;
- **Route Analytics** — training-only 10th/median/90th percentile fare summaries by booking horizon;
- **What-if Simulator** — airline, departure-time, and stops counterfactuals;
- **Route Explorer** — route/class historical context;
- **Fare Trends** — booking-horizon analytics;
- **Model Insights** — champion metadata, warnings, SHAP, and live local telemetry; and
- **About** — model scope and limitations.

## Counterfactual safety

Phase 7 showed that tree-model days-left curves may contain sharp split-driven discontinuities. Phase 9 therefore never presents the model curve alone as a future-price forecast.

The UI overlays:

```text
Model what-if curve
+
Historical route/class booking-horizon median
```

and explicitly labels the curve as a model counterfactual rather than a guaranteed temporal trajectory.

## Streamlit API client

`frontend/client.py` provides the UI-to-API contract. The backend location is configured through:

```text
FLIGHT_FARE_API_URL
```

Default local value:

```text
http://127.0.0.1:8000
```

Docker Compose overrides it with the internal service address:

```text
http://api:8000
```

The client converts backend HTTP/network failures into product-readable errors instead of crashing the Streamlit process.

## Lightweight observability

Phase 8 already attached request IDs, latency headers, and structured request logs. Phase 9 adds bounded in-memory request telemetry through:

```text
GET /v1/telemetry
```

The endpoint reports:

- total requests;
- error requests and error rate;
- recent mean, p50, and p95 API latency;
- HTTP status counts; and
- highest-volume paths.

Telemetry is intentionally lightweight and resets when the API process restarts. It demonstrates post-deployment visibility without pretending to replace a production monitoring platform such as Prometheus/Grafana or an APM service.

## Warning-free API testing

The FastAPI smoke tests and API contract tests no longer use the deprecated Starlette/FastAPI `TestClient` path that produced dependency warnings in Phase 8.

They now use HTTPX `ASGITransport` directly against the application lifespan. This keeps the tests fully in-process while removing the previous TestClient/AnyIO deprecation warnings.

## Docker Compose

Phase 9 introduces a two-service deployment:

```text
api       → port 8000

dashboard → port 8501
```

Run both with:

```bash
make stack
```

or:

```bash
docker compose up --build
```

The dashboard waits for the API health check and talks to it over Docker's internal service network.

## Local development

Terminal 1:

```bash
make api
```

Terminal 2:

```bash
make dashboard
```

Open:

```text
http://localhost:8501
```

FastAPI documentation remains available at:

```text
http://localhost:8000/docs
```

## Quality gate

Run:

```bash
make phase9
```

The phase gate performs:

1. environment/import verification including Streamlit and Plotly;
2. strict canonical dataset validation;
3. real-artifact API/frontend contract smoke testing;
4. compilation of the Streamlit entrypoint;
5. lightweight telemetry validation;
6. safe Ruff auto-fixes across `src`, `scripts`, `tests`, `api`, and `frontend`;
7. Ruff formatting;
8. strict final Ruff validation; and
9. the full automated test suite.

Unsafe Ruff fixes remain disabled automatically. If a change could alter behavior, the gate fails visibly instead of silently modifying model/application logic.

## Phase 9 artifacts

Generated smoke report:

```text
reports/metrics/phase9_dashboard_smoke.json
```

Tracked application code:

```text
frontend/
├── app.py
├── client.py
├── dashboard_data.py
└── viewmodels.py
```

Tracked deployment configuration:

```text
.streamlit/config.toml
Dockerfile.dashboard
compose.yaml
```

## Limitations preserved in the UI

The interface must not imply capabilities the dataset does not support:

- no live airline inventory;
- no live booking availability;
- no guaranteed future fares;
- no claim that SHAP effects are causal;
- reliability score is not probability of correctness; and
- Buy/Wait guidance is historical/model-based decision support only.

Phase 10 will capture the actual application screenshot, architecture diagram, definitive results, setup guide, and recruiter-facing project narrative.
