# Flight Fare Intelligence API

Phase 8 exposes the locked XGBoost + Phase 7 intelligence stack through a versioned FastAPI service.

Run locally after generating the Phase 4 and Phase 7 artifacts:

```bash
make api
```

Interactive OpenAPI docs:

```text
http://localhost:8000/docs
```

Core endpoints:

```text
GET  /v1/health
GET  /v1/model
POST /v1/predict
POST /v1/predict/batch
POST /v1/what-if
POST /v1/booking-horizon
GET  /v1/routes
GET  /v1/route-analytics
```

The service returns point predictions, conformal intervals, Fare Opportunity Score, comparative reliability, model-based booking guidance, and SHAP explanations. Guidance is historical/model-based only and does not represent live airline inventory or guaranteed future fare movement.
