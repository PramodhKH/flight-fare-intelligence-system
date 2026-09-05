"""Phase 8 FastAPI service for flight-fare prediction and analytics."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from flight_fare_intelligence.api_schemas import (
    BatchPredictionRequest,
    FlightScenario,
    WhatIfRequest,
)
from flight_fare_intelligence.service import FareIntelligenceEngine

LOGGER = logging.getLogger("flight_fare_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _default_engine() -> FareIntelligenceEngine:
    return FareIntelligenceEngine.from_paths(
        model_path=os.getenv("MODEL_PATH", "models/phase4_champion.joblib"),
        intelligence_path=os.getenv(
            "INTELLIGENCE_BUNDLE_PATH",
            "models/phase7_intelligence_bundle.joblib",
        ),
    )


def create_app(engine: Any | None = None) -> FastAPI:
    """Create the API application, optionally injecting an engine for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = _default_engine()
        LOGGER.info("Flight fare intelligence artifacts loaded")
        yield

    app = FastAPI(
        title="Flight Fare Intelligence API",
        version="1.0.0",
        description=(
            "Explainable fare prediction, uncertainty, contextual scoring, route analytics, "
            "and model-based what-if decision support."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start = perf_counter()
        response = await call_next(request)
        elapsed_ms = (perf_counter() - start) * 1_000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        LOGGER.info(
            "%s %s status=%s latency_ms=%.3f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    @app.exception_handler(KeyError)
    async def key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "Flight Fare Intelligence API",
            "version": "v1",
            "docs": "/docs",
            "health": "/v1/health",
        }

    @app.get("/v1/health")
    def health(request: Request) -> dict[str, Any]:
        active_engine = request.app.state.engine
        metadata = active_engine.metadata()
        return {
            "status": "ok",
            "model_loaded": True,
            "champion": metadata["champion"],
            "test_set_scored": metadata["test_set_scored"],
        }

    @app.get("/v1/model")
    def model_metadata(request: Request) -> dict[str, Any]:
        return request.app.state.engine.metadata()

    @app.post("/v1/predict")
    def predict(payload: FlightScenario, request: Request) -> dict[str, Any]:
        return request.app.state.engine.predict(payload.as_model_dict())

    @app.post("/v1/predict/batch")
    def batch_predict(payload: BatchPredictionRequest, request: Request) -> dict[str, Any]:
        scenarios = [scenario.as_model_dict() for scenario in payload.scenarios]
        predictions = request.app.state.engine.batch_predict(
            scenarios,
            include_explanations=payload.include_explanations,
        )
        return {"rows": len(predictions), "predictions": predictions}

    @app.post("/v1/what-if")
    def what_if(payload: WhatIfRequest, request: Request) -> dict[str, Any]:
        base = payload.scenario.as_model_dict()
        normalized_values: list[str | int | float] = []
        for value in payload.values:
            changed = dict(base)
            changed[payload.feature] = value
            try:
                normalized = FlightScenario.model_validate(changed).as_model_dict()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            normalized_values.append(normalized[payload.feature])
        return request.app.state.engine.what_if(
            base,
            feature=payload.feature,
            values=normalized_values,
        )

    @app.post("/v1/booking-horizon")
    def booking_horizon(payload: FlightScenario, request: Request) -> dict[str, Any]:
        return request.app.state.engine.booking_horizon_curve(payload.as_model_dict())

    @app.get("/v1/routes")
    def routes(request: Request) -> dict[str, Any]:
        rows = request.app.state.engine.routes()
        return {"rows": len(rows), "routes": rows}

    @app.get("/v1/route-analytics")
    def route_analytics(
        request: Request,
        source_city: str,
        destination_city: str,
        cabin_class: str = Query(alias="class"),
    ) -> dict[str, Any]:
        if source_city == destination_city:
            raise HTTPException(status_code=422, detail="source and destination must differ")
        return request.app.state.engine.route_analytics(
            source_city=source_city,
            destination_city=destination_city,
            cabin_class=cabin_class,
        )

    return app


app = create_app()
