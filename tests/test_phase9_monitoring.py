"""Phase 9 lightweight API telemetry tests."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from api.main import create_app
from flight_fare_intelligence.monitoring import RequestTelemetry


class MinimalEngine:
    def metadata(self) -> dict[str, Any]:
        return {
            "champion": "xgboost",
            "test_set_scored": False,
            "model_features": [],
            "nominal_interval_coverage": 0.9,
            "uncertainty_calibration_rows": 1,
            "warnings": [],
        }


def test_empty_telemetry_snapshot_is_stable() -> None:
    telemetry = RequestTelemetry(max_latency_samples=5)
    snapshot = telemetry.snapshot()
    assert snapshot["total_requests"] == 0
    assert snapshot["p95_latency_ms"] == 0.0
    assert snapshot["error_rate_percent"] == 0.0


def test_telemetry_tracks_status_latency_and_bounded_samples() -> None:
    telemetry = RequestTelemetry(max_latency_samples=2)
    telemetry.observe(path="/v1/predict", status_code=200, latency_ms=10.0)
    telemetry.observe(path="/v1/predict", status_code=500, latency_ms=30.0)
    telemetry.observe(path="/v1/health", status_code=200, latency_ms=20.0)
    snapshot = telemetry.snapshot()
    assert snapshot["total_requests"] == 3
    assert snapshot["error_requests"] == 1
    assert snapshot["latency_samples"] == 2
    assert snapshot["status_counts"] == {"200": 2, "500": 1}
    assert snapshot["top_paths"][0] == {"path": "/v1/predict", "requests": 2}


def test_api_telemetry_endpoint_reports_completed_requests() -> None:
    async def exercise() -> dict[str, Any]:
        app = create_app(MinimalEngine())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                await client.get("/v1/health")
                response = await client.get("/v1/telemetry")
                return response.json()

    snapshot = asyncio.run(exercise())
    assert snapshot["total_requests"] >= 1
    assert snapshot["error_rate_percent"] == 0.0
