#!/usr/bin/env python
"""Run a real-artifact Phase 8 API smoke test without touching the sealed test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

from api.main import create_app
from flight_fare_intelligence.service import FareIntelligenceEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--champion-model",
        type=Path,
        default=Path("models/phase4_champion.joblib"),
    )
    parser.add_argument(
        "--intelligence-bundle",
        type=Path,
        default=Path("models/phase7_intelligence_bundle.joblib"),
    )
    parser.add_argument(
        "--phase7-report",
        type=Path,
        default=Path("reports/metrics/phase7_intelligence_summary.json"),
    )
    args = parser.parse_args()

    for required in [args.champion_model, args.intelligence_bundle, args.phase7_report]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing Phase 8 prerequisite: {required}. Rerun make phase7 first."
            )

    phase7 = json.loads(args.phase7_report.read_text())
    if bool(phase7.get("test_set_scored", True)):
        raise RuntimeError("Phase 8 requires the Phase 7 test set to remain sealed")

    engine = FareIntelligenceEngine.from_paths(
        model_path=args.champion_model,
        intelligence_path=args.intelligence_bundle,
    )
    app = create_app(engine)
    scenario = dict(phase7["demo"]["scenario"])

    start = perf_counter()
    with TestClient(app) as client:
        health = client.get("/v1/health")
        model = client.get("/v1/model")
        prediction = client.post("/v1/predict", json=scenario)
        batch = client.post(
            "/v1/predict/batch",
            json={"scenarios": [scenario, scenario], "include_explanations": False},
        )
        what_if = client.post(
            "/v1/what-if",
            json={"scenario": scenario, "feature": "days_left", "values": [7, 13, 21]},
        )
        routes = client.get("/v1/routes")
        route_analytics = client.get(
            "/v1/route-analytics",
            params={
                "source_city": scenario["source_city"],
                "destination_city": scenario["destination_city"],
                "class": scenario["class"],
            },
        )
        booking_curve = client.post("/v1/booking-horizon", json=scenario)
    total_ms = (perf_counter() - start) * 1_000.0

    responses = {
        "health": health,
        "model": model,
        "prediction": prediction,
        "batch": batch,
        "what_if": what_if,
        "routes": routes,
        "route_analytics": route_analytics,
        "booking_horizon": booking_curve,
    }
    failures = {
        name: response.status_code
        for name, response in responses.items()
        if response.status_code != 200
    }
    if failures:
        raise RuntimeError(f"Phase 8 API smoke test failed: {failures}")

    prediction_payload = prediction.json()
    report = {
        "phase": 8,
        "service": "FastAPI",
        "champion": model.json()["champion"],
        "test_set_scored": False,
        "endpoints_smoke_tested": list(responses),
        "smoke_status": "passed",
        "total_testclient_smoke_ms": round(total_ms, 3),
        "demo": {
            "scenario": scenario,
            "predicted_fare": prediction_payload["predicted_fare"],
            "prediction_interval": prediction_payload["prediction_interval"],
            "fare_opportunity": prediction_payload["fare_opportunity"],
            "reliability": prediction_payload["reliability"],
            "booking_guidance": prediction_payload["booking_guidance"],
            "explanation_top_drivers": prediction_payload["explanation"]["top_drivers"],
        },
        "production_contract": {
            "max_batch_rows": 100,
            "api_prefix": "/v1",
            "openapi_docs": "/docs",
            "request_id_header": "X-Request-ID",
            "latency_header": "X-Process-Time-Ms",
        },
    }

    output = Path("reports/metrics/phase8_api_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
