"""Run the Phase 9 dashboard/telemetry smoke gate with real local artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from api.main import create_app
from flight_fare_intelligence.service import FareIntelligenceEngine
from frontend.dashboard_data import AIRLINES, DEPARTURE_PERIODS, STOP_OPTIONS
from frontend.viewmodels import booking_curve_frame, prediction_cards, shap_frame, what_if_frame


async def _exercise_api(app: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://phase9") as client:
            prediction = await client.post("/v1/predict", json=scenario)
            booking = await client.post("/v1/booking-horizon", json=scenario)
            route = await client.get(
                "/v1/route-analytics",
                params={
                    "source_city": scenario["source_city"],
                    "destination_city": scenario["destination_city"],
                    "class": scenario["class"],
                },
            )
            what_if = {}
            for feature, values in {
                "airline": AIRLINES,
                "departure_time": DEPARTURE_PERIODS,
                "stops": STOP_OPTIONS,
            }.items():
                response = await client.post(
                    "/v1/what-if",
                    json={"scenario": scenario, "feature": feature, "values": values},
                )
                what_if[feature] = response
            telemetry = await client.get("/v1/telemetry")

    responses = {
        "prediction": prediction,
        "booking_horizon": booking,
        "route_analytics": route,
        **{f"what_if_{name}": response for name, response in what_if.items()},
        "telemetry": telemetry,
    }
    failures = {
        name: response.status_code
        for name, response in responses.items()
        if response.status_code != 200
    }
    if failures:
        raise RuntimeError(f"Phase 9 API/frontend smoke test failed: {failures}")
    return {name: response.json() for name, response in responses.items()}


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
                f"Missing Phase 9 prerequisite: {required}. Rerun make phase7 first."
            )

    phase7 = json.loads(args.phase7_report.read_text())
    if bool(phase7.get("test_set_scored", True)):
        raise RuntimeError("Phase 9 requires the Phase 7 test set to remain sealed")

    app_source = Path("frontend/app.py").read_text()
    compile(app_source, "frontend/app.py", "exec")

    engine = FareIntelligenceEngine.from_paths(
        model_path=args.champion_model,
        intelligence_path=args.intelligence_bundle,
    )
    app = create_app(engine)
    scenario = dict(phase7["demo"]["scenario"])
    payloads = asyncio.run(_exercise_api(app, scenario))

    prediction = payloads["prediction"]
    cards = prediction_cards(prediction)
    shap_rows = shap_frame(prediction)
    booking_rows = booking_curve_frame(
        payloads["booking_horizon"],
        current_days_left=int(scenario["days_left"]),
    )
    airline_rows = what_if_frame(payloads["what_if_airline"])

    if len(shap_rows) < 3:
        raise RuntimeError("Phase 9 expected at least three SHAP drivers")
    if len(booking_rows) != 49:
        raise RuntimeError("Phase 9 booking-horizon view must contain days 1..49")
    if airline_rows.empty:
        raise RuntimeError("Phase 9 airline what-if view is empty")

    report = {
        "phase": 9,
        "frontend": "Streamlit",
        "backend": "FastAPI",
        "champion": "xgboost",
        "test_set_scored": False,
        "smoke_status": "passed",
        "dashboard_contract": {
            "headline_cards": cards,
            "shap_drivers": len(shap_rows),
            "booking_horizon_rows": len(booking_rows),
            "airline_what_if_rows": len(airline_rows),
            "navigation_pages": [
                "Dashboard",
                "Route Explorer",
                "Fare Trends",
                "Model Insights",
                "About",
            ],
        },
        "telemetry": payloads["telemetry"],
        "warnings": [
            "Model counterfactual curves are not guaranteed future fare trajectories.",
            "Reliability is comparative uncertainty, not probability of correctness.",
            "The system does not use live airline inventory.",
        ],
    }
    output = Path("reports/metrics/phase9_dashboard_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
