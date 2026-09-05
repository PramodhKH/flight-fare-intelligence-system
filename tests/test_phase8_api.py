"""Phase 8 FastAPI contract tests with an injected deterministic engine."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from api.main import create_app
from flight_fare_intelligence.api_schemas import FlightScenario


class FakeEngine:
    def metadata(self) -> dict[str, Any]:
        return {
            "phase": 8,
            "champion": "xgboost",
            "model_features": ["airline", "source_city"],
            "nominal_interval_coverage": 0.90,
            "uncertainty_calibration_rows": 22_407,
            "test_set_scored": False,
            "warnings": [],
        }

    def predict(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {"scenario": scenario, "predicted_fare": 11_801.69}

    def batch_predict(
        self,
        scenarios: list[dict[str, Any]],
        *,
        include_explanations: bool,
    ) -> list[dict[str, Any]]:
        return [
            {
                "scenario": scenario,
                "predicted_fare": 11_801.69,
                "include_explanations": include_explanations,
            }
            for scenario in scenarios
        ]

    def what_if(
        self,
        scenario: dict[str, Any],
        *,
        feature: str,
        values: list[Any],
    ) -> dict[str, Any]:
        return {
            "base_predicted_fare": 11_801.69,
            "changed_feature": feature,
            "scenarios": [{"changed_value": value} for value in values],
        }

    def booking_horizon_curve(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {"scenario": scenario, "curve": [{"days_left": 1, "predicted_fare": 15_000.0}]}

    def routes(self) -> list[dict[str, str]]:
        return [
            {
                "route": "Delhi>Mumbai",
                "source_city": "Delhi",
                "destination_city": "Mumbai",
            }
        ]

    def route_analytics(
        self,
        *,
        source_city: str,
        destination_city: str,
        cabin_class: str,
    ) -> dict[str, Any]:
        return {
            "route": f"{source_city}>{destination_city}",
            "class": cabin_class,
            "horizons": [],
        }


def _payload() -> dict[str, Any]:
    return {
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": "one",
        "class": "Economy",
        "duration": 11.67,
        "days_left": 13,
    }


def _client() -> TestClient:
    return TestClient(create_app(FakeEngine()))


def test_scenario_rejects_same_source_and_destination() -> None:
    payload = _payload()
    payload["destination_city"] = "Delhi"
    try:
        FlightScenario.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("same-city route should be rejected")


def test_health_and_observability_headers() -> None:
    with _client() as client:
        response = client.get("/v1/health", headers={"X-Request-ID": "phase8-test"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"] == "phase8-test"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0


def test_predict_uses_class_alias_contract() -> None:
    with _client() as client:
        response = client.post("/v1/predict", json=_payload())
    assert response.status_code == 200
    assert response.json()["scenario"]["class"] == "Economy"


def test_invalid_route_returns_422() -> None:
    payload = _payload()
    payload["destination_city"] = "Delhi"
    with _client() as client:
        response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction_is_bounded_and_returns_row_count() -> None:
    with _client() as client:
        response = client.post(
            "/v1/predict/batch",
            json={"scenarios": [_payload(), _payload()], "include_explanations": False},
        )
    assert response.status_code == 200
    assert response.json()["rows"] == 2


def test_what_if_validates_changed_scenarios() -> None:
    with _client() as client:
        response = client.post(
            "/v1/what-if",
            json={"scenario": _payload(), "feature": "days_left", "values": [7, 13, 21]},
        )
    assert response.status_code == 200
    assert response.json()["changed_feature"] == "days_left"


def test_routes_and_route_analytics_endpoints() -> None:
    with _client() as client:
        routes = client.get("/v1/routes")
        analytics = client.get(
            "/v1/route-analytics",
            params={"source_city": "Delhi", "destination_city": "Mumbai", "class": "Economy"},
        )
    assert routes.status_code == 200
    assert routes.json()["rows"] == 1
    assert analytics.status_code == 200
    assert analytics.json()["route"] == "Delhi>Mumbai"


def test_booking_horizon_endpoint() -> None:
    with _client() as client:
        response = client.post("/v1/booking-horizon", json=_payload())
    assert response.status_code == 200
    assert response.json()["curve"][0]["days_left"] == 1
