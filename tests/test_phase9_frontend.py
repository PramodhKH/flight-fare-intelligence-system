"""Phase 9 dashboard helper and API-client tests without launching a browser."""

from __future__ import annotations

import json
from typing import Any

import httpx

from frontend.client import APIClientError, FlightFareAPIClient
from frontend.dashboard_data import AIRLINES, alternatives, build_dashboard_bundle, default_scenario
from frontend.viewmodels import (
    booking_curve_frame,
    prediction_cards,
    pretty_label,
    what_if_frame,
)


class FakeDashboardClient:
    def predict(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {
            "scenario": scenario,
            "predicted_fare": 11_801.69,
            "prediction_interval": {"lower": 9_312.3, "upper": 15_342.89},
            "fare_opportunity": {
                "fare_opportunity_score": 34,
                "fare_position": "ABOVE_TYPICAL",
            },
            "reliability": {"score": 45, "label": "MEDIUM"},
            "booking_guidance": {
                "recommendation": "MONITOR",
                "expected_wait_change_percent": 10.2,
            },
        }

    def booking_horizon(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {
            "curve": [
                {
                    "days_left": day,
                    "predicted_fare": 10_000.0 + day,
                    "historical_horizon_median": 9_500.0,
                }
                for day in range(1, 50)
            ]
        }

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

    def what_if(
        self,
        scenario: dict[str, Any],
        *,
        feature: str,
        values: list[str | int | float],
    ) -> dict[str, Any]:
        return {
            "changed_feature": feature,
            "scenarios": [
                {
                    "changed_value": value,
                    "predicted_fare": 10_000.0 + index,
                    "difference_from_base": float(index),
                }
                for index, value in enumerate(values)
            ],
        }


def test_default_scenario_matches_product_demo() -> None:
    scenario = default_scenario()
    assert scenario["source_city"] == "Delhi"
    assert scenario["destination_city"] == "Mumbai"
    assert scenario["days_left"] == 13


def test_alternatives_keep_current_first_without_duplicates() -> None:
    result = alternatives("Vistara", AIRLINES)
    assert result[0] == "Vistara"
    assert len(result) == len(set(result))
    assert set(result) == set(AIRLINES)


def test_build_dashboard_bundle_fetches_all_decision_views() -> None:
    bundle = build_dashboard_bundle(FakeDashboardClient(), default_scenario())
    assert bundle["prediction"]["predicted_fare"] == 11_801.69
    assert len(bundle["booking_horizon"]["curve"]) == 49
    assert set(bundle["what_if"]) == {"airline", "departure_time", "stops"}


def test_viewmodel_formatting_and_current_booking_marker() -> None:
    client = FakeDashboardClient()
    scenario = default_scenario()
    bundle = build_dashboard_bundle(client, scenario)
    cards = prediction_cards(bundle["prediction"])
    curve = booking_curve_frame(bundle["booking_horizon"], current_days_left=13)
    assert cards["predicted_fare"] == "₹11,802"
    assert cards["opportunity_label"] == "Above Typical"
    assert int(curve["is_current"].sum()) == 1


def test_pretty_label_and_what_if_frame_are_human_readable() -> None:
    assert pretty_label("two_or_more") == "2+ Stops"
    payload = FakeDashboardClient().what_if(
        default_scenario(),
        feature="airline",
        values=["Vistara", "Air_India"],
    )
    frame = what_if_frame(payload)
    assert set(frame["display_value"]) == {"Vistara", "Air India"}


def test_api_client_supports_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"detail": "not found"})

    client = FlightFareAPIClient(
        "http://testserver",
        transport=httpx.MockTransport(handler),
    )
    assert client.health()["status"] == "ok"


def test_api_client_raises_product_error_for_http_failures() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=json.dumps({"detail": "offline"}))

    client = FlightFareAPIClient(
        "http://testserver",
        transport=httpx.MockTransport(handler),
    )
    try:
        client.health()
    except APIClientError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("503 response should raise APIClientError")
