"""Pure dashboard-data orchestration shared by Streamlit and tests."""

from __future__ import annotations

from typing import Any, Protocol

AIRLINES = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
DEPARTURE_PERIODS = [
    "Early_Morning",
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
    "Late_Night",
]
STOP_OPTIONS = ["zero", "one", "two_or_more"]
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
CABIN_CLASSES = ["Economy", "Business"]


class DashboardClient(Protocol):
    def predict(self, scenario: dict[str, Any]) -> dict[str, Any]: ...

    def booking_horizon(self, scenario: dict[str, Any]) -> dict[str, Any]: ...

    def route_analytics(
        self,
        *,
        source_city: str,
        destination_city: str,
        cabin_class: str,
    ) -> dict[str, Any]: ...

    def what_if(
        self,
        scenario: dict[str, Any],
        *,
        feature: str,
        values: list[str | int | float],
    ) -> dict[str, Any]: ...


def default_scenario() -> dict[str, Any]:
    """Return the canonical deterministic demo used throughout Phases 7–9."""
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


def alternatives(current: str, values: list[str]) -> list[str]:
    """Return deterministic what-if choices with the current value retained first."""
    return [current, *[value for value in values if value != current]]


def build_dashboard_bundle(
    client: DashboardClient,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Fetch the complete set of API payloads required by the main dashboard."""
    prediction = client.predict(scenario)
    booking = client.booking_horizon(scenario)
    route = client.route_analytics(
        source_city=str(scenario["source_city"]),
        destination_city=str(scenario["destination_city"]),
        cabin_class=str(scenario["class"]),
    )
    airline = client.what_if(
        scenario,
        feature="airline",
        values=alternatives(str(scenario["airline"]), AIRLINES),
    )
    departure = client.what_if(
        scenario,
        feature="departure_time",
        values=alternatives(str(scenario["departure_time"]), DEPARTURE_PERIODS),
    )
    stops = client.what_if(
        scenario,
        feature="stops",
        values=alternatives(str(scenario["stops"]), STOP_OPTIONS),
    )
    return {
        "prediction": prediction,
        "booking_horizon": booking,
        "route_analytics": route,
        "what_if": {
            "airline": airline,
            "departure_time": departure,
            "stops": stops,
        },
    }
