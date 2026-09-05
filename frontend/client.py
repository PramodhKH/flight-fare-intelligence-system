"""HTTP client used by the Streamlit dashboard to call the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class APIClientError(RuntimeError):
    """Raised when the dashboard cannot obtain a valid backend response."""


class FlightFareAPIClient:
    """Small typed facade over the versioned Flight Fare Intelligence API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved = base_url or os.getenv("FLIGHT_FARE_API_URL", DEFAULT_API_URL)
        self.base_url = resolved.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise APIClientError(
                f"API request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise APIClientError(
                f"Unable to reach Flight Fare Intelligence API at {self.base_url}: {exc}"
            ) from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def model_metadata(self) -> dict[str, Any]:
        return self._request("GET", "/v1/model")

    def telemetry(self) -> dict[str, Any]:
        return self._request("GET", "/v1/telemetry")

    def predict(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/predict", json=scenario)

    def batch_predict(
        self,
        scenarios: list[dict[str, Any]],
        *,
        include_explanations: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/predict/batch",
            json={
                "scenarios": scenarios,
                "include_explanations": include_explanations,
            },
        )

    def what_if(
        self,
        scenario: dict[str, Any],
        *,
        feature: str,
        values: list[str | int | float],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/what-if",
            json={"scenario": scenario, "feature": feature, "values": values},
        )

    def booking_horizon(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/booking-horizon", json=scenario)

    def routes(self) -> dict[str, Any]:
        return self._request("GET", "/v1/routes")

    def route_analytics(
        self,
        *,
        source_city: str,
        destination_city: str,
        cabin_class: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/v1/route-analytics",
            params={
                "source_city": source_city,
                "destination_city": destination_city,
                "class": cabin_class,
            },
        )
