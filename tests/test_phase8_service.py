"""Phase 8 production-service unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flight_fare_intelligence.intelligence import ComparableFareIndex
from flight_fare_intelligence.service import FareIntelligenceEngine


class DummyModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prices = []
        for _, row in frame.iterrows():
            base = 50_000.0 if row["class"] == "Business" else 6_000.0
            prices.append(base + (50 - float(row["days_left"])) * 100.0)
        return np.asarray(prices, dtype=float)


class DummyCalibrator:
    def interval_frame(
        self,
        features: pd.DataFrame,
        predicted_price: np.ndarray,
    ) -> pd.DataFrame:
        prediction = np.asarray(predicted_price, dtype=float)
        return pd.DataFrame(
            {
                "prediction_lower": prediction - 1_000.0,
                "prediction_upper": prediction + 1_000.0,
                "interval_width": np.full(len(prediction), 2_000.0),
                "calibration_level": ["class_horizon"] * len(prediction),
                "calibration_rows": [500] * len(prediction),
            },
            index=features.index,
        )


def _scenario() -> dict[str, str | float | int]:
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


def _engine() -> FareIntelligenceEngine:
    index_values = {}
    for horizon in ["1-7", "8-14", "15-21", "22-35", "36-49"]:
        index_values[("Delhi>Mumbai", "Economy", horizon)] = np.array(
            [5_000.0, 7_000.0, 9_000.0, 11_000.0, 13_000.0]
        )
    return FareIntelligenceEngine(
        model=DummyModel(),
        calibrator=DummyCalibrator(),
        comparable_index=ComparableFareIndex(values=index_values),
        reliability_reference=np.linspace(0.05, 0.50, 100),
        bundle_metadata={"coverage": 0.90, "calibration_rows": 100, "test_set_scored": False},
    )


def test_metadata_preserves_sealed_test_contract() -> None:
    metadata = _engine().metadata()
    assert metadata["champion"] == "xgboost"
    assert metadata["test_set_scored"] is False
    assert metadata["nominal_interval_coverage"] == 0.90


def test_prediction_returns_core_intelligence_without_explanation() -> None:
    result = _engine().predict(
        _scenario(),
        include_explanation=False,
        include_guidance=True,
    )
    assert result["predicted_fare"] > 0
    assert result["prediction_interval"]["lower"] < result["predicted_fare"]
    assert result["prediction_interval"]["upper"] > result["predicted_fare"]
    assert 0 <= result["fare_opportunity"]["fare_opportunity_score"] <= 100
    assert result["reliability"]["label"] in {"HIGH", "MEDIUM", "LOW"}
    assert result["booking_guidance"]["recommendation"] in {
        "BUY_NOW",
        "MONITOR",
        "WAIT_OR_MONITOR",
    }


def test_batch_prediction_disables_guidance_for_efficiency() -> None:
    results = _engine().batch_predict([_scenario(), _scenario()])
    assert len(results) == 2
    assert all("booking_guidance" not in result for result in results)
    assert all("explanation" not in result for result in results)


def test_what_if_reports_delta_from_base() -> None:
    result = _engine().what_if(_scenario(), feature="days_left", values=[7, 13, 21])
    assert len(result["scenarios"]) == 3
    current = next(row for row in result["scenarios"] if row["changed_value"] == 13)
    assert abs(float(current["difference_from_base"])) < 1e-9


def test_routes_are_directed_and_unique() -> None:
    routes = _engine().routes()
    assert routes == [
        {
            "route": "Delhi>Mumbai",
            "source_city": "Delhi",
            "destination_city": "Mumbai",
        }
    ]


def test_route_analytics_returns_all_horizons() -> None:
    result = _engine().route_analytics(
        source_city="Delhi",
        destination_city="Mumbai",
        cabin_class="Economy",
    )
    assert result["route"] == "Delhi>Mumbai"
    assert len(result["horizons"]) == 5
    assert result["horizons"][0]["median"] == 9_000.0


def test_booking_horizon_curve_contains_model_and_historical_context() -> None:
    result = _engine().booking_horizon_curve(_scenario())
    assert len(result["curve"]) == 49
    assert result["curve"][0]["days_left"] == 1
    assert result["curve"][-1]["days_left"] == 49
    assert result["curve"][0]["historical_horizon_median"] == 9_000.0
