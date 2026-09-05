from __future__ import annotations

import numpy as np
import pandas as pd

from flight_fare_intelligence.intelligence import (
    benchmark_table,
    booking_guidance,
    booking_horizon_label,
    build_comparable_fare_index,
    counterfactual_table,
    fare_opportunity,
)


class DummyFareModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            4_000.0
            + frame["days_left"].to_numpy(dtype=float) * 50.0
            + frame["duration"].to_numpy(dtype=float) * 100.0
        )


def _scenario() -> dict[str, object]:
    return {
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": "one",
        "class": "Economy",
        "duration": 2.0,
        "days_left": 12,
    }


def test_booking_horizon_labels_match_project_contract() -> None:
    assert booking_horizon_label(1) == "1-7"
    assert booking_horizon_label(12) == "8-14"
    assert booking_horizon_label(21) == "15-21"
    assert booking_horizon_label(35) == "22-35"
    assert booking_horizon_label(49) == "36-49"


def test_fare_opportunity_rewards_lower_comparable_fares() -> None:
    training = pd.DataFrame(
        {
            "source_city": ["Delhi"] * 10,
            "destination_city": ["Mumbai"] * 10,
            "class": ["Economy"] * 10,
            "days_left": [12] * 10,
            "price": np.arange(5_000.0, 15_000.0, 1_000.0),
        }
    )
    index = build_comparable_fare_index(training)
    low = fare_opportunity(5_500.0, _scenario(), index)
    high = fare_opportunity(13_500.0, _scenario(), index)
    assert int(low["fare_opportunity_score"]) > int(high["fare_opportunity_score"])
    assert low["fare_position"] in {"EXCELLENT_VALUE", "GOOD_VALUE"}
    assert high["fare_position"] in {"ABOVE_TYPICAL", "EXPENSIVE"}
    summary = benchmark_table(index)
    assert int(summary.iloc[0]["rows"]) == 10


def test_counterfactual_table_changes_only_requested_dimension() -> None:
    model = DummyFareModel()
    table = counterfactual_table(
        model,
        _scenario(),
        feature="days_left",
        values=[5, 10, 20],
    )
    assert table["changed_feature"].unique().tolist() == ["days_left"]
    assert table["changed_value"].tolist() == [5, 10, 20]
    assert table["predicted_fare"].is_monotonic_increasing


def test_booking_guidance_uses_cautious_three_state_policy() -> None:
    curve = pd.DataFrame(
        {
            "days_left": list(range(1, 13)),
            "predicted_fare": [15_000.0 - value * 500.0 for value in range(1, 13)],
        }
    )
    guidance = booking_guidance(
        current_days_left=12,
        current_fare=9_000.0,
        opportunity_score=65,
        curve=curve,
        wait_window_days=7,
        material_change_percent=5.0,
    )
    assert guidance["recommendation"] == "BUY_NOW"
    assert float(guidance["expected_wait_change_percent"]) > 5.0
