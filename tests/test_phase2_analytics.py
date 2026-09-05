from __future__ import annotations

import pandas as pd

from flight_fare_intelligence.analytics import add_analysis_features, scenario_duplication_summary


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "airline": ["Vistara", "Vistara", "Indigo"],
            "source_city": ["Delhi", "Delhi", "Mumbai"],
            "destination_city": ["Mumbai", "Mumbai", "Delhi"],
            "departure_time": ["Morning", "Morning", "Evening"],
            "stops": ["zero", "zero", "one"],
            "class": ["Economy", "Economy", "Economy"],
            "duration": [2.0, 2.0, 3.0],
            "days_left": [5, 5, 20],
            "price": [5000, 5500, 6000],
        }
    )


def test_add_analysis_features_creates_route_and_horizon() -> None:
    frame = add_analysis_features(_sample())
    assert frame.loc[0, "route"] == "Delhi → Mumbai"
    assert str(frame.loc[0, "booking_horizon"]) == "1-7"
    assert str(frame.loc[2, "booking_horizon"]) == "15-21"


def test_scenario_duplication_detects_repeated_production_inputs() -> None:
    summary = scenario_duplication_summary(_sample())
    assert summary["rows_with_repeated_feature_vector"] == 1
    assert summary["repeated_feature_groups"] == 1
    assert summary["repeated_groups_with_price_variation"] == 1
    assert summary["maximum_within_group_price_range"] == 500.0
