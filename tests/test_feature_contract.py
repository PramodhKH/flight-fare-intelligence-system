from flight_fare_intelligence.schema import ANALYSIS_ONLY_COLUMNS, MODEL_FEATURES, TARGET_COLUMN


def test_deployment_feature_contract_is_exact():
    assert MODEL_FEATURES == [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "stops",
        "class",
        "duration",
        "days_left",
    ]


def test_flight_and_arrival_are_not_deployed_features():
    assert set(ANALYSIS_ONLY_COLUMNS).isdisjoint(MODEL_FEATURES)
    assert TARGET_COLUMN not in MODEL_FEATURES
