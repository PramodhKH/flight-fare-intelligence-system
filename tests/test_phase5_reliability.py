from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_fare_intelligence.reliability import (
    add_market_support,
    build_reliability_frame,
    score_segment,
    segment_table,
    stress_test_table,
    worst_error_cases,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "airline": ["Vistara", "Vistara", "Indigo", "Air_India"],
            "source_city": ["Delhi", "Delhi", "Mumbai", "Delhi"],
            "destination_city": ["Mumbai", "Mumbai", "Delhi", "Bangalore"],
            "departure_time": ["Morning", "Morning", "Evening", "Night"],
            "stops": ["zero", "zero", "one", "one"],
            "class": ["Economy", "Economy", "Economy", "Business"],
            "duration": [2.0, 2.2, 3.0, 5.0],
            "days_left": [12, 5, 20, 3],
        }
    )


def test_score_segment_reports_bias_and_underprediction() -> None:
    frame = pd.DataFrame(
        {
            "actual_price": [100.0, 200.0, 300.0],
            "predicted_price": [90.0, 210.0, 270.0],
        }
    )
    metrics = score_segment(frame)
    assert metrics["rows"] == 3
    assert metrics["mae"] == pytest.approx(16.6666667)
    assert metrics["mean_bias"] == pytest.approx(-10.0)
    assert metrics["underprediction_rate"] == pytest.approx(66.6666667)


def test_add_market_support_marks_unseen_context() -> None:
    training = _features().iloc[:3].copy()
    evaluation = _features().iloc[[0, 3]].copy()
    enriched = add_market_support(evaluation, training)
    assert enriched["training_context_rows"].tolist() == [2, 0]
    assert enriched["support_bucket"].tolist() == ["low_1_25", "unseen"]


def test_build_reliability_frame_adds_required_diagnostics() -> None:
    validation = _features().iloc[:2].copy()
    training = _features().iloc[:3].copy()
    frame = build_reliability_frame(
        validation_features=validation,
        validation_record_ids=pd.Series([10, 11]),
        actual_price=pd.Series([7_000.0, 8_000.0]),
        predicted_price=np.array([6_800.0, 8_500.0]),
        training_frame=training,
    )
    assert {
        "record_id",
        "actual_price",
        "predicted_price",
        "absolute_error",
        "route",
        "booking_horizon",
        "fare_band",
        "training_context_rows",
        "support_bucket",
    }.issubset(frame.columns)
    assert frame["route"].tolist() == ["Delhi>Mumbai", "Delhi>Mumbai"]


def test_segment_table_scores_each_requested_dimension() -> None:
    validation = _features().copy()
    frame = build_reliability_frame(
        validation_features=validation,
        validation_record_ids=pd.Series(range(4)),
        actual_price=pd.Series([7_000.0, 8_000.0, 9_000.0, 50_000.0]),
        predicted_price=np.array([7_100.0, 7_800.0, 9_300.0, 48_000.0]),
        training_frame=validation,
    )
    segments = segment_table(frame, ["class", "booking_horizon"])
    assert set(segments["dimension"]) == {"class", "booking_horizon"}
    assert "Business" in set(segments["segment"])


def test_stress_table_keeps_small_tail_segments_visible() -> None:
    validation = _features().copy()
    frame = build_reliability_frame(
        validation_features=validation,
        validation_record_ids=pd.Series(range(4)),
        actual_price=pd.Series([7_000.0, 8_000.0, 9_000.0, 101_000.0]),
        predicted_price=np.array([7_100.0, 7_800.0, 9_300.0, 85_000.0]),
        training_frame=validation,
    )
    stress = stress_test_table(frame, long_duration_threshold=4.0)
    tail = stress.loc[stress["stress_segment"] == "very_high_fare_100k_plus"].iloc[0]
    assert tail["rows"] == 1
    assert tail["mean_bias"] == pytest.approx(-16_000.0)


def test_worst_error_cases_rejects_nonpositive_limit() -> None:
    frame = pd.DataFrame({"absolute_error": [1.0]})
    with pytest.raises(ValueError, match="positive"):
        worst_error_cases(frame, limit=0)
