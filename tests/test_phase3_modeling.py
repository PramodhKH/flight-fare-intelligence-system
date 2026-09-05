from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_fare_intelligence.modeling import (
    attach_split_assignments,
    build_linear_regression_pipeline,
    prediction_frame,
    regression_metrics,
)
from flight_fare_intelligence.schema import MODEL_FEATURES


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3],
            "airline": ["Indigo", "Vistara", "Indigo", "Vistara"],
            "source_city": ["Delhi", "Delhi", "Mumbai", "Mumbai"],
            "destination_city": ["Mumbai", "Mumbai", "Delhi", "Delhi"],
            "departure_time": ["Morning", "Evening", "Morning", "Evening"],
            "stops": ["zero", "one", "zero", "one"],
            "class": ["Economy", "Economy", "Business", "Business"],
            "duration": [2.0, 3.0, 2.2, 3.2],
            "days_left": [30, 10, 25, 5],
            "price": [5000, 7000, 45000, 52000],
        }
    )


def test_linear_pipeline_accepts_production_feature_contract() -> None:
    frame = _tiny_frame()
    pipeline = build_linear_regression_pipeline()
    pipeline.fit(frame[MODEL_FEATURES], frame["price"])
    predictions = pipeline.predict(frame[MODEL_FEATURES])
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_pipeline_handles_unseen_categories_at_inference() -> None:
    frame = _tiny_frame()
    pipeline = build_linear_regression_pipeline()
    pipeline.fit(frame[MODEL_FEATURES], frame["price"])
    scenario = frame[MODEL_FEATURES].iloc[[0]].copy()
    scenario.loc[:, "airline"] = "AirAsia"
    with pytest.warns(UserWarning, match="Found unknown categories"):
        prediction = pipeline.predict(scenario)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_regression_metrics_are_calculated_in_expected_units() -> None:
    truth = np.array([100.0, 200.0, 300.0])
    prediction = np.array([110.0, 190.0, 330.0])
    metrics = regression_metrics(truth, prediction)
    assert metrics.rmse == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert metrics.mae == pytest.approx(50 / 3)
    assert metrics.mape == pytest.approx((0.10 + 0.05 + 0.10) / 3 * 100)
    assert metrics.p90_absolute_error >= metrics.median_absolute_error


def test_split_assignment_merge_preserves_rows_and_assignments() -> None:
    frame = _tiny_frame()
    assignments = pd.DataFrame(
        {
            "record_id": [0, 1, 2, 3],
            "scenario_id": [10, 11, 12, 13],
            "split": ["train", "train", "validation", "test"],
        }
    )
    merged = attach_split_assignments(frame, assignments)
    assert len(merged) == len(frame)
    assert merged["split"].tolist() == ["train", "train", "validation", "test"]


def test_prediction_frame_residual_sign_is_actual_minus_predicted() -> None:
    result = prediction_frame(
        pd.Series([1, 2]),
        pd.Series([100.0, 200.0]),
        np.array([90.0, 220.0]),
    )
    assert result["residual"].tolist() == [10.0, -20.0]
    assert result["absolute_error"].tolist() == [10.0, 20.0]
