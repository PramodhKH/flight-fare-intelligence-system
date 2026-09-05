from __future__ import annotations

import numpy as np
import pandas as pd

from flight_fare_intelligence.uncertainty import (
    comparative_reliability_score,
    deterministic_validation_partition,
    evaluate_intervals,
    fit_segment_conformal_calibrator,
    reliability_reference,
)


def _features(rows: int, *, cabin_class: str = "Economy", days_left: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": [cabin_class] * rows,
            "days_left": [days_left] * rows,
        }
    )


def test_validation_partition_keeps_scenarios_together() -> None:
    frame = pd.DataFrame(
        {
            "scenario_id": [11, 11, 22, 22, 33, 44, 55, 66, 77, 88],
        }
    )
    mask = deterministic_validation_partition(frame, calibration_fraction=0.5, random_seed=42)
    grouped = (
        pd.DataFrame({"scenario_id": frame["scenario_id"], "side": mask})
        .groupby("scenario_id")["side"]
        .nunique()
    )
    assert int((grouped > 1).sum()) == 0
    assert mask.any()
    assert (~mask).any()


def test_segment_conformal_returns_ordered_nonnegative_intervals() -> None:
    rows = 300
    features = _features(rows)
    prediction = np.linspace(5_000.0, 15_000.0, rows)
    residual = np.tile(np.array([-1_000.0, -250.0, 100.0, 900.0, 1_500.0]), 60)
    actual = prediction + residual
    calibrator = fit_segment_conformal_calibrator(
        features,
        actual,
        prediction,
        coverage=0.90,
        min_rows=100,
    )
    intervals = calibrator.interval_frame(features.iloc[:5], prediction[:5])
    assert (intervals["prediction_lower"] >= 0.0).all()
    assert (intervals["prediction_upper"] >= intervals["prediction_lower"]).all()
    assert (intervals["calibration_rows"] >= 100).all()


def test_high_fare_business_tail_guardrail_is_more_conservative() -> None:
    rows = 220
    economy = _features(rows, cabin_class="Economy", days_left=5)
    business = _features(rows, cabin_class="Business", days_left=5)
    features = pd.concat([economy, business], ignore_index=True)
    prediction = np.concatenate(
        [
            np.full(rows, 70_000.0),
            np.full(rows, 70_000.0),
        ]
    )
    base_residuals = np.linspace(-2_000.0, 12_000.0, rows)
    residuals = np.concatenate([base_residuals, base_residuals])
    actual = prediction + residuals
    calibrator = fit_segment_conformal_calibrator(
        features,
        actual,
        prediction,
        coverage=0.90,
        min_rows=100,
        tail_min_rows=50,
        tail_upper_quantile=0.995,
    )
    intervals = calibrator.interval_frame(
        pd.DataFrame(
            {
                "class": ["Economy", "Business"],
                "days_left": [5, 5],
            }
        ),
        np.array([70_000.0, 70_000.0]),
    )
    economy_upper = float(intervals.iloc[0]["prediction_upper"])
    business_upper = float(intervals.iloc[1]["prediction_upper"])
    assert business_upper >= economy_upper


def test_interval_evaluation_reports_expected_coverage() -> None:
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    intervals = pd.DataFrame(
        {
            "prediction_lower": [5.0, 25.0, 25.0, 35.0],
            "prediction_upper": [15.0, 30.0, 35.0, 45.0],
        }
    )
    metrics = evaluate_intervals(actual, intervals)
    assert metrics["rows"] == 4
    assert metrics["coverage_percent"] == 75.0
    assert metrics["lower_miss_percent"] == 25.0
    assert metrics["upper_miss_percent"] == 0.0


def test_reliability_score_decreases_as_relative_interval_widens() -> None:
    reference = reliability_reference(np.linspace(0.05, 0.50, 100))
    narrow_score, narrow_label, _ = comparative_reliability_score(0.08, reference)
    wide_score, wide_label, _ = comparative_reliability_score(0.45, reference)
    assert narrow_score > wide_score
    assert narrow_label == "HIGH"
    assert wide_label == "LOW"
