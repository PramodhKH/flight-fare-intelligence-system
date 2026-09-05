from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_fare_intelligence.benchmarking import (
    build_catboost_model,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    measure_inference_latency,
    select_champion,
)


def _tiny_frame() -> tuple[pd.DataFrame, pd.Series]:
    rows = 36
    frame = pd.DataFrame(
        {
            "airline": ["Vistara", "Indigo", "Air_India"] * 12,
            "source_city": ["Delhi", "Mumbai", "Bangalore"] * 12,
            "destination_city": ["Mumbai", "Delhi", "Chennai"] * 12,
            "departure_time": ["Morning", "Evening", "Afternoon"] * 12,
            "stops": ["zero", "one", "zero"] * 12,
            "class": ["Economy", "Economy", "Business"] * 12,
            "duration": np.linspace(1.5, 8.0, rows),
            "days_left": np.tile(np.arange(1, 13), 3),
        }
    )
    target = pd.Series(2_500 + frame["duration"] * 500 + frame["days_left"] * 30)
    return frame, target


def test_random_forest_pipeline_fits_and_predicts() -> None:
    features, target = _tiny_frame()
    model = build_random_forest_pipeline(
        {
            "n_estimators": 5,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "max_features": 0.8,
            "max_samples": 0.8,
            "n_jobs": 1,
        },
        random_seed=42,
    )
    model.fit(features, target)
    predictions = model.predict(features.iloc[:4])
    assert predictions.shape == (4,)


def test_xgboost_pipeline_fits_and_predicts() -> None:
    features, target = _tiny_frame()
    model = build_xgboost_pipeline(
        {
            "n_estimators": 8,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "n_jobs": 1,
        },
        random_seed=42,
    )
    model.fit(features, target)
    predictions = model.predict(features.iloc[:4])
    assert predictions.shape == (4,)


def test_catboost_model_fits_and_predicts() -> None:
    features, target = _tiny_frame()
    model = build_catboost_model(
        {
            "iterations": 5,
            "depth": 3,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "thread_count": 1,
        },
        random_seed=42,
    )
    model.fit(features, target)
    predictions = model.predict(features.iloc[:4])
    assert len(predictions) == 4


def test_latency_benchmark_rejects_invalid_repeats() -> None:
    features, target = _tiny_frame()
    model = build_random_forest_pipeline(
        {
            "n_estimators": 2,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "max_samples": 1.0,
            "n_jobs": 1,
        },
        random_seed=42,
    )
    model.fit(features, target)
    with pytest.raises(ValueError, match="repeats"):
        measure_inference_latency(model, features, random_seed=42, repeats=0)


def test_champion_selection_uses_rmse_first() -> None:
    comparison = pd.DataFrame(
        [
            {
                "model": "fast_model",
                "rmse": 2_600.0,
                "mae": 1_200.0,
                "median_batch_ms": 2.0,
                "model_size_mb": 1.0,
            },
            {
                "model": "accurate_model",
                "rmse": 2_500.0,
                "mae": 1_300.0,
                "median_batch_ms": 20.0,
                "model_size_mb": 20.0,
            },
        ]
    )
    assert select_champion(comparison) == "accurate_model"
