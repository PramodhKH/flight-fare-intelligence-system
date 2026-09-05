from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_fare_intelligence.benchmarking import build_xgboost_pipeline
from flight_fare_intelligence.explainability import (
    categorical_effects,
    explain_raw_features,
    global_importance,
    local_explanation_table,
    segment_importance,
    select_representative_index,
    transformed_feature_groups,
)
from flight_fare_intelligence.schema import MODEL_FEATURES


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "airline": ["Vistara", "Indigo", "Air_India", "Vistara", "Indigo", "Air_India"],
            "source_city": ["Delhi", "Delhi", "Mumbai", "Delhi", "Mumbai", "Delhi"],
            "destination_city": [
                "Mumbai",
                "Bangalore",
                "Delhi",
                "Mumbai",
                "Delhi",
                "Bangalore",
            ],
            "departure_time": ["Morning", "Evening", "Night", "Morning", "Evening", "Night"],
            "stops": ["zero", "one", "one", "zero", "one", "one"],
            "class": ["Economy", "Economy", "Business", "Business", "Economy", "Business"],
            "duration": [2.0, 3.5, 5.0, 2.2, 4.0, 5.5],
            "days_left": [20, 10, 3, 5, 30, 2],
        }
    )


def _target() -> pd.Series:
    return pd.Series([6_000.0, 8_000.0, 48_000.0, 52_000.0, 7_000.0, 55_000.0])


def _tiny_xgb_pipeline():
    model = build_xgboost_pipeline(
        {
            "n_estimators": 8,
            "max_depth": 2,
            "learning_rate": 0.2,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "n_jobs": 1,
        },
        random_seed=42,
    )
    model.fit(_features(), _target())
    return model


def test_transformed_feature_groups_maps_all_deployed_features() -> None:
    names = [
        "categorical__airline_Vistara",
        "categorical__source_city_Delhi",
        "categorical__destination_city_Mumbai",
        "categorical__departure_time_Morning",
        "categorical__stops_zero",
        "categorical__class_Economy",
        "numeric__duration",
        "numeric__days_left",
    ]
    groups = transformed_feature_groups(names)
    assert set(groups) == set(MODEL_FEATURES)
    assert all(len(indices) == 1 for indices in groups.values())


def test_explain_raw_features_preserves_prediction_additivity() -> None:
    model = _tiny_xgb_pipeline()
    features = _features().iloc[:3].copy()
    explanation = explain_raw_features(model, features, approximate=True)
    predictions = model.predict(features)
    reconstructed = explanation.expected_value + explanation.raw_values.sum(axis=1)
    assert np.allclose(reconstructed.to_numpy(), predictions, atol=1.0)
    assert explanation.max_reconstruction_error < 1.0


def test_global_importance_returns_all_raw_features() -> None:
    raw_shap = pd.DataFrame(
        {feature: np.arange(1, 5, dtype=float) for feature in MODEL_FEATURES}
    )
    importance = global_importance(raw_shap)
    assert set(importance["feature"]) == set(MODEL_FEATURES)
    assert importance["importance_percent"].sum() == pytest.approx(100.0)


def test_segment_importance_scores_requested_masks() -> None:
    raw_shap = pd.DataFrame(
        {feature: np.arange(1, 5, dtype=float) for feature in MODEL_FEATURES}
    )
    metadata = pd.DataFrame({"class": ["Economy", "Business", "Economy", "Business"]})
    result = segment_importance(
        raw_shap,
        metadata,
        {
            "economy": metadata["class"].eq("Economy"),
            "business": metadata["class"].eq("Business"),
        },
    )
    assert set(result["segment"]) == {"economy", "business"}
    assert len(result) == 2 * len(MODEL_FEATURES)


def test_categorical_effects_summarizes_observed_values() -> None:
    features = _features().reset_index(drop=True)
    raw_shap = pd.DataFrame(
        {feature: np.ones(len(features), dtype=float) for feature in MODEL_FEATURES}
    )
    result = categorical_effects(features, raw_shap, ["airline", "class"], min_rows=1)
    assert set(result["feature"]) == {"airline", "class"}
    assert "Vistara" in set(result["value"])


def test_representative_index_selects_median_error_case() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [5, 6, 7],
            "absolute_error": [100.0, 200.0, 1_000.0],
        }
    )
    selected = select_representative_index(frame, np.array([True, True, True]))
    assert selected == 1


def test_local_explanation_table_ranks_largest_contribution() -> None:
    features = _features().iloc[0]
    shap_row = pd.Series({feature: 10.0 for feature in MODEL_FEATURES})
    shap_row["class"] = -500.0
    table = local_explanation_table(
        case_label="example",
        record_id=10,
        feature_row=features,
        shap_row=shap_row,
        expected_value=20_000.0,
        predicted_price=19_570.0,
        actual_price=19_500.0,
    )
    assert table.iloc[0]["feature"] == "class"
    assert table.iloc[0]["direction"] == "decreases_fare"
    assert table.iloc[0]["rank"] == 1
