"""Reusable regression training and evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import MODEL_FEATURES, TARGET_COLUMN

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "stops",
    "class",
]
NUMERIC_FEATURES = ["duration", "days_left"]


@dataclass(frozen=True)
class RegressionMetrics:
    """Canonical regression metrics for model comparison."""

    rmse: float
    mae: float
    r2: float
    mape: float
    median_absolute_error: float
    p90_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def build_linear_regression_pipeline() -> Pipeline:
    """Build the Phase 3 deployment-aligned Linear Regression baseline."""
    categorical_pipeline = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
    )
    numeric_pipeline = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def attach_split_assignments(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the definitive Phase 2 split to every raw record."""
    required_assignment_columns = {"record_id", "scenario_id", "split"}
    missing_assignments = sorted(required_assignment_columns - set(assignments.columns))
    if missing_assignments:
        raise ValueError(f"Missing split assignment columns: {missing_assignments}")

    missing_model_columns = [
        column for column in MODEL_FEATURES + [TARGET_COLUMN] if column not in df.columns
    ]
    if missing_model_columns:
        raise ValueError(f"Missing modeling columns: {missing_model_columns}")

    record_ids = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else pd.Series(df.index)
    modeling_frame = df[MODEL_FEATURES + [TARGET_COLUMN]].copy()
    modeling_frame.insert(0, "record_id", record_ids.to_numpy())

    merged = modeling_frame.merge(
        assignments[["record_id", "scenario_id", "split"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Some dataset rows are missing Phase 2 split assignments")
    if len(merged) != len(df):
        raise RuntimeError("Split merge changed the dataset row count")
    return merged


def split_xy(
    modeling_frame: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return X, y, and record ids for one named split."""
    subset = modeling_frame[modeling_frame["split"] == split_name]
    if subset.empty:
        raise ValueError(f"Split '{split_name}' is empty")
    return (
        subset[MODEL_FEATURES].copy(),
        subset[TARGET_COLUMN].copy(),
        subset["record_id"].copy(),
    )


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate the canonical Phase 3 regression metric set."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(truth - prediction)

    return RegressionMetrics(
        rmse=float(np.sqrt(mean_squared_error(truth, prediction))),
        mae=float(mean_absolute_error(truth, prediction)),
        r2=float(r2_score(truth, prediction)),
        mape=float(mean_absolute_percentage_error(truth, prediction) * 100.0),
        median_absolute_error=float(np.median(absolute_error)),
        p90_absolute_error=float(np.quantile(absolute_error, 0.90)),
    )


def prediction_frame(
    record_ids: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return record-level predictions and residual diagnostics."""
    result = pd.DataFrame(
        {
            "record_id": record_ids.to_numpy(),
            "actual_price": y_true.to_numpy(dtype=float),
            "predicted_price": np.asarray(y_pred, dtype=float),
        }
    )
    result["residual"] = result["actual_price"] - result["predicted_price"]
    result["absolute_error"] = result["residual"].abs()
    result["absolute_percentage_error"] = result["absolute_error"] / result["actual_price"] * 100.0
    return result


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Return deployment-feature coefficients from a fitted linear pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(feature_names) != len(coefficients):
        raise RuntimeError("Coefficient count does not match transformed feature count")

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted model pipeline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination


def model_metadata(pipeline: Pipeline) -> dict[str, Any]:
    """Return compact metadata for a fitted Linear Regression pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_features = preprocessor.get_feature_names_out()
    return {
        "model_type": type(pipeline.named_steps["model"]).__name__,
        "raw_features": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "transformed_feature_count": len(transformed_features),
    }
