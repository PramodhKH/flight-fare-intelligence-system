"""Phase 5 reliability, segment-error, and robustness diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

BOOKING_HORIZON_BINS = [0, 7, 14, 21, 35, 49]
BOOKING_HORIZON_LABELS = ["1-7", "8-14", "15-21", "22-35", "36-49"]
FARE_BINS = [0, 10_000, 20_000, 40_000, 60_000, 80_000, np.inf]
FARE_LABELS = ["<10k", "10-20k", "20-40k", "40-60k", "60-80k", "80k+"]
SUPPORT_CONTEXT = [
    "airline",
    "source_city",
    "destination_city",
    "class",
    "departure_time",
    "stops",
]
SUPPORT_BINS = [-1, 0, 25, 100, np.inf]
SUPPORT_LABELS = ["unseen", "low_1_25", "medium_26_100", "high_101_plus"]


def _metric_payload(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int | None]:
    """Return canonical reliability metrics for one evaluation segment."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residual = predicted_array - actual_array
    absolute_error = np.abs(residual)

    if len(actual_array) == 0:
        raise ValueError("Cannot score an empty segment")

    r2: float | None = None
    if len(actual_array) >= 2 and float(np.var(actual_array)) > 0.0:
        r2 = float(r2_score(actual_array, predicted_array))

    return {
        "rows": len(actual_array),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "r2": r2,
        "mape": float(mean_absolute_percentage_error(actual_array, predicted_array) * 100.0),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "mean_bias": float(np.mean(residual)),
        "underprediction_rate": float(np.mean(predicted_array < actual_array) * 100.0),
    }


def score_segment(
    frame: pd.DataFrame,
    *,
    actual_column: str = "actual_price",
    prediction_column: str = "predicted_price",
) -> dict[str, float | int | None]:
    """Score one already-filtered validation segment."""
    return _metric_payload(frame[actual_column], frame[prediction_column])


def add_market_support(
    evaluation: pd.DataFrame,
    training: pd.DataFrame,
    *,
    context_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Attach training support counts for each categorical market context."""
    context = context_columns or SUPPORT_CONTEXT
    missing_train = [column for column in context if column not in training.columns]
    missing_eval = [column for column in context if column not in evaluation.columns]
    if missing_train or missing_eval:
        raise ValueError(
            f"Missing support columns; training={missing_train}, evaluation={missing_eval}"
        )

    counts = (
        training.groupby(context, observed=True)
        .size()
        .rename("training_context_rows")
        .reset_index()
    )
    enriched = evaluation.merge(counts, on=context, how="left", validate="many_to_one")
    enriched["training_context_rows"] = enriched["training_context_rows"].fillna(0).astype(int)
    enriched["support_bucket"] = pd.cut(
        enriched["training_context_rows"],
        bins=SUPPORT_BINS,
        labels=SUPPORT_LABELS,
        include_lowest=True,
    ).astype(str)
    return enriched


def build_reliability_frame(
    *,
    validation_features: pd.DataFrame,
    validation_record_ids: pd.Series,
    actual_price: pd.Series,
    predicted_price: np.ndarray,
    training_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical Phase 5 validation diagnostics frame."""
    if not (
        len(validation_features)
        == len(validation_record_ids)
        == len(actual_price)
        == len(predicted_price)
    ):
        raise ValueError("Validation inputs must have identical row counts")

    frame = validation_features.reset_index(drop=True).copy()
    frame.insert(0, "record_id", validation_record_ids.reset_index(drop=True).to_numpy())
    frame["actual_price"] = actual_price.reset_index(drop=True).to_numpy(dtype=float)
    frame["predicted_price"] = np.asarray(predicted_price, dtype=float)
    frame["residual"] = frame["predicted_price"] - frame["actual_price"]
    frame["absolute_error"] = frame["residual"].abs()
    frame["route"] = frame["source_city"].astype(str) + ">" + frame["destination_city"].astype(str)
    frame["booking_horizon"] = pd.cut(
        frame["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    ).astype(str)
    frame["fare_band"] = pd.cut(
        frame["actual_price"],
        bins=FARE_BINS,
        labels=FARE_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    return add_market_support(frame, training_frame)


def segment_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    min_rows: int = 1,
) -> pd.DataFrame:
    """Compute reliability metrics for one or more segmentation dimensions."""
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"Missing segmentation column: {column}")
        for value, segment in frame.groupby(column, observed=True, dropna=False):
            if len(segment) < min_rows:
                continue
            rows.append(
                {
                    "dimension": column,
                    "segment": str(value),
                    **score_segment(segment),
                }
            )
    return pd.DataFrame(rows)


def stress_test_table(frame: pd.DataFrame, *, long_duration_threshold: float) -> pd.DataFrame:
    """Score predefined high-risk segments without altering the champion model."""
    segments: dict[str, pd.Series] = {
        "overall_validation": pd.Series(True, index=frame.index),
        "economy": frame["class"].eq("Economy"),
        "business": frame["class"].eq("Business"),
        "last_minute_1_7_days": frame["days_left"].between(1, 7),
        "high_fare_80k_plus": frame["actual_price"].ge(80_000),
        "very_high_fare_100k_plus": frame["actual_price"].ge(100_000),
        "sparse_context_25_or_less": frame["training_context_rows"].between(1, 25),
        "unseen_market_context": frame["training_context_rows"].eq(0),
        "long_duration_p90_plus": frame["duration"].ge(long_duration_threshold),
    }

    rows: list[dict[str, Any]] = []
    for name, mask in segments.items():
        segment = frame.loc[mask]
        if segment.empty:
            rows.append({"stress_segment": name, "rows": 0})
            continue
        rows.append({"stress_segment": name, **score_segment(segment)})
    return pd.DataFrame(rows)


def worst_error_cases(frame: pd.DataFrame, *, limit: int = 50) -> pd.DataFrame:
    """Return the highest absolute validation errors for manual diagnostics."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    columns = [
        "record_id",
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "stops",
        "class",
        "duration",
        "days_left",
        "actual_price",
        "predicted_price",
        "residual",
        "absolute_error",
        "fare_band",
        "booking_horizon",
        "training_context_rows",
        "support_bucket",
    ]
    return frame.nlargest(limit, "absolute_error")[columns].reset_index(drop=True)
