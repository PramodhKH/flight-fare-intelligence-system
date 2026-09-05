"""Phase 7 fare intelligence, contextual scoring, and what-if simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .reliability import BOOKING_HORIZON_BINS, BOOKING_HORIZON_LABELS
from .schema import MODEL_FEATURES

FARE_POSITION_LABELS = [
    (80, "EXCELLENT_VALUE"),
    (60, "GOOD_VALUE"),
    (40, "TYPICAL"),
    (20, "ABOVE_TYPICAL"),
    (0, "EXPENSIVE"),
]


@dataclass
class ComparableFareIndex:
    """Historical comparable-fare distributions keyed by route/class/horizon."""

    values: dict[tuple[str, str, str], np.ndarray]

    def lookup(
        self,
        scenario: pd.Series | dict[str, Any],
    ) -> tuple[np.ndarray, tuple[str, str, str]]:
        """Return historical comparable fares and the resolved benchmark key."""
        source = str(scenario["source_city"])
        destination = str(scenario["destination_city"])
        cabin_class = str(scenario["class"])
        horizon = booking_horizon_label(int(scenario["days_left"]))
        key = (f"{source}>{destination}", cabin_class, horizon)
        comparable = self.values.get(key)
        if comparable is None or len(comparable) == 0:
            raise KeyError(f"No historical fare benchmark for {key}")
        return comparable, key


def booking_horizon_label(days_left: int) -> str:
    """Return the canonical Phase 2/5 booking-horizon label."""
    value = int(days_left)
    if not 1 <= value <= 49:
        raise ValueError("days_left must be between 1 and 49")
    if value <= 7:
        return "1-7"
    if value <= 14:
        return "8-14"
    if value <= 21:
        return "15-21"
    if value <= 35:
        return "22-35"
    return "36-49"


def build_comparable_fare_index(training_frame: pd.DataFrame) -> ComparableFareIndex:
    """Build route/class/booking-horizon empirical fare distributions."""
    required = {"source_city", "destination_city", "class", "days_left", "price"}
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")

    working = training_frame.copy()
    working["route"] = (
        working["source_city"].astype(str) + ">" + working["destination_city"].astype(str)
    )
    working["booking_horizon"] = pd.cut(
        working["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    ).astype(str)

    values: dict[tuple[str, str, str], np.ndarray] = {}
    group_columns = ["route", "class", "booking_horizon"]
    for key_value, segment in working.groupby(group_columns, observed=True):
        key = tuple(str(item) for item in key_value)
        values[key] = np.sort(segment["price"].to_numpy(dtype=float))
    if not values:
        raise RuntimeError("Comparable fare index is empty")
    return ComparableFareIndex(values=values)


def benchmark_table(index: ComparableFareIndex) -> pd.DataFrame:
    """Return compact quantile summaries for all comparable-fare groups."""
    rows: list[dict[str, float | int | str]] = []
    for (route, cabin_class, horizon), fares in sorted(index.values.items()):
        rows.append(
            {
                "route": route,
                "class": cabin_class,
                "booking_horizon": horizon,
                "rows": len(fares),
                "q10": float(np.quantile(fares, 0.10)),
                "q25": float(np.quantile(fares, 0.25)),
                "median": float(np.median(fares)),
                "mean": float(np.mean(fares)),
                "q75": float(np.quantile(fares, 0.75)),
                "q90": float(np.quantile(fares, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def fare_opportunity(
    estimated_fare: float,
    scenario: pd.Series | dict[str, Any],
    index: ComparableFareIndex,
) -> dict[str, float | int | str]:
    """Contextualize a fare against historical comparable fares."""
    if estimated_fare < 0.0:
        raise ValueError("estimated_fare must be non-negative")
    comparable, key = index.lookup(scenario)
    percentile = float(
        np.searchsorted(comparable, estimated_fare, side="right") / len(comparable) * 100.0
    )
    score = round(float(np.clip(100.0 - percentile, 0.0, 100.0)))
    position = next(label for threshold, label in FARE_POSITION_LABELS if score >= threshold)
    median = float(np.median(comparable))
    difference = estimated_fare - median
    difference_percent = difference / median * 100.0 if median > 0.0 else 0.0
    return {
        "fare_opportunity_score": score,
        "fare_position": position,
        "historical_percentile": percentile,
        "benchmark_rows": len(comparable),
        "benchmark_route": key[0],
        "benchmark_class": key[1],
        "benchmark_horizon": key[2],
        "benchmark_median": median,
        "difference_from_median": difference,
        "difference_from_median_percent": difference_percent,
    }


def scenario_frame(scenario: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Convert one scenario into the exact deployed model feature frame."""
    values = dict(scenario)
    missing = [feature for feature in MODEL_FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing scenario features: {missing}")
    return pd.DataFrame([{feature: values[feature] for feature in MODEL_FEATURES}])


def predict_scenario(pipeline: Pipeline, scenario: pd.Series | dict[str, Any]) -> float:
    """Predict one scenario using the locked deployment-aligned pipeline."""
    prediction = np.asarray(pipeline.predict(scenario_frame(scenario)), dtype=float)
    return float(max(0.0, prediction[0]))


def counterfactual_table(
    pipeline: Pipeline,
    scenario: pd.Series | dict[str, Any],
    *,
    feature: str,
    values: list[Any],
) -> pd.DataFrame:
    """Vary one deployed feature while holding every other input fixed."""
    if feature not in MODEL_FEATURES:
        raise ValueError(f"Unknown deployed feature: {feature}")
    if not values:
        raise ValueError("Counterfactual values cannot be empty")

    base = dict(scenario)
    missing = [name for name in MODEL_FEATURES if name not in base]
    if missing:
        raise ValueError(f"Missing scenario features: {missing}")
    rows: list[dict[str, Any]] = []
    for value in values:
        changed = dict(base)
        changed[feature] = value
        prediction = predict_scenario(pipeline, changed)
        rows.append(
            {
                "changed_feature": feature,
                "changed_value": value,
                "predicted_fare": prediction,
            }
        )
    return pd.DataFrame(rows)


def days_left_curve(
    pipeline: Pipeline,
    scenario: pd.Series | dict[str, Any],
    *,
    min_days: int = 1,
    max_days: int = 49,
) -> pd.DataFrame:
    """Simulate expected fare across the valid booking-horizon range."""
    if not 1 <= min_days <= max_days <= 49:
        raise ValueError("days-left curve must stay inside 1..49")
    values = list(range(min_days, max_days + 1))
    result = counterfactual_table(pipeline, scenario, feature="days_left", values=values)
    return result.rename(columns={"changed_value": "days_left"}).drop(columns="changed_feature")


def booking_guidance(
    *,
    current_days_left: int,
    current_fare: float,
    opportunity_score: int,
    curve: pd.DataFrame,
    wait_window_days: int = 7,
    material_change_percent: float = 5.0,
) -> dict[str, float | int | str]:
    """Generate cautious historical/model-based Buy Now vs Monitor guidance."""
    if current_days_left < 1:
        raise ValueError("current_days_left must be positive")
    if current_fare <= 0.0:
        raise ValueError("current_fare must be positive")
    if wait_window_days < 1:
        raise ValueError("wait_window_days must be positive")

    future_days = list(range(max(1, current_days_left - wait_window_days), current_days_left))
    future = curve[curve["days_left"].isin(future_days)]
    if future.empty:
        return {
            "recommendation": "BUY_NOW",
            "wait_window_days": 0,
            "future_median_predicted_fare": current_fare,
            "expected_wait_change": 0.0,
            "expected_wait_change_percent": 0.0,
            "rationale_code": "departure_imminent",
        }

    future_median = float(future["predicted_fare"].median())
    change = future_median - current_fare
    change_percent = change / current_fare * 100.0

    if change_percent >= material_change_percent:
        if opportunity_score >= 40:
            recommendation = "BUY_NOW"
            rationale = "model_indicates_rising_fares_and_current_fare_not_expensive"
        else:
            recommendation = "MONITOR"
            rationale = "rising_fare_signal_but_current_fare_is_above_typical"
    elif change_percent <= -material_change_percent:
        if opportunity_score < 80:
            recommendation = "WAIT_OR_MONITOR"
            rationale = "model_indicates_potential_near_term_softening"
        else:
            recommendation = "MONITOR"
            rationale = "softening_signal_but_current_fare_is_already_excellent_value"
    else:
        recommendation = "MONITOR"
        rationale = "model_change_is_not_material"

    return {
        "recommendation": recommendation,
        "wait_window_days": min(wait_window_days, current_days_left - 1),
        "future_median_predicted_fare": future_median,
        "expected_wait_change": change,
        "expected_wait_change_percent": change_percent,
        "rationale_code": rationale,
    }
