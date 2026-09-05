"""Phase 7 uncertainty calibration and comparative reliability utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .reliability import BOOKING_HORIZON_BINS, BOOKING_HORIZON_LABELS

PREDICTED_FARE_BINS = [0, 10_000, 20_000, 40_000, 60_000, 80_000, np.inf]
PREDICTED_FARE_LABELS = ["<10k", "10-20k", "20-40k", "40-60k", "60-80k", "80k+"]


@dataclass(frozen=True)
class ResidualIntervalRule:
    """One residual-quantile rule used by the hierarchical interval calibrator."""

    level: str
    key: tuple[str, ...]
    rows: int
    lower_offset: float
    upper_offset: float


@dataclass
class SegmentConformalCalibrator:
    """Hierarchical asymmetric split-conformal residual calibrator."""

    coverage: float
    min_rows: int
    tail_min_rows: int
    tail_upper_quantile: float
    rules: dict[tuple[str, tuple[str, ...]], ResidualIntervalRule]
    global_rule: ResidualIntervalRule

    def interval_frame(
        self,
        features: pd.DataFrame,
        predicted_price: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Return prediction intervals plus calibration provenance for each row."""
        prepared = _prepare_interval_features(features, predicted_price)
        rows: list[dict[str, float | int | str]] = []
        for row in prepared.itertuples(index=False):
            rule = self._select_rule(row)
            point = float(row.predicted_price)
            lower = max(0.0, point + rule.lower_offset)
            upper = max(lower, point + rule.upper_offset)
            rows.append(
                {
                    "prediction_lower": lower,
                    "prediction_upper": upper,
                    "interval_width": upper - lower,
                    "calibration_level": rule.level,
                    "calibration_rows": rule.rows,
                }
            )
        return pd.DataFrame(rows, index=features.index)

    def _select_rule(self, row: object) -> ResidualIntervalRule:
        class_value = str(row.class_value)
        horizon = str(row.booking_horizon)
        fare_band = str(row.predicted_fare_band)
        candidates = [
            ("class_horizon_fare", (class_value, horizon, fare_band)),
            ("class_fare", (class_value, fare_band)),
            ("class_horizon", (class_value, horizon)),
            ("class", (class_value,)),
        ]
        for level, key in candidates:
            rule = self.rules.get((level, key))
            if rule is not None:
                return rule
        return self.global_rule


def deterministic_validation_partition(
    validation_rows: pd.DataFrame,
    *,
    calibration_fraction: float = 0.50,
    random_seed: int = 42,
) -> pd.Series:
    """Partition validation scenarios deterministically for calibration vs evaluation.

    All records sharing the same Phase 2 scenario_id stay on the same side of the
    uncertainty split. The original train/validation/test split is never modified.
    """
    if "scenario_id" not in validation_rows.columns:
        raise ValueError("scenario_id is required for uncertainty partitioning")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    scenario = validation_rows["scenario_id"].astype("uint64")
    modulus = 10_000
    threshold = round(calibration_fraction * modulus)
    mixed = pd.util.hash_pandas_object(
        scenario.astype(str) + f"|{random_seed}",
        index=False,
    ).astype("uint64")
    calibration_mask = (mixed % np.uint64(modulus)) < np.uint64(threshold)

    scenario_side_counts = (
        pd.DataFrame({"scenario_id": scenario, "calibration": calibration_mask})
        .groupby("scenario_id")["calibration"]
        .nunique()
    )
    if int((scenario_side_counts > 1).sum()) != 0:
        raise RuntimeError("A scenario crossed the Phase 7 calibration/evaluation partition")
    if calibration_mask.all() or (~calibration_mask).all():
        raise RuntimeError("Uncertainty partition produced an empty side")
    return pd.Series(calibration_mask.to_numpy(dtype=bool), index=validation_rows.index)


def fit_segment_conformal_calibrator(
    features: pd.DataFrame,
    actual_price: pd.Series | np.ndarray,
    predicted_price: pd.Series | np.ndarray,
    *,
    coverage: float = 0.90,
    min_rows: int = 150,
    tail_min_rows: int = 50,
    tail_upper_quantile: float = 0.995,
) -> SegmentConformalCalibrator:
    """Fit hierarchical asymmetric residual intervals on a calibration subset."""
    if not 0.50 < coverage < 1.0:
        raise ValueError("coverage must be between 0.50 and 1.0")
    if min_rows < 20:
        raise ValueError("min_rows must be at least 20")
    if tail_min_rows < 20:
        raise ValueError("tail_min_rows must be at least 20")
    if not 0.95 <= tail_upper_quantile < 1.0:
        raise ValueError("tail_upper_quantile must be between 0.95 and 1.0")
    prepared = _prepare_interval_features(features, predicted_price)
    actual = np.asarray(actual_price, dtype=float)
    prediction = np.asarray(predicted_price, dtype=float)
    if len(prepared) != len(actual):
        raise ValueError("Calibration features and targets must have identical row counts")
    residual = actual - prediction
    if len(residual) < min_rows:
        raise ValueError("Calibration subset is too small")

    working = prepared.copy()
    working["residual"] = residual
    rules: dict[tuple[str, tuple[str, ...]], ResidualIntervalRule] = {}
    levels = {
        "class_horizon_fare": ["class_value", "booking_horizon", "predicted_fare_band"],
        "class_fare": ["class_value", "predicted_fare_band"],
        "class_horizon": ["class_value", "booking_horizon"],
        "class": ["class_value"],
    }
    for level, columns in levels.items():
        for key_value, segment in working.groupby(columns, observed=True, dropna=False):
            key = _normalize_group_key(key_value)
            tail_risk = _is_high_fare_business_key(key)
            required_rows = tail_min_rows if tail_risk else min_rows
            if len(segment) < required_rows:
                continue
            upper_probability = tail_upper_quantile if tail_risk else None
            lower, upper = _conservative_residual_bounds(
                segment["residual"],
                coverage,
                upper_probability=upper_probability,
            )
            rules[(level, key)] = ResidualIntervalRule(
                level=level,
                key=key,
                rows=len(segment),
                lower_offset=lower,
                upper_offset=upper,
            )

    global_lower, global_upper = _conservative_residual_bounds(working["residual"], coverage)
    global_rule = ResidualIntervalRule(
        level="global",
        key=("global",),
        rows=len(working),
        lower_offset=global_lower,
        upper_offset=global_upper,
    )
    return SegmentConformalCalibrator(
        coverage=coverage,
        min_rows=min_rows,
        tail_min_rows=tail_min_rows,
        tail_upper_quantile=tail_upper_quantile,
        rules=rules,
        global_rule=global_rule,
    )


def evaluate_intervals(
    actual_price: pd.Series | np.ndarray,
    interval_frame: pd.DataFrame,
) -> dict[str, float | int]:
    """Return empirical coverage and interval-width diagnostics."""
    actual = np.asarray(actual_price, dtype=float)
    if len(actual) != len(interval_frame):
        raise ValueError("Actual prices and intervals must have identical row counts")
    lower = interval_frame["prediction_lower"].to_numpy(dtype=float)
    upper = interval_frame["prediction_upper"].to_numpy(dtype=float)
    width = upper - lower
    covered = (actual >= lower) & (actual <= upper)
    below = actual < lower
    above = actual > upper
    return {
        "rows": len(actual),
        "coverage_percent": float(covered.mean() * 100.0),
        "lower_miss_percent": float(below.mean() * 100.0),
        "upper_miss_percent": float(above.mean() * 100.0),
        "mean_interval_width": float(width.mean()),
        "median_interval_width": float(np.median(width)),
        "p90_interval_width": float(np.quantile(width, 0.90)),
    }


def reliability_reference(relative_width: pd.Series | np.ndarray) -> np.ndarray:
    """Return a sorted empirical reference distribution for comparative reliability."""
    values = np.asarray(relative_width, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        raise ValueError("At least 20 relative-width values are required")
    return np.sort(values)


def comparative_reliability_score(
    relative_width: float,
    reference: np.ndarray,
) -> tuple[int, str, float]:
    """Score uncertainty relative to the calibration distribution.

    A higher score means the interval is narrower than most calibrated intervals.
    This is a comparative uncertainty score, not a probability of correctness.
    """
    if relative_width < 0.0 or not np.isfinite(relative_width):
        raise ValueError("relative_width must be finite and non-negative")
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 1 or len(ref) == 0:
        raise ValueError("reference must be a non-empty one-dimensional array")
    percentile = float(np.searchsorted(ref, relative_width, side="right") / len(ref) * 100.0)
    score = round(float(np.clip(100.0 - percentile, 5.0, 95.0)))
    if score >= 70:
        label = "HIGH"
    elif score >= 40:
        label = "MEDIUM"
    else:
        label = "LOW"
    return score, label, percentile


def _prepare_interval_features(
    features: pd.DataFrame,
    predicted_price: pd.Series | np.ndarray,
) -> pd.DataFrame:
    required = {"class", "days_left"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing interval features: {missing}")
    prediction = np.asarray(predicted_price, dtype=float)
    if len(features) != len(prediction):
        raise ValueError("Features and predictions must have identical row counts")

    prepared = pd.DataFrame(index=features.index)
    prepared["class_value"] = features["class"].astype(str)
    prepared["booking_horizon"] = pd.cut(
        features["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    ).astype(str)
    prepared["predicted_fare_band"] = pd.cut(
        prediction,
        bins=PREDICTED_FARE_BINS,
        labels=PREDICTED_FARE_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    prepared["predicted_price"] = prediction
    return prepared


def _conservative_residual_bounds(
    residuals: pd.Series | np.ndarray,
    coverage: float,
    *,
    upper_probability: float | None = None,
) -> tuple[float, float]:
    values = np.sort(np.asarray(residuals, dtype=float))
    if len(values) == 0:
        raise ValueError("Cannot calibrate on an empty residual array")
    alpha = 1.0 - coverage
    lower_probability = alpha / 2.0
    resolved_upper_probability = 1.0 - alpha / 2.0
    if upper_probability is not None:
        resolved_upper_probability = max(resolved_upper_probability, upper_probability)
    lower_rank = max(1, int(np.floor((len(values) + 1) * lower_probability)))
    upper_rank = min(
        len(values),
        int(np.ceil((len(values) + 1) * resolved_upper_probability)),
    )
    return float(values[lower_rank - 1]), float(values[upper_rank - 1])


def _is_high_fare_business_key(key: tuple[str, ...]) -> bool:
    values = set(key)
    return "Business" in values and bool(values & {"60-80k", "80k+"})


def _normalize_group_key(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)
