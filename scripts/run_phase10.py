"""Run the final held-out evaluation and Phase 10 portfolio completion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flight_fare_intelligence.data import load_raw_dataset
from flight_fare_intelligence.modeling import (
    attach_split_assignments,
    prediction_frame,
    regression_metrics,
)
from flight_fare_intelligence.reliability import BOOKING_HORIZON_BINS, BOOKING_HORIZON_LABELS
from flight_fare_intelligence.schema import MODEL_FEATURES, TARGET_COLUMN
from flight_fare_intelligence.uncertainty import evaluate_intervals


def _round_mapping(payload: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), digits)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded


def _require_phase9_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 9 report: {path}. Run `make phase9` first.")
    report = json.loads(path.read_text())
    if report.get("smoke_status") != "passed":
        raise RuntimeError("Phase 10 requires a passing Phase 9 dashboard/API smoke gate")
    if report.get("test_set_scored") is not False:
        raise RuntimeError("The test split must remain sealed until the Phase 10 final evaluation")
    return report


def _segment_metrics(
    frame: pd.DataFrame,
    prediction: pd.Series,
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, segment in frame.groupby(dimension, observed=True):
        segment_prediction = prediction.loc[segment.index].to_numpy(dtype=float)
        metrics = regression_metrics(segment[TARGET_COLUMN], segment_prediction).to_dict()
        rows.append(
            {
                "dimension": dimension,
                "segment": str(value),
                "rows": len(segment),
                **_round_mapping(metrics),
            }
        )
    return rows


def _coverage_by_class(
    test: pd.DataFrame,
    intervals: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_value, segment in test.groupby("class", observed=True):
        metrics = evaluate_intervals(segment[TARGET_COLUMN], intervals.loc[segment.index])
        rows.append(
            {
                "class": str(class_value),
                **_round_mapping(metrics),
            }
        )
    return rows


def _tail_metrics(
    test: pd.DataFrame,
    prediction: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    segment = test.loc[test[TARGET_COLUMN] >= threshold]
    segment_prediction = prediction.loc[segment.index]
    metrics = regression_metrics(segment[TARGET_COLUMN], segment_prediction.to_numpy()).to_dict()
    bias = float((segment_prediction - segment[TARGET_COLUMN]).mean())
    underprediction_rate = float((segment_prediction < segment[TARGET_COLUMN]).mean() * 100.0)
    return {
        "rows": len(segment),
        **_round_mapping(metrics),
        "mean_bias": round(bias, 4),
        "underprediction_rate": round(underprediction_rate, 4),
    }


def _write_figures(
    test: pd.DataFrame,
    prediction: pd.Series,
    class_coverage: list[dict[str, Any]],
) -> list[str]:
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    rng = np.random.default_rng(42)
    sample_size = min(8_000, len(test))
    sample_index = rng.choice(test.index.to_numpy(), size=sample_size, replace=False)

    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    axis.scatter(
        test.loc[sample_index, TARGET_COLUMN],
        prediction.loc[sample_index],
        s=9,
        alpha=0.22,
    )
    minimum = float(test[TARGET_COLUMN].min())
    maximum = float(test[TARGET_COLUMN].max())
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=1.2)
    axis.set_title("Final Held-Out Test: Actual vs Predicted Fare")
    axis.set_xlabel("Actual fare (INR)")
    axis.set_ylabel("Predicted fare (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase10_test_actual_vs_predicted.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    generated.append(str(path))

    coverage_frame = pd.DataFrame(class_coverage)
    figure, axis = plt.subplots(figsize=(7, 4.6))
    bars = axis.bar(coverage_frame["class"], coverage_frame["coverage_percent"])
    axis.axhline(90.0, linestyle="--", linewidth=1.2, label="90% target")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Empirical coverage (%)")
    axis.set_title("Final Held-Out Test: 90% Interval Coverage by Cabin Class")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, coverage_frame["coverage_percent"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    path = figures_dir / "phase10_test_interval_coverage_by_class.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    generated.append(str(path))

    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=Path("data/raw/Flight_Booking.csv"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/processed/phase2_split_assignments.csv"),
    )
    parser.add_argument(
        "--champion-model",
        type=Path,
        default=Path("models/phase4_champion.joblib"),
    )
    parser.add_argument(
        "--intelligence-bundle",
        type=Path,
        default=Path("models/phase7_intelligence_bundle.joblib"),
    )
    parser.add_argument(
        "--phase9-report",
        type=Path,
        default=Path("reports/metrics/phase9_dashboard_smoke.json"),
    )
    args = parser.parse_args()

    for required in [
        args.data_path,
        args.assignments,
        args.champion_model,
        args.intelligence_bundle,
        args.phase9_report,
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required Phase 10 input: {required}")

    _require_phase9_gate(args.phase9_report)

    data = load_raw_dataset(args.data_path)
    assignments = pd.read_csv(args.assignments)
    modeling = attach_split_assignments(data, assignments)
    test = modeling.loc[modeling["split"] == "test"].copy()
    if len(test) != 45_009:
        raise RuntimeError(f"Expected 45,009 held-out test rows, found {len(test):,}")

    model = joblib.load(args.champion_model)
    intelligence = joblib.load(args.intelligence_bundle)
    calibrator = intelligence["calibrator"]

    prediction_array = np.asarray(model.predict(test[MODEL_FEATURES]), dtype=float)
    prediction = pd.Series(prediction_array, index=test.index, name="predicted_price")
    metrics = regression_metrics(test[TARGET_COLUMN], prediction_array).to_dict()

    intervals = calibrator.interval_frame(test[MODEL_FEATURES], prediction_array)
    intervals.index = test.index
    interval_metrics = evaluate_intervals(test[TARGET_COLUMN], intervals)
    class_coverage = _coverage_by_class(test, intervals)

    booking_horizon = pd.cut(
        test["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    )
    test["booking_horizon"] = booking_horizon.astype(str)

    segment_rows = _segment_metrics(test, prediction, dimension="class")
    segment_rows.extend(_segment_metrics(test, prediction, dimension="booking_horizon"))

    predictions = prediction_frame(
        test["record_id"],
        test[TARGET_COLUMN],
        prediction_array,
    )
    predictions["prediction_lower"] = intervals["prediction_lower"].to_numpy(dtype=float)
    predictions["prediction_upper"] = intervals["prediction_upper"].to_numpy(dtype=float)
    predictions["interval_covered"] = (
        predictions["actual_price"] >= predictions["prediction_lower"]
    ) & (predictions["actual_price"] <= predictions["prediction_upper"])
    prediction_path = Path("reports/predictions/phase10_test_predictions.csv")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)

    figures = _write_figures(test, prediction, class_coverage)

    summary = {
        "phase": 10,
        "status": "complete",
        "champion": "xgboost",
        "final_evaluation_policy": {
            "test_rows": len(test),
            "test_used_for_training_or_tuning": False,
            "test_first_scored_in_phase10": True,
            "post_test_retuning_allowed": False,
        },
        "held_out_test_metrics": _round_mapping(metrics),
        "held_out_interval_metrics": _round_mapping(interval_metrics),
        "held_out_interval_coverage_by_class": class_coverage,
        "held_out_segment_metrics": segment_rows,
        "held_out_tail_reliability": {
            "fare_80k_plus": _tail_metrics(test, prediction, threshold=80_000.0),
            "fare_100k_plus": _tail_metrics(test, prediction, threshold=100_000.0),
        },
        "portfolio": {
            "dataset_rows": 300_153,
            "directed_routes": 30,
            "production_features": list(MODEL_FEATURES),
            "api": "FastAPI",
            "frontend": "Streamlit",
            "explainability": "SHAP",
            "uncertainty": "tail-aware hierarchical asymmetric split conformal",
        },
        "artifacts": {
            "test_predictions": str(prediction_path),
            "figures": figures,
        },
    }

    report_path = Path("reports/metrics/phase10_portfolio_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"report": str(report_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
