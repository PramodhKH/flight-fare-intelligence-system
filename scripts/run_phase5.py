#!/usr/bin/env python
"""Run Phase 5 model reliability, segment error analysis, and robustness diagnostics."""

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
from flight_fare_intelligence.modeling import attach_split_assignments, split_xy
from flight_fare_intelligence.reliability import (
    BOOKING_HORIZON_LABELS,
    FARE_LABELS,
    SUPPORT_LABELS,
    build_reliability_frame,
    score_segment,
    segment_table,
    stress_test_table,
    worst_error_cases,
)

SEGMENT_DIMENSIONS = [
    "class",
    "airline",
    "route",
    "booking_horizon",
    "stops",
    "departure_time",
    "fare_band",
    "support_bucket",
]


def _round_payload(payload: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), digits)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded


def _bar_with_values(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: Path,
    order: list[str] | None = None,
) -> None:
    plot_data = data.copy()
    if order is not None:
        rank = {label: index for index, label in enumerate(order)}
        plot_data["_order"] = plot_data[x].map(rank)
        plot_data = plot_data.sort_values("_order")

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(plot_data[x], plot_data[y])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, plot_data[y], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _generate_figures(segments: pd.DataFrame) -> list[str]:
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    fare = segments.loc[segments["dimension"] == "fare_band"]
    path = figures_dir / "phase5_rmse_by_fare_band.png"
    _bar_with_values(
        fare,
        x="segment",
        y="rmse",
        title="XGBoost Validation RMSE by Fare Band",
        ylabel="RMSE (INR)",
        output_path=path,
        order=FARE_LABELS,
    )
    generated.append(str(path))

    fare_bias = fare.copy()
    path = figures_dir / "phase5_bias_by_fare_band.png"
    _bar_with_values(
        fare_bias,
        x="segment",
        y="mean_bias",
        title="XGBoost Mean Prediction Bias by Fare Band",
        ylabel="Mean bias: predicted - actual (INR)",
        output_path=path,
        order=FARE_LABELS,
    )
    generated.append(str(path))

    class_metrics = segments.loc[segments["dimension"] == "class"]
    path = figures_dir / "phase5_rmse_by_class.png"
    _bar_with_values(
        class_metrics,
        x="segment",
        y="rmse",
        title="XGBoost Validation RMSE by Cabin Class",
        ylabel="RMSE (INR)",
        output_path=path,
    )
    generated.append(str(path))

    horizon = segments.loc[segments["dimension"] == "booking_horizon"]
    path = figures_dir / "phase5_rmse_by_booking_horizon.png"
    _bar_with_values(
        horizon,
        x="segment",
        y="rmse",
        title="XGBoost Validation RMSE by Booking Horizon",
        ylabel="RMSE (INR)",
        output_path=path,
        order=BOOKING_HORIZON_LABELS,
    )
    generated.append(str(path))

    support = segments.loc[segments["dimension"] == "support_bucket"]
    path = figures_dir / "phase5_rmse_by_training_support.png"
    _bar_with_values(
        support,
        x="segment",
        y="rmse",
        title="XGBoost RMSE by Training Market-Context Support",
        ylabel="RMSE (INR)",
        output_path=path,
        order=SUPPORT_LABELS,
    )
    generated.append(str(path))

    route = segments.loc[segments["dimension"] == "route"].copy()
    route[["source", "destination"]] = route["segment"].str.split(">", expand=True)
    pivot = route.pivot(index="source", columns="destination", values="rmse")
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    axis.set_title("Route-Level XGBoost Validation RMSE")
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_xlabel("Destination")
    axis.set_ylabel("Source")
    figure.colorbar(image, ax=axis, label="RMSE (INR)")
    figure.tight_layout()
    path = figures_dir / "phase5_route_rmse_heatmap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
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
        "--phase4-report",
        type=Path,
        default=Path("reports/metrics/phase4_model_benchmark.json"),
    )
    parser.add_argument(
        "--runner-up-predictions",
        type=Path,
        default=Path("reports/predictions/phase4_random_forest_validation.csv"),
    )
    parser.add_argument("--worst-cases", type=int, default=50)
    args = parser.parse_args()

    required_inputs = [
        args.assignments,
        args.champion_model,
        args.phase4_report,
        args.runner_up_predictions,
    ]
    for required in required_inputs:
        if not required.exists():
            raise FileNotFoundError(f"Missing required Phase 5 input: {required}")

    phase4_report = json.loads(args.phase4_report.read_text())
    champion_name = str(phase4_report["champion"])
    if champion_name != "xgboost":
        raise RuntimeError(
            f"Phase 5 expects the locked Phase 4 XGBoost champion, found {champion_name!r}"
        )
    if bool(phase4_report.get("test_set_scored", True)):
        raise RuntimeError("Phase 4 report indicates the test set was scored; refusing Phase 5")

    df = load_raw_dataset(args.data_path)
    assignments = pd.read_csv(args.assignments)
    modeling_frame = attach_split_assignments(df, assignments)
    training_frame = modeling_frame.loc[modeling_frame["split"] == "train"].copy()
    validation_features, validation_target, validation_ids = split_xy(
        modeling_frame,
        "validation",
    )
    sealed_test_rows = int((modeling_frame["split"] == "test").sum())

    champion = joblib.load(args.champion_model)
    predictions = np.asarray(champion.predict(validation_features), dtype=float)
    reliability = build_reliability_frame(
        validation_features=validation_features,
        validation_record_ids=validation_ids,
        actual_price=validation_target,
        predicted_price=predictions,
        training_frame=training_frame,
    )

    runner_up_raw = pd.read_csv(args.runner_up_predictions)
    required_runner_columns = {"record_id", "random_forest_prediction"}
    missing_runner_columns = required_runner_columns - set(runner_up_raw.columns)
    if missing_runner_columns:
        raise ValueError(
            f"Runner-up predictions missing columns: {sorted(missing_runner_columns)}"
        )
    runner_lookup = runner_up_raw.set_index("record_id")["random_forest_prediction"]
    runner_predictions = validation_ids.map(runner_lookup)
    if runner_predictions.isna().any():
        raise RuntimeError("Runner-up prediction file does not cover every validation record")
    runner_reliability = build_reliability_frame(
        validation_features=validation_features,
        validation_record_ids=validation_ids,
        actual_price=validation_target,
        predicted_price=runner_predictions.to_numpy(dtype=float),
        training_frame=training_frame,
    )

    segments = segment_table(reliability, SEGMENT_DIMENSIONS)
    long_duration_threshold = float(training_frame["duration"].quantile(0.90))
    stress = stress_test_table(reliability, long_duration_threshold=long_duration_threshold)
    stress.insert(0, "model", "xgboost")
    runner_stress = stress_test_table(
        runner_reliability,
        long_duration_threshold=long_duration_threshold,
    )
    runner_stress.insert(0, "model", "random_forest")
    champion_vs_runner = pd.concat([stress, runner_stress], ignore_index=True)
    worst = worst_error_cases(reliability, limit=args.worst_cases)

    metrics_dir = Path("reports/metrics")
    predictions_dir = Path("reports/predictions")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    segment_path = predictions_dir / "phase5_segment_metrics.csv"
    stress_path = predictions_dir / "phase5_stress_tests.csv"
    comparison_path = predictions_dir / "phase5_champion_vs_runner_up.csv"
    diagnostics_path = predictions_dir / "phase5_validation_diagnostics.csv"
    worst_path = predictions_dir / "phase5_worst_errors.csv"
    segments.to_csv(segment_path, index=False)
    stress.to_csv(stress_path, index=False)
    champion_vs_runner.to_csv(comparison_path, index=False)
    reliability.to_csv(diagnostics_path, index=False)
    worst.to_csv(worst_path, index=False)
    figures = _generate_figures(segments)

    overall = score_segment(reliability)
    high_fare = reliability.loc[reliability["actual_price"] >= 80_000]
    very_high_fare = reliability.loc[reliability["actual_price"] >= 100_000]
    sparse = reliability.loc[reliability["training_context_rows"] <= 25]
    unseen = reliability.loc[reliability["training_context_rows"] == 0]

    report = {
        "phase": 5,
        "champion": champion_name,
        "evaluation_split": "validation",
        "test_set_scored": False,
        "rows": {
            "train_reference": len(training_frame),
            "validation_scored": len(reliability),
            "test_sealed": sealed_test_rows,
        },
        "overall_validation": _round_payload(overall),
        "tail_reliability": {
            "fare_80k_plus": _round_payload(score_segment(high_fare)),
            "fare_100k_plus": _round_payload(score_segment(very_high_fare)),
        },
        "runner_up_challenge": {
            "runner_up": "random_forest",
            "policy": (
                "Compare locked models on predefined stress segments without retuning. "
                "RMSE remains the primary reliability criterion."
            ),
            "xgboost_overall_rmse": round(float(score_segment(reliability)["rmse"]), 4),
            "random_forest_overall_rmse": round(
                float(score_segment(runner_reliability)["rmse"]),
                4,
            ),
        },
        "market_context_support": {
            "definition": (
                "training count for airline + source + destination + class + "
                "departure_time + stops"
            ),
            "sparse_threshold_rows": 25,
            "sparse_validation_rows": len(sparse),
            "unseen_validation_rows": len(unseen),
        },
        "long_duration_threshold_p90_hours": round(long_duration_threshold, 4),
        "diagnostic_policy": (
            "Phase 5 diagnoses the locked Phase 4 champion only. No hyperparameters are "
            "retuned and the sealed test split is not scored."
        ),
        "artifacts": {
            "segment_metrics_csv": str(segment_path),
            "stress_tests_csv": str(stress_path),
            "champion_vs_runner_up_csv": str(comparison_path),
            "validation_diagnostics_csv": str(diagnostics_path),
            "worst_errors_csv": str(worst_path),
            "figures": figures,
        },
    }

    report_path = metrics_dir / "phase5_reliability_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
