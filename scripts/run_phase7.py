#!/usr/bin/env python
"""Run Phase 7 fare intelligence, uncertainty, and decision-support diagnostics."""

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
from flight_fare_intelligence.intelligence import (
    benchmark_table,
    booking_guidance,
    build_comparable_fare_index,
    counterfactual_table,
    days_left_curve,
    fare_opportunity,
)
from flight_fare_intelligence.modeling import attach_split_assignments
from flight_fare_intelligence.reliability import BOOKING_HORIZON_BINS, BOOKING_HORIZON_LABELS
from flight_fare_intelligence.schema import MODEL_FEATURES, TARGET_COLUMN
from flight_fare_intelligence.uncertainty import (
    PREDICTED_FARE_BINS,
    PREDICTED_FARE_LABELS,
    comparative_reliability_score,
    deterministic_validation_partition,
    evaluate_intervals,
    fit_segment_conformal_calibrator,
    reliability_reference,
)


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


def _load_phase6_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 6 report: {path}. Run `make phase6` first.")
    report = json.loads(path.read_text())
    if report.get("champion") != "xgboost":
        raise RuntimeError("Phase 7 requires the locked XGBoost champion")
    if report.get("test_set_scored") is not False:
        raise RuntimeError("Phase 6 must confirm the test split remains sealed")
    return report


def _coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dimension in ["class", "booking_horizon", "predicted_fare_band"]:
        for value, segment in frame.groupby(dimension, observed=True):
            metrics = evaluate_intervals(segment["actual_price"], segment)
            rows.append({"dimension": dimension, "segment": str(value), **metrics})
    return pd.DataFrame(rows)


def _bar_plot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: Path,
    target_line: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(data[x].astype(str), data[y])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    if target_line is not None:
        axis.axhline(target_line, linestyle="--", linewidth=1.2, label="Target")
        axis.legend()
    for bar, value in zip(bars, data[y], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _generate_figures(
    evaluation: pd.DataFrame,
    coverage: pd.DataFrame,
    demo_curve: pd.DataFrame,
    *,
    current_days_left: int,
) -> list[str]:
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    class_coverage = coverage.loc[coverage["dimension"] == "class"].copy()
    path = figures_dir / "phase7_interval_coverage_by_class.png"
    _bar_plot(
        class_coverage,
        x="segment",
        y="coverage_percent",
        title="Phase 7 Prediction-Interval Coverage by Cabin Class",
        ylabel="Empirical coverage (%)",
        output_path=path,
        target_line=90.0,
    )
    generated.append(str(path))

    horizon = evaluation.groupby("booking_horizon", observed=True)["interval_width"].median()
    horizon = horizon.reindex(BOOKING_HORIZON_LABELS).dropna().reset_index()
    horizon.columns = ["booking_horizon", "median_interval_width"]
    path = figures_dir / "phase7_interval_width_by_booking_horizon.png"
    _bar_plot(
        horizon,
        x="booking_horizon",
        y="median_interval_width",
        title="Median 90% Prediction-Interval Width by Booking Horizon",
        ylabel="Interval width (INR)",
        output_path=path,
    )
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(evaluation["fare_opportunity_score"], bins=np.arange(-0.5, 101.5, 5))
    axis.set_title("Fare Opportunity Score Distribution")
    axis.set_xlabel("Fare Opportunity Score")
    axis.set_ylabel("Validation evaluation rows")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase7_fare_opportunity_score_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(evaluation["reliability_score"], bins=np.arange(0, 105, 5))
    axis.set_title("Comparative Prediction Reliability Score")
    axis.set_xlabel("Reliability score")
    axis.set_ylabel("Validation evaluation rows")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase7_reliability_score_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(demo_curve["days_left"], demo_curve["predicted_fare"], marker="o", markersize=2)
    current = demo_curve.loc[demo_curve["days_left"] == current_days_left]
    if not current.empty:
        axis.scatter(current["days_left"], current["predicted_fare"], s=70, zorder=3)
    axis.set_title("Demo What-If: Predicted Fare vs Days Before Departure")
    axis.set_xlabel("Days before departure")
    axis.set_ylabel("Predicted fare (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase7_demo_booking_horizon_curve.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    return generated


def _select_demo_scenario(evaluation: pd.DataFrame) -> pd.Series:
    preferred = evaluation.loc[
        evaluation["source_city"].eq("Delhi")
        & evaluation["destination_city"].eq("Mumbai")
        & evaluation["airline"].eq("Vistara")
        & evaluation["class"].eq("Economy")
        & evaluation["departure_time"].eq("Morning")
    ].copy()
    if preferred.empty:
        preferred = evaluation.loc[evaluation["class"].eq("Economy")].copy()
    preferred["distance_to_12_days"] = (preferred["days_left"] - 12).abs()
    return preferred.sort_values(["distance_to_12_days", "record_id"]).iloc[0]


def _valid_counterfactual_values(
    training: pd.DataFrame,
    demo: pd.Series,
    feature: str,
) -> list[str]:
    context = training.loc[
        training["source_city"].eq(demo["source_city"])
        & training["destination_city"].eq(demo["destination_city"])
        & training["class"].eq(demo["class"])
    ]
    return sorted(str(value) for value in context[feature].dropna().unique())


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
        "--phase6-report",
        type=Path,
        default=Path("reports/metrics/phase6_explainability_summary.json"),
    )
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--min-calibration-rows", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for required in [args.assignments, args.champion_model, args.phase6_report]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required Phase 7 input: {required}")
    phase6_report = _load_phase6_gate(args.phase6_report)

    data = load_raw_dataset(args.data_path)
    assignments = pd.read_csv(args.assignments)
    modeling = attach_split_assignments(data, assignments)
    training = modeling.loc[modeling["split"] == "train"].copy()
    validation = modeling.loc[modeling["split"] == "validation"].copy()
    sealed_test_rows = int((modeling["split"] == "test").sum())
    if sealed_test_rows != int(phase6_report["rows"]["test_sealed"]):
        raise RuntimeError("Sealed test row count changed after Phase 6")

    model = joblib.load(args.champion_model)
    validation_features = validation[MODEL_FEATURES].copy()
    validation_prediction = np.asarray(model.predict(validation_features), dtype=float)

    calibration_mask = deterministic_validation_partition(
        validation,
        calibration_fraction=0.50,
        random_seed=args.seed,
    )
    calibration = validation.loc[calibration_mask].copy()
    evaluation = validation.loc[~calibration_mask].copy()
    calibration_prediction = validation_prediction[calibration_mask.to_numpy()]
    evaluation_prediction = validation_prediction[(~calibration_mask).to_numpy()]

    calibrator = fit_segment_conformal_calibrator(
        calibration[MODEL_FEATURES],
        calibration[TARGET_COLUMN],
        calibration_prediction,
        coverage=args.coverage,
        min_rows=args.min_calibration_rows,
    )
    calibration_intervals = calibrator.interval_frame(
        calibration[MODEL_FEATURES],
        calibration_prediction,
    )
    evaluation_intervals = calibrator.interval_frame(
        evaluation[MODEL_FEATURES],
        evaluation_prediction,
    )

    index = build_comparable_fare_index(training)
    benchmarks = benchmark_table(index)

    calibration_interval_width = calibration_intervals["interval_width"].to_numpy(dtype=float)
    calibration_relative_width = calibration_interval_width / np.maximum(
        calibration_prediction,
        1_000.0,
    )
    rel_reference = reliability_reference(calibration_relative_width)

    diagnostics = evaluation[MODEL_FEATURES + ["record_id", "scenario_id"]].copy()
    diagnostics["actual_price"] = evaluation[TARGET_COLUMN].to_numpy(dtype=float)
    diagnostics["predicted_price"] = evaluation_prediction
    diagnostics["booking_horizon"] = pd.cut(
        diagnostics["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    ).astype(str)
    diagnostics["predicted_fare_band"] = pd.cut(
        diagnostics["predicted_price"],
        bins=PREDICTED_FARE_BINS,
        labels=PREDICTED_FARE_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    for column in evaluation_intervals.columns:
        diagnostics[column] = evaluation_intervals[column].to_numpy()
    diagnostics["covered"] = diagnostics["actual_price"].between(
        diagnostics["prediction_lower"],
        diagnostics["prediction_upper"],
    )
    diagnostics["relative_interval_width"] = diagnostics["interval_width"] / np.maximum(
        diagnostics["predicted_price"],
        1_000.0,
    )

    opportunity_rows: list[dict[str, float | int | str]] = []
    reliability_rows: list[tuple[int, str, float]] = []
    scenarios = diagnostics[MODEL_FEATURES].to_dict(orient="records")
    predicted_values = diagnostics["predicted_price"].to_numpy(dtype=float)
    relative_width_values = diagnostics["relative_interval_width"].to_numpy(dtype=float)
    for scenario, predicted_price, relative_width in zip(
        scenarios,
        predicted_values,
        relative_width_values,
        strict=True,
    ):
        opportunity_rows.append(fare_opportunity(float(predicted_price), scenario, index))
        reliability_rows.append(comparative_reliability_score(float(relative_width), rel_reference))
    opportunity_frame = pd.DataFrame(opportunity_rows, index=diagnostics.index)
    for column in opportunity_frame.columns:
        diagnostics[column] = opportunity_frame[column]
    diagnostics["reliability_score"] = [item[0] for item in reliability_rows]
    diagnostics["reliability_label"] = [item[1] for item in reliability_rows]
    diagnostics["relative_uncertainty_percentile"] = [item[2] for item in reliability_rows]

    coverage = _coverage_table(diagnostics)
    overall_interval = evaluate_intervals(diagnostics["actual_price"], diagnostics)
    high_fare = diagnostics.loc[diagnostics["actual_price"] >= 80_000]
    last_minute = diagnostics.loc[diagnostics["days_left"].between(1, 7)]

    demo = _select_demo_scenario(diagnostics)
    demo_scenario = {feature: demo[feature] for feature in MODEL_FEATURES}
    demo_point = float(model.predict(pd.DataFrame([demo_scenario]))[0])
    demo_interval = calibrator.interval_frame(
        pd.DataFrame([demo_scenario]),
        np.array([demo_point]),
    ).iloc[0]
    demo_opportunity = fare_opportunity(demo_point, demo_scenario, index)
    demo_relative_width = float(demo_interval["interval_width"] / max(demo_point, 1_000.0))
    demo_reliability = comparative_reliability_score(demo_relative_width, rel_reference)
    demo_curve = days_left_curve(model, demo_scenario)
    demo_guidance = booking_guidance(
        current_days_left=int(demo_scenario["days_left"]),
        current_fare=demo_point,
        opportunity_score=int(demo_opportunity["fare_opportunity_score"]),
        curve=demo_curve,
    )

    counterfactual_frames: list[pd.DataFrame] = []
    for feature in ["airline", "departure_time", "stops"]:
        values = _valid_counterfactual_values(training, demo, feature)
        table = counterfactual_table(model, demo_scenario, feature=feature, values=values)
        table.insert(0, "dimension", feature)
        counterfactual_frames.append(table)
    demo_counterfactuals = pd.concat(counterfactual_frames, ignore_index=True)

    training_curve = training.copy()
    training_curve["route"] = (
        training_curve["source_city"].astype(str)
        + ">"
        + training_curve["destination_city"].astype(str)
    )
    booking_horizon_intelligence = (
        training_curve.groupby(["route", "class", "days_left"], observed=True)[TARGET_COLUMN]
        .agg(rows="size", median_fare="median", mean_fare="mean")
        .reset_index()
    )

    metrics_dir = Path("reports/metrics")
    predictions_dir = Path("reports/predictions")
    models_dir = Path("models")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = predictions_dir / "phase7_uncertainty_evaluation.csv"
    coverage_path = predictions_dir / "phase7_interval_coverage_by_segment.csv"
    benchmark_path = predictions_dir / "phase7_route_benchmarks.csv"
    booking_path = predictions_dir / "phase7_booking_horizon_intelligence.csv"
    demo_curve_path = predictions_dir / "phase7_demo_days_left_curve.csv"
    demo_counterfactual_path = predictions_dir / "phase7_demo_counterfactuals.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    benchmarks.to_csv(benchmark_path, index=False)
    booking_horizon_intelligence.to_csv(booking_path, index=False)
    demo_curve.to_csv(demo_curve_path, index=False)
    demo_counterfactuals.to_csv(demo_counterfactual_path, index=False)

    bundle_path = models_dir / "phase7_intelligence_bundle.joblib"
    joblib.dump(
        {
            "calibrator": calibrator,
            "comparable_fare_index": index,
            "reliability_reference": rel_reference,
            "metadata": {
                "coverage": args.coverage,
                "calibration_rows": len(calibration),
                "calibration_fraction_of_validation": len(calibration) / len(validation),
                "test_set_scored": False,
            },
        },
        bundle_path,
    )

    figures = _generate_figures(
        diagnostics,
        coverage,
        demo_curve,
        current_days_left=int(demo_scenario["days_left"]),
    )

    report = {
        "phase": 7,
        "champion": "xgboost",
        "test_set_scored": False,
        "uncertainty_policy": {
            "method": "tail_aware_hierarchical_asymmetric_split_conformal_residual_intervals",
            "nominal_coverage_percent": round(args.coverage * 100.0, 2),
            "validation_partition": (
                "Scenario-preserving deterministic 50/50 calibration/evaluation split "
                "inside the Phase 2 validation split."
            ),
            "hierarchy": [
                "class + booking_horizon + predicted_fare_band",
                "class + predicted_fare_band",
                "class + booking_horizon",
                "class",
                "global fallback",
            ],
            "minimum_rows_per_rule": args.min_calibration_rows,
            "high_fare_business_guardrail": {
                "trigger": "Business predictions in the 60-80k or 80k+ predicted-fare bands",
                "upper_residual_quantile": calibrator.tail_upper_quantile,
                "minimum_rows": calibrator.tail_min_rows,
                "motivation": (
                    "Phase 5 found systematic underprediction in the extreme high-fare tail, "
                    "so Phase 7 uses a deliberately conservative upper residual bound there."
                ),
            },
        },
        "rows": {
            "train_reference": len(training),
            "validation_total": len(validation),
            "uncertainty_calibration": len(calibration),
            "uncertainty_evaluation": len(evaluation),
            "test_sealed": sealed_test_rows,
        },
        "interval_evaluation": _round_payload(overall_interval),
        "stress_interval_coverage": {
            "high_fare_80k_plus": _round_payload(
                evaluate_intervals(high_fare["actual_price"], high_fare)
            ),
            "last_minute_1_7_days": _round_payload(
                evaluate_intervals(last_minute["actual_price"], last_minute)
            ),
        },
        "fare_opportunity_policy": {
            "comparison_group": "directed route + cabin class + booking horizon",
            "score_definition": (
                "100 minus the empirical percentile of the model-estimated fare among "
                "historical training fares for comparable flights."
            ),
            "labels": {
                "80-100": "EXCELLENT_VALUE",
                "60-79": "GOOD_VALUE",
                "40-59": "TYPICAL",
                "20-39": "ABOVE_TYPICAL",
                "0-19": "EXPENSIVE",
            },
        },
        "reliability_policy": {
            "definition": (
                "Comparative score based on the percentile of prediction-interval width "
                "relative to predicted fare against the calibration distribution."
            ),
            "warning": "The score is comparative uncertainty, not probability of correctness.",
        },
        "decision_policy": {
            "guidance_labels": ["BUY_NOW", "MONITOR", "WAIT_OR_MONITOR"],
            "wait_window_days": 7,
            "material_change_threshold_percent": 5.0,
            "warning": (
                "Guidance is historical/model-based counterfactual decision support, not a "
                "guaranteed forecast of live airline prices."
            ),
        },
        "demo": {
            "record_id": int(demo["record_id"]),
            "scenario": {
                "airline": str(demo_scenario["airline"]),
                "source_city": str(demo_scenario["source_city"]),
                "destination_city": str(demo_scenario["destination_city"]),
                "departure_time": str(demo_scenario["departure_time"]),
                "stops": str(demo_scenario["stops"]),
                "class": str(demo_scenario["class"]),
                "duration": float(demo_scenario["duration"]),
                "days_left": int(demo_scenario["days_left"]),
            },
            "actual_validation_fare": round(float(demo["actual_price"]), 2),
            "predicted_fare": round(demo_point, 2),
            "prediction_interval": {
                "lower": round(float(demo_interval["prediction_lower"]), 2),
                "upper": round(float(demo_interval["prediction_upper"]), 2),
                "calibration_level": str(demo_interval["calibration_level"]),
                "calibration_rows": int(demo_interval["calibration_rows"]),
            },
            "fare_opportunity": _round_payload(demo_opportunity),
            "reliability": {
                "score": demo_reliability[0],
                "label": demo_reliability[1],
                "relative_uncertainty_percentile": round(demo_reliability[2], 4),
            },
            "booking_guidance": _round_payload(demo_guidance),
        },
        "artifacts": {
            "intelligence_bundle": str(bundle_path),
            "uncertainty_evaluation_csv": str(diagnostics_path),
            "interval_coverage_csv": str(coverage_path),
            "route_benchmarks_csv": str(benchmark_path),
            "booking_horizon_intelligence_csv": str(booking_path),
            "demo_days_left_curve_csv": str(demo_curve_path),
            "demo_counterfactuals_csv": str(demo_counterfactual_path),
            "figures": figures,
        },
    }

    report_path = metrics_dir / "phase7_intelligence_summary.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"report": str(report_path), **report}, indent=2, default=str))


if __name__ == "__main__":
    main()
