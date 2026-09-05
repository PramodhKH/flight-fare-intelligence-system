#!/usr/bin/env python
"""Run Phase 6 global, segment, and local SHAP explanations for XGBoost."""

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
from flight_fare_intelligence.explainability import (
    categorical_effects,
    explain_raw_features,
    global_importance,
    local_explanation_table,
    segment_importance,
    select_representative_index,
)
from flight_fare_intelligence.modeling import (
    CATEGORICAL_FEATURES,
    attach_split_assignments,
    split_xy,
)

SEGMENT_ORDER = [
    "overall_validation",
    "economy",
    "business",
    "last_minute_1_7_days",
    "high_fare_80k_plus",
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


def _load_phase5_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 5 reliability report: {path}. Run `make phase5` first."
        )
    report = json.loads(path.read_text())
    if report.get("champion") != "xgboost":
        raise RuntimeError("Phase 6 requires the locked Phase 5 XGBoost champion")
    if report.get("test_set_scored") is not False:
        raise RuntimeError("Phase 5 must confirm the test split is still sealed")
    return report


def _global_importance_plot(importance: pd.DataFrame, output_path: Path) -> None:
    plot_data = importance.sort_values("mean_abs_shap_inr", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(plot_data["feature"], plot_data["mean_abs_shap_inr"])
    axis.set_title("Global XGBoost SHAP Importance")
    axis.set_xlabel("Mean absolute SHAP contribution (INR)")
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, plot_data["mean_abs_shap_inr"], strict=True):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"₹{value:,.0f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _segment_heatmap(segment_table: pd.DataFrame, output_path: Path) -> None:
    pivot = segment_table.pivot(
        index="segment",
        columns="feature",
        values="importance_percent",
    ).reindex(SEGMENT_ORDER)
    ordered_features = (
        segment_table.loc[segment_table["segment"] == "overall_validation"]
        .sort_values("importance_percent", ascending=False)["feature"]
        .tolist()
    )
    pivot = pivot[ordered_features]

    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    axis.set_title("SHAP Driver Share by Reliability Segment")
    axis.set_ylabel("Segment")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Share of mean |SHAP| (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _numeric_dependence_plot(
    frame: pd.DataFrame,
    *,
    feature: str,
    shap_column: str,
    title: str,
    output_path: Path,
    random_seed: int,
    max_points: int = 8_000,
) -> None:
    plot_data = frame
    if len(plot_data) > max_points:
        plot_data = plot_data.sample(n=max_points, random_state=random_seed)

    figure, axis = plt.subplots(figsize=(8, 5))
    for cabin_class, subset in plot_data.groupby("class", observed=True):
        axis.scatter(
            subset[feature],
            subset[shap_column],
            alpha=0.18,
            s=10,
            label=str(cabin_class),
        )
    grouped = frame.groupby(feature, observed=True)[shap_column].mean().sort_index()
    axis.plot(grouped.index, grouped.values, linewidth=2.0, label="mean contribution")
    axis.axhline(0.0, linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel(feature.replace("_", " ").title())
    axis.set_ylabel("SHAP contribution to predicted fare (INR)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _local_plot(table: pd.DataFrame, output_path: Path) -> None:
    plot_data = table.sort_values("shap_value_inr")
    case = str(plot_data.iloc[0]["case_label"])
    actual = float(plot_data.iloc[0]["actual_price"])
    predicted = float(plot_data.iloc[0]["predicted_price"])
    expected = float(plot_data.iloc[0]["expected_value"])
    labels = [
        f"{row.feature}={row.feature_value}"
        for row in plot_data.itertuples(index=False)
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.barh(labels, plot_data["shap_value_inr"])
    axis.axvline(0.0, linewidth=1.0)
    axis.set_title(
        f"{case}: local SHAP explanation\n"
        f"baseline ₹{expected:,.0f} → predicted ₹{predicted:,.0f}; actual ₹{actual:,.0f}"
    )
    axis.set_xlabel("Contribution to predicted fare (INR)")
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, plot_data["shap_value_inr"], strict=True):
        alignment = "left" if value >= 0 else "right"
        padding = 70 if value >= 0 else -70
        axis.text(
            value + padding,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+,.0f}",
            ha=alignment,
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _representative_cases(diagnostics: pd.DataFrame) -> dict[str, int]:
    return {
        "representative_economy": select_representative_index(
            diagnostics,
            diagnostics["class"].eq("Economy"),
        ),
        "representative_business": select_representative_index(
            diagnostics,
            diagnostics["class"].eq("Business"),
        ),
        "representative_last_minute": select_representative_index(
            diagnostics,
            diagnostics["days_left"].between(1, 7),
        ),
        "representative_high_fare": select_representative_index(
            diagnostics,
            diagnostics["actual_price"].ge(80_000),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--phase5-report",
        type=Path,
        default=Path("reports/metrics/phase5_reliability_summary.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    phase5_report = _load_phase5_gate(args.phase5_report)
    if not args.champion_model.exists():
        raise FileNotFoundError(
            f"Missing champion artifact: {args.champion_model}. Run `make phase4` first."
        )

    raw = load_raw_dataset(args.data_path)
    assignments = pd.read_csv(args.assignments)
    modeling_frame = attach_split_assignments(raw, assignments)
    validation_features, validation_target, validation_ids = split_xy(
        modeling_frame,
        "validation",
    )
    validation_features = validation_features.reset_index(drop=True)
    validation_target = validation_target.reset_index(drop=True)
    validation_ids = validation_ids.reset_index(drop=True)
    sealed_test_rows = len(modeling_frame.loc[modeling_frame["split"] == "test"])

    champion = joblib.load(args.champion_model)
    predictions = np.asarray(champion.predict(validation_features), dtype=float)
    diagnostics = validation_features.copy()
    diagnostics.insert(0, "record_id", validation_ids.to_numpy())
    diagnostics["actual_price"] = validation_target.to_numpy(dtype=float)
    diagnostics["predicted_price"] = predictions
    diagnostics["absolute_error"] = np.abs(
        diagnostics["predicted_price"] - diagnostics["actual_price"]
    )

    global_shap = explain_raw_features(
        champion,
        validation_features,
        approximate=True,
    )
    if global_shap.max_reconstruction_error > 1.0:
        raise RuntimeError(
            "Approximate TreeSHAP reconstruction exceeded the 1 INR tolerance: "
            f"{global_shap.max_reconstruction_error:.4f}"
        )

    importance = global_importance(global_shap.raw_values)
    segment_masks = {
        "overall_validation": np.ones(len(diagnostics), dtype=bool),
        "economy": diagnostics["class"].eq("Economy"),
        "business": diagnostics["class"].eq("Business"),
        "last_minute_1_7_days": diagnostics["days_left"].between(1, 7),
        "high_fare_80k_plus": diagnostics["actual_price"].ge(80_000),
    }
    segment_drivers = segment_importance(
        global_shap.raw_values,
        diagnostics,
        segment_masks,
    )
    categorical = categorical_effects(
        validation_features,
        global_shap.raw_values,
        CATEGORICAL_FEATURES,
        min_rows=20,
    )

    shap_validation = pd.concat(
        [
            diagnostics[["record_id", "actual_price", "predicted_price"]].copy(),
            global_shap.raw_values.add_prefix("shap_"),
        ],
        axis=1,
    )
    numeric_dependence = diagnostics[
        ["record_id", "class", "actual_price", "predicted_price", "days_left", "duration"]
    ].copy()
    numeric_dependence["shap_days_left"] = global_shap.raw_values["days_left"]
    numeric_dependence["shap_duration"] = global_shap.raw_values["duration"]

    representative = _representative_cases(diagnostics)
    local_positions = list(representative.values())
    local_features = validation_features.iloc[local_positions].reset_index(drop=True)
    exact_local = explain_raw_features(champion, local_features, approximate=False)
    if exact_local.max_reconstruction_error > 1.0:
        raise RuntimeError(
            "Exact TreeSHAP reconstruction exceeded the 1 INR tolerance: "
            f"{exact_local.max_reconstruction_error:.4f}"
        )

    local_tables: list[pd.DataFrame] = []
    case_figure_paths: list[str] = []
    figures_dir = Path("reports/figures")
    metrics_dir = Path("reports/metrics")
    predictions_dir = Path("reports/predictions")
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for local_offset, (case_label, diagnostic_position) in enumerate(representative.items()):
        diagnostic_row = diagnostics.iloc[diagnostic_position]
        explanation = local_explanation_table(
            case_label=case_label,
            record_id=int(diagnostic_row["record_id"]),
            feature_row=local_features.iloc[local_offset],
            shap_row=exact_local.raw_values.iloc[local_offset],
            expected_value=exact_local.expected_value,
            predicted_price=float(diagnostic_row["predicted_price"]),
            actual_price=float(diagnostic_row["actual_price"]),
        )
        local_tables.append(explanation)
        figure_path = figures_dir / f"phase6_{case_label}_shap.png"
        _local_plot(explanation, figure_path)
        case_figure_paths.append(str(figure_path))

    local_explanations = pd.concat(local_tables, ignore_index=True)

    importance_path = predictions_dir / "phase6_global_feature_importance.csv"
    segment_path = predictions_dir / "phase6_segment_feature_importance.csv"
    categorical_path = predictions_dir / "phase6_categorical_effects.csv"
    validation_shap_path = predictions_dir / "phase6_validation_shap.csv"
    numeric_path = predictions_dir / "phase6_numeric_dependence.csv"
    local_path = predictions_dir / "phase6_local_explanations.csv"
    importance.to_csv(importance_path, index=False)
    segment_drivers.to_csv(segment_path, index=False)
    categorical.to_csv(categorical_path, index=False)
    shap_validation.to_csv(validation_shap_path, index=False)
    numeric_dependence.to_csv(numeric_path, index=False)
    local_explanations.to_csv(local_path, index=False)

    global_figure = figures_dir / "phase6_global_shap_importance.png"
    segment_figure = figures_dir / "phase6_segment_driver_heatmap.png"
    days_figure = figures_dir / "phase6_days_left_shap_dependence.png"
    duration_figure = figures_dir / "phase6_duration_shap_dependence.png"
    _global_importance_plot(importance, global_figure)
    _segment_heatmap(segment_drivers, segment_figure)
    _numeric_dependence_plot(
        numeric_dependence,
        feature="days_left",
        shap_column="shap_days_left",
        title="Booking-Horizon SHAP Effect",
        output_path=days_figure,
        random_seed=args.seed,
    )
    _numeric_dependence_plot(
        numeric_dependence,
        feature="duration",
        shap_column="shap_duration",
        title="Duration SHAP Effect",
        output_path=duration_figure,
        random_seed=args.seed,
    )

    segment_top_features: dict[str, list[dict[str, Any]]] = {}
    for segment in SEGMENT_ORDER:
        top = (
            segment_drivers.loc[segment_drivers["segment"] == segment]
            .nlargest(3, "mean_abs_shap_inr")
            [["feature", "mean_abs_shap_inr", "importance_percent", "mean_shap_inr"]]
        )
        segment_top_features[segment] = [
            _round_payload(row) for row in top.to_dict(orient="records")
        ]

    local_case_summary: dict[str, dict[str, Any]] = {}
    for case_label, diagnostic_position in representative.items():
        row = diagnostics.iloc[diagnostic_position]
        top = (
            local_explanations.loc[local_explanations["case_label"] == case_label]
            .nsmallest(3, "rank")
            [["feature", "feature_value", "shap_value_inr"]]
        )
        local_case_summary[case_label] = {
            "record_id": int(row["record_id"]),
            "actual_price": round(float(row["actual_price"]), 2),
            "predicted_price": round(float(row["predicted_price"]), 2),
            "absolute_error": round(float(row["absolute_error"]), 2),
            "top_contributions": [
                _round_payload(item) for item in top.to_dict(orient="records")
            ],
        }

    report = {
        "phase": 6,
        "champion": "xgboost",
        "evaluation_split": "validation",
        "test_set_scored": False,
        "rows": {
            "validation_explained": len(validation_features),
            "test_sealed": sealed_test_rows,
        },
        "explainability_policy": {
            "global_and_segment_method": "XGBoost approximate TreeSHAP over full validation split",
            "local_method": "XGBoost exact TreeSHAP for deterministic representative cases",
            "raw_feature_aggregation": (
                "One-hot transformed SHAP values are summed back to the eight deployed "
                "raw features, preserving additive prediction decomposition."
            ),
            "retuning_allowed": False,
        },
        "base_value_inr": round(global_shap.expected_value, 4),
        "approximate_max_reconstruction_error_inr": round(
            global_shap.max_reconstruction_error,
            6,
        ),
        "exact_local_max_reconstruction_error_inr": round(
            exact_local.max_reconstruction_error,
            6,
        ),
        "global_top_features": [
            _round_payload(row)
            for row in importance.head(5).to_dict(orient="records")
        ],
        "segment_top_features": segment_top_features,
        "representative_local_cases": local_case_summary,
        "phase5_anchor": {
            "validation_rmse": phase5_report["overall_validation"]["rmse"],
            "high_fare_80k_plus_bias": phase5_report["tail_reliability"]["fare_80k_plus"][
                "mean_bias"
            ],
        },
        "artifacts": {
            "global_feature_importance_csv": str(importance_path),
            "segment_feature_importance_csv": str(segment_path),
            "categorical_effects_csv": str(categorical_path),
            "validation_shap_csv": str(validation_shap_path),
            "numeric_dependence_csv": str(numeric_path),
            "local_explanations_csv": str(local_path),
            "figures": [
                str(global_figure),
                str(segment_figure),
                str(days_figure),
                str(duration_figure),
                *case_figure_paths,
            ],
        },
    }
    report_path = metrics_dir / "phase6_explainability_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
