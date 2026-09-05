#!/usr/bin/env python
"""Train and evaluate the Phase 3 Linear Regression baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from flight_fare_intelligence.data import load_raw_dataset
from flight_fare_intelligence.modeling import (
    attach_split_assignments,
    build_linear_regression_pipeline,
    coefficient_table,
    model_metadata,
    prediction_frame,
    regression_metrics,
    save_model,
    split_xy,
)


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in metrics.items()}


def _generate_figures(
    predictions: pd.DataFrame, output_dir: Path, random_seed: int
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    figure, axis = plt.subplots(figsize=(7, 6))
    sample = predictions.sample(min(12_000, len(predictions)), random_state=random_seed)
    axis.scatter(sample["actual_price"], sample["predicted_price"], alpha=0.18, s=10)
    lower = min(sample["actual_price"].min(), sample["predicted_price"].min())
    upper = max(sample["actual_price"].max(), sample["predicted_price"].max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    axis.set_title("Linear Regression: Actual vs Predicted Fare")
    axis.set_xlabel("Actual Fare (INR)")
    axis.set_ylabel("Predicted Fare (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "phase3_actual_vs_predicted.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    clipped = predictions["residual"].clip(
        predictions["residual"].quantile(0.01),
        predictions["residual"].quantile(0.99),
    )
    axis.hist(clipped, bins=70)
    axis.axvline(0.0, linestyle="--", linewidth=1)
    axis.set_title("Linear Regression Validation Residual Distribution")
    axis.set_xlabel("Residual = Actual - Predicted (INR)")
    axis.set_ylabel("Records")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_dir / "phase3_residual_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    sample = predictions.sample(min(12_000, len(predictions)), random_state=random_seed)
    axis.scatter(sample["predicted_price"], sample["residual"], alpha=0.18, s=10)
    axis.axhline(0.0, linestyle="--", linewidth=1)
    axis.set_title("Linear Regression Residuals vs Predicted Fare")
    axis.set_xlabel("Predicted Fare (INR)")
    axis.set_ylabel("Residual (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "phase3_residuals_vs_predicted.png"
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw_dataset(args.data_path)
    assignments = pd.read_csv(args.assignments)
    modeling_frame = attach_split_assignments(df, assignments)

    x_train, y_train, _ = split_xy(modeling_frame, "train")
    x_validation, y_validation, validation_ids = split_xy(modeling_frame, "validation")

    pipeline = build_linear_regression_pipeline()
    fit_started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    train_prediction = pipeline.predict(x_train)
    validation_prediction = pipeline.predict(x_validation)
    inference_seconds = time.perf_counter() - inference_started

    train_metrics = regression_metrics(y_train, train_prediction)
    validation_metrics = regression_metrics(y_validation, validation_prediction)
    validation_predictions = prediction_frame(
        validation_ids,
        y_validation,
        validation_prediction,
    )

    validation_context = modeling_frame.loc[
        modeling_frame["split"] == "validation", ["record_id", "class"]
    ].reset_index(drop=True)
    validation_diagnostics = validation_predictions.merge(
        validation_context, on="record_id", how="left", validate="one_to_one"
    )
    validation_by_class: dict[str, dict[str, float]] = {}
    for cabin_class, group in validation_diagnostics.groupby("class", sort=True):
        class_metrics = regression_metrics(group["actual_price"], group["predicted_price"].to_numpy())
        validation_by_class[str(cabin_class)] = _round_metrics(class_metrics.to_dict())

    metrics_dir = Path("reports/metrics")
    predictions_dir = Path("reports/predictions")
    figures_dir = Path("reports/figures")
    model_dir = Path("models")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = save_model(pipeline, model_dir / "phase3_linear_regression.joblib")
    prediction_path = predictions_dir / "phase3_linear_regression_validation.csv"
    validation_predictions.to_csv(prediction_path, index=False)

    coefficients = coefficient_table(pipeline)
    coefficient_path = predictions_dir / "phase3_linear_regression_coefficients.csv"
    coefficients.to_csv(coefficient_path, index=False)

    figure_paths = _generate_figures(validation_predictions, figures_dir, args.seed)

    total_scored_rows = len(x_train) + len(x_validation)
    report = {
        "phase": 3,
        "model": "LinearRegression",
        "random_seed": args.seed,
        "split_policy": {
            "train": "fit baseline",
            "validation": "official Phase 3 benchmark",
            "test": "sealed; not scored in Phase 3",
        },
        "rows": {
            "train": len(x_train),
            "validation": len(x_validation),
            "test": int((modeling_frame["split"] == "test").sum()),
        },
        "metrics": {
            "train": _round_metrics(train_metrics.to_dict()),
            "validation": _round_metrics(validation_metrics.to_dict()),
            "validation_by_class": validation_by_class,
        },
        "timing": {
            "fit_seconds": round(fit_seconds, 4),
            "train_plus_validation_inference_seconds": round(inference_seconds, 4),
            "average_inference_ms_per_row": round(
                inference_seconds / total_scored_rows * 1000.0,
                6,
            ),
        },
        "model_metadata": model_metadata(pipeline),
        "residual_summary_validation": {
            "mean_residual": round(float(validation_predictions["residual"].mean()), 4),
            "median_residual": round(float(validation_predictions["residual"].median()), 4),
            "mean_absolute_percentage_error": round(
                float(validation_predictions["absolute_percentage_error"].mean()), 4
            ),
            "negative_prediction_count": int((validation_predictions["predicted_price"] < 0).sum()),
            "negative_prediction_rate_percent": round(
                float((validation_predictions["predicted_price"] < 0).mean() * 100.0), 4
            ),
            "within_1000_inr_percent": round(
                float((validation_predictions["absolute_error"] <= 1000).mean() * 100.0), 4
            ),
            "within_2000_inr_percent": round(
                float((validation_predictions["absolute_error"] <= 2000).mean() * 100.0), 4
            ),
            "within_5000_inr_percent": round(
                float((validation_predictions["absolute_error"] <= 5000).mean() * 100.0), 4
            ),
        },
        "artifact": str(artifact_path),
        "validation_predictions": str(prediction_path),
        "coefficients": str(coefficient_path),
        "figures": figure_paths,
    }

    report_path = metrics_dir / "phase3_linear_regression_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
