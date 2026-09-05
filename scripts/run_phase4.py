#!/usr/bin/env python
"""Assemble isolated Phase 4 family benchmarks and select the champion model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flight_fare_intelligence.benchmarking import (
    measure_inference_latency,
    model_size_mb,
    select_champion,
)
from flight_fare_intelligence.data import load_raw_dataset
from flight_fare_intelligence.modeling import attach_split_assignments, prediction_frame, split_xy

MODEL_FAMILIES = ["random_forest", "xgboost", "catboost"]


def _generate_figures(comparison: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    ordered = comparison.sort_values("rmse")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(ordered["model"], ordered["rmse"])
    axis.set_title("Phase 4 Validation RMSE by Model")
    axis.set_ylabel("RMSE (INR)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase4_validation_rmse.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in comparison.itertuples(index=False):
        axis.scatter(row.median_batch_ms, row.rmse, s=70)
        axis.annotate(
            row.model,
            (row.median_batch_ms, row.rmse),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_title("Validation Accuracy vs Batch Inference Latency")
    axis.set_xlabel("Median latency for 5,000 rows (ms)")
    axis.set_ylabel("RMSE (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase4_accuracy_vs_latency.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    sample = predictions.sample(min(12_000, len(predictions)), random_state=42)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(sample["actual_price"], sample["champion_prediction"], alpha=0.18, s=10)
    lower = min(sample["actual_price"].min(), sample["champion_prediction"].min())
    upper = max(sample["actual_price"].max(), sample["champion_prediction"].max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    axis.set_title("Phase 4 Champion: Actual vs Predicted Fare")
    axis.set_xlabel("Actual Fare (INR)")
    axis.set_ylabel("Predicted Fare (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = figures_dir / "phase4_champion_actual_vs_predicted.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    return generated


def _load_family_report(family: str) -> dict:
    path = Path("reports/metrics") / f"phase4_{family}_benchmark.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {family} benchmark. Run scripts/benchmark_family.py for all families first."
        )
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/processed/phase2_split_assignments.csv"),
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("models/phase3_linear_regression.joblib"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("reports/metrics/phase3_linear_regression_metrics.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.baseline_model.exists() or not args.baseline_report.exists():
        raise FileNotFoundError(
            "Phase 3 baseline artifacts are required. Run `make phase3` before Phase 4."
        )

    family_reports = {family: _load_family_report(family) for family in MODEL_FAMILIES}

    df = load_raw_dataset(args.data_path)
    assignments = pd.read_csv(args.assignments)
    modeling_frame = attach_split_assignments(df, assignments)
    x_train, _, _ = split_xy(modeling_frame, "train")
    x_validation, y_validation, validation_ids = split_xy(modeling_frame, "validation")
    test_rows = int((modeling_frame["split"] == "test").sum())

    baseline_model = joblib.load(args.baseline_model)
    baseline_report = json.loads(args.baseline_report.read_text())
    baseline_predictions = np.asarray(baseline_model.predict(x_validation), dtype=float)
    baseline_latency = measure_inference_latency(
        baseline_model,
        x_validation,
        random_seed=args.seed,
    )
    baseline_metrics = baseline_report["metrics"]["validation"]

    comparison_rows: list[dict] = [
        {
            "model": "linear_regression",
            "candidate": "phase3_baseline",
            "rmse": float(baseline_metrics["rmse"]),
            "mae": float(baseline_metrics["mae"]),
            "r2": float(baseline_metrics["r2"]),
            "mape": float(baseline_metrics["mape"]),
            "fit_seconds": float(baseline_report["timing"]["fit_seconds"]),
            "median_batch_ms": baseline_latency.median_batch_ms,
            "median_ms_per_row": baseline_latency.median_ms_per_row,
            "model_size_mb": model_size_mb(args.baseline_model),
        }
    ]

    for family in MODEL_FAMILIES:
        report = family_reports[family]
        metrics = report["validation_metrics"]
        latency = report["latency"]
        comparison_rows.append(
            {
                "model": family,
                "candidate": report["best_candidate"],
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "mape": float(metrics["mape"]),
                "fit_seconds": float(report["fit_seconds"]),
                "median_batch_ms": float(latency["median_batch_ms"]),
                "median_ms_per_row": float(latency["median_ms_per_row"]),
                "model_size_mb": float(report["model_size_mb"]),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    champion = select_champion(comparison)
    if champion == "linear_regression":
        champion_source = args.baseline_model
        champion_predictions = baseline_predictions
    else:
        champion_source = Path(family_reports[champion]["model_path"])
        champion_frame = pd.read_csv(family_reports[champion]["predictions_path"])
        champion_predictions = champion_frame[f"{champion}_prediction"].to_numpy()

    champion_path = Path("models/phase4_champion.joblib")
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(champion_source, champion_path)

    predictions = prediction_frame(validation_ids, y_validation, champion_predictions)
    predictions = predictions.rename(columns={"predicted_price": "champion_prediction"})
    predictions["linear_regression_prediction"] = baseline_predictions
    for family in MODEL_FAMILIES:
        family_predictions = pd.read_csv(family_reports[family]["predictions_path"])
        predictions = predictions.merge(
            family_predictions[["record_id", f"{family}_prediction"]],
            on="record_id",
            how="left",
            validate="one_to_one",
        )

    metrics_dir = Path("reports/metrics")
    predictions_dir = Path("reports/predictions")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = predictions_dir / "phase4_model_comparison.csv"
    comparison.sort_values("rmse").to_csv(comparison_path, index=False)
    predictions_path = predictions_dir / "phase4_validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    figures = _generate_figures(comparison, predictions)

    baseline_rmse = float(
        comparison.loc[comparison["model"] == "linear_regression", "rmse"].iloc[0]
    )
    best_rmse = float(comparison["rmse"].min())
    report = {
        "phase": 4,
        "random_seed": args.seed,
        "split_policy": {
            "train": "fit all tree-model candidates",
            "validation": "tune candidates and select champion",
            "test": "sealed; not scored in Phase 4",
        },
        "rows": {
            "train": len(x_train),
            "validation": len(x_validation),
            "test_sealed": test_rows,
        },
        "execution_design": (
            "Each tree-model family is trained in a separate Python process. This isolates "
            "native thread runtimes, bounds peak memory, and makes local benchmarking stable."
        ),
        "family_reports": {
            family: f"reports/metrics/phase4_{family}_benchmark.json"
            for family in MODEL_FAMILIES
        },
        "comparison": [
            {
                key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value
                for key, value in row.items()
            }
            for row in comparison.sort_values("rmse").to_dict(orient="records")
        ],
        "selection_policy": (
            "Lowest validation RMSE; MAE, median batch latency, and model size are "
            "deterministic production tie-breakers."
        ),
        "champion": champion,
        "champion_artifact": str(champion_path),
        "validation_rmse_improvement_vs_linear_percent": round(
            (baseline_rmse - best_rmse) / baseline_rmse * 100.0,
            4,
        ),
        "test_set_scored": False,
        "artifacts": {
            "comparison_csv": str(comparison_path),
            "validation_predictions_csv": str(predictions_path),
            "figures": figures,
        },
    }

    report_path = metrics_dir / "phase4_model_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
