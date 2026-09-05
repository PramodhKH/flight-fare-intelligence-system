#!/usr/bin/env python
"""Isolated Phase 4 worker for one native tree-model family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from flight_fare_intelligence.benchmarking import (
    fit_family_candidates,
    measure_inference_latency,
    model_size_mb,
    save_benchmark_model,
)
from flight_fare_intelligence.data import load_raw_dataset
from flight_fare_intelligence.modeling import attach_split_assignments, prediction_frame, split_xy


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--family", choices=["random_forest", "xgboost", "catboost"], required=True)
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

    model, best, predictions, benchmarks = fit_family_candidates(
        family=args.family,
        random_seed=args.seed,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    model_path = save_benchmark_model(model, Path("models") / f"phase4_{args.family}.joblib")
    latency = measure_inference_latency(model, x_validation, random_seed=args.seed)

    predictions_dir = Path("reports/predictions")
    metrics_dir = Path("reports/metrics")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    predictions_frame = prediction_frame(validation_ids, y_validation, predictions)
    predictions_frame = predictions_frame[["record_id", "actual_price", "predicted_price"]]
    predictions_frame = predictions_frame.rename(
        columns={"predicted_price": f"{args.family}_prediction"}
    )
    predictions_path = predictions_dir / f"phase4_{args.family}_validation.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    report = {
        "family": args.family,
        "best_candidate": best.candidate,
        "best_parameters": best.parameters,
        "validation_metrics": _round_metrics(best.metrics.to_dict()),
        "fit_seconds": round(best.fit_seconds, 4),
        "latency": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in latency.to_dict().items()
        },
        "model_size_mb": round(model_size_mb(model_path), 6),
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "candidates": [
            {
                "candidate": benchmark.candidate,
                "parameters": benchmark.parameters,
                "fit_seconds": round(benchmark.fit_seconds, 4),
                "validation_metrics": _round_metrics(benchmark.metrics.to_dict()),
            }
            for benchmark in benchmarks
        ],
    }
    report_path = metrics_dir / f"phase4_{args.family}_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
