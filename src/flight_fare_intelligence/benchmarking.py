"""Phase 4 tree-model construction, benchmarking, and champion selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RegressionMetrics, regression_metrics

PHASE4_CANDIDATES: dict[str, dict[str, dict[str, Any]]] = {
    "random_forest": {
        "balanced": {
            "n_estimators": 45,
            "max_depth": 18,
            "min_samples_leaf": 2,
            "max_features": 0.65,
            "max_samples": 0.55,
            "n_jobs": 4,
        },
        "high_capacity": {
            "n_estimators": 55,
            "max_depth": 20,
            "min_samples_leaf": 1,
            "max_features": 0.70,
            "max_samples": 0.60,
            "n_jobs": 4,
        },
    },
    "xgboost": {
        "balanced": {
            "n_estimators": 500,
            "max_depth": 9,
            "learning_rate": 0.055,
            "min_child_weight": 2,
            "subsample": 0.90,
            "colsample_bytree": 0.95,
            "reg_lambda": 3.0,
            "n_jobs": 4,
        },
        "high_capacity": {
            "n_estimators": 650,
            "max_depth": 9,
            "learning_rate": 0.045,
            "min_child_weight": 2,
            "subsample": 0.90,
            "colsample_bytree": 0.95,
            "reg_lambda": 3.0,
            "n_jobs": 4,
        },
    },
    "catboost": {
        "compact_native_cat": {
            "iterations": 150,
            "depth": 8,
            "learning_rate": 0.10,
            "l2_leaf_reg": 5.0,
            "thread_count": 4,
        },
        "expanded_native_cat": {
            "iterations": 180,
            "depth": 8,
            "learning_rate": 0.10,
            "l2_leaf_reg": 5.0,
            "thread_count": 4,
        },
    },
}


@dataclass(frozen=True)
class CandidateBenchmark:
    """Validation result for one hyperparameter candidate."""

    family: str
    candidate: str
    parameters: dict[str, Any]
    fit_seconds: float
    metrics: RegressionMetrics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


@dataclass(frozen=True)
class InferenceBenchmark:
    """Repeatable batch inference latency summary."""

    sample_rows: int
    repeats: int
    median_batch_ms: float
    p90_batch_ms: float
    median_ms_per_row: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def build_tree_preprocessor() -> ColumnTransformer:
    """Build the shared dense one-hot feature encoder for RF and XGBoost."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_random_forest_pipeline(parameters: dict[str, Any], random_seed: int) -> Pipeline:
    """Build a deployment-aligned Random Forest regression pipeline."""
    model = RandomForestRegressor(random_state=random_seed, **parameters)
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor()),
            ("model", model),
        ]
    )


def build_xgboost_pipeline(parameters: dict[str, Any], random_seed: int) -> Pipeline:
    """Build a deployment-aligned XGBoost regression pipeline."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_seed,
        verbosity=0,
        **parameters,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor()),
            ("model", model),
        ]
    )


def build_catboost_model(parameters: dict[str, Any], random_seed: int) -> Any:
    """Build CatBoost with native categorical feature handling."""
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        cat_features=CATEGORICAL_FEATURES,
        **parameters,
    )


def build_model(family: str, parameters: dict[str, Any], random_seed: int) -> Any:
    """Build one configured Phase 4 model family."""
    if family == "random_forest":
        return build_random_forest_pipeline(parameters, random_seed)
    if family == "xgboost":
        return build_xgboost_pipeline(parameters, random_seed)
    if family == "catboost":
        return build_catboost_model(parameters, random_seed)
    raise ValueError(f"Unsupported model family: {family}")


def fit_candidate(
    *,
    family: str,
    candidate: str,
    model: Any,
    parameters: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[Any, CandidateBenchmark, np.ndarray]:
    """Fit one candidate and score it only on the Phase 2 validation split."""
    started = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - started
    predictions = np.asarray(model.predict(x_validation), dtype=float)
    metrics = regression_metrics(y_validation, predictions)
    benchmark = CandidateBenchmark(
        family=family,
        candidate=candidate,
        parameters=parameters,
        fit_seconds=fit_seconds,
        metrics=metrics,
    )
    return model, benchmark, predictions


def fit_family_candidates(
    *,
    family: str,
    random_seed: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[Any, CandidateBenchmark, np.ndarray, list[CandidateBenchmark]]:
    """Tune one model family and return its best validation candidate."""
    if family not in PHASE4_CANDIDATES:
        raise ValueError(f"Unsupported model family: {family}")

    best_model: Any | None = None
    best_benchmark: CandidateBenchmark | None = None
    best_predictions: np.ndarray | None = None
    benchmarks: list[CandidateBenchmark] = []

    for candidate_name, parameters in PHASE4_CANDIDATES[family].items():
        print(f"Training {family}/{candidate_name}...", flush=True)
        fitted, benchmark, predictions = fit_candidate(
            family=family,
            candidate=candidate_name,
            model=build_model(family, parameters, random_seed),
            parameters=parameters,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )
        benchmarks.append(benchmark)
        print(
            f"Completed {family}/{candidate_name}: "
            f"RMSE={benchmark.metrics.rmse:.2f}, fit={benchmark.fit_seconds:.2f}s",
            flush=True,
        )
        if best_benchmark is None or benchmark.metrics.rmse < best_benchmark.metrics.rmse:
            best_model = fitted
            best_benchmark = benchmark
            best_predictions = predictions

    if best_model is None or best_benchmark is None or best_predictions is None:
        raise RuntimeError(f"No candidate completed for {family}")
    return best_model, best_benchmark, best_predictions, benchmarks


def measure_inference_latency(
    model: Any,
    features: pd.DataFrame,
    *,
    random_seed: int,
    sample_rows: int = 5_000,
    repeats: int = 5,
) -> InferenceBenchmark:
    """Measure warm, repeated batch inference on a deterministic validation sample."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if features.empty:
        raise ValueError("features cannot be empty")

    sample_size = min(sample_rows, len(features))
    sample = features.sample(n=sample_size, random_state=random_seed)
    warmup = sample.iloc[: min(128, sample_size)]
    model.predict(warmup)

    elapsed_ms: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        model.predict(sample)
        elapsed_ms.append((perf_counter() - started) * 1000.0)

    timings = np.asarray(elapsed_ms, dtype=float)
    median_batch_ms = float(np.median(timings))
    return InferenceBenchmark(
        sample_rows=sample_size,
        repeats=repeats,
        median_batch_ms=median_batch_ms,
        p90_batch_ms=float(np.quantile(timings, 0.90)),
        median_ms_per_row=median_batch_ms / sample_size,
    )


def save_benchmark_model(model: Any, path: str | Path) -> Path:
    """Persist one fitted benchmark model."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def model_size_mb(path: str | Path) -> float:
    """Return serialized model size in MiB."""
    return Path(path).stat().st_size / (1024.0**2)


def select_champion(comparison: pd.DataFrame) -> str:
    """Select the best validation model using RMSE, then production tie-breakers."""
    required = {"model", "rmse", "mae", "median_batch_ms", "model_size_mb"}
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"Missing champion-selection columns: {missing}")
    if comparison.empty:
        raise ValueError("comparison cannot be empty")

    ordered = comparison.sort_values(
        ["rmse", "mae", "median_batch_ms", "model_size_mb", "model"],
        ascending=[True, True, True, True, True],
    )
    return str(ordered.iloc[0]["model"])
