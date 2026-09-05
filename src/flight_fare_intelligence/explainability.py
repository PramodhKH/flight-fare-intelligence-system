"""Phase 6 SHAP utilities for the locked XGBoost fare model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .schema import MODEL_FEATURES


@dataclass(frozen=True)
class ShapResult:
    """Raw-feature SHAP attributions and additivity diagnostics."""

    expected_value: float
    raw_values: pd.DataFrame
    predicted_from_shap: np.ndarray
    max_reconstruction_error: float
    method: str


def _pipeline_parts(pipeline: Pipeline) -> tuple[Any, Any]:
    """Return the fitted preprocessor and XGBoost estimator from the pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Phase 6 expects the Phase 4 sklearn Pipeline champion")
    if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Champion pipeline must expose 'preprocessor' and 'model' steps")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "get_booster"):
        raise TypeError("Phase 6 explainability currently supports the XGBoost champion")
    return preprocessor, model


def transformed_feature_groups(
    transformed_feature_names: list[str] | np.ndarray,
) -> dict[str, list[int]]:
    """Map transformed one-hot/numeric columns back to the eight deployed features."""
    groups: dict[str, list[int]] = {feature: [] for feature in MODEL_FEATURES}
    names = [str(name) for name in transformed_feature_names]

    for index, name in enumerate(names):
        matched = False
        for feature in MODEL_FEATURES:
            categorical_prefix = f"categorical__{feature}_"
            numeric_name = f"numeric__{feature}"
            if name.startswith(categorical_prefix) or name == numeric_name:
                groups[feature].append(index)
                matched = True
                break
        if not matched:
            raise ValueError(f"Unable to map transformed feature back to raw contract: {name}")

    missing = [feature for feature, indices in groups.items() if not indices]
    if missing:
        raise ValueError(f"No transformed columns found for raw features: {missing}")
    return groups


def _xgboost_contributions(
    pipeline: Pipeline,
    features: pd.DataFrame,
    *,
    approximate: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return XGBoost TreeSHAP contributions in transformed feature space."""
    if features.empty:
        raise ValueError("features cannot be empty")
    missing = [feature for feature in MODEL_FEATURES if feature not in features.columns]
    if missing:
        raise ValueError(f"Missing deployed features for SHAP: {missing}")

    preprocessor, model = _pipeline_parts(pipeline)
    transformed = np.asarray(
        preprocessor.transform(features[MODEL_FEATURES]),
        dtype=np.float32,
    )
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]

    model_predictions = np.asarray(pipeline.predict(features[MODEL_FEATURES]), dtype=float)

    if approximate:
        from xgboost import DMatrix

        matrix = DMatrix(transformed, feature_names=transformed_names)
        contributions = np.asarray(
            model.get_booster().predict(
                matrix,
                pred_contribs=True,
                approx_contribs=True,
            ),
            dtype=float,
        )
        if contributions.shape[1] != transformed.shape[1] + 1:
            raise RuntimeError("Unexpected XGBoost SHAP contribution shape")
        expected_values = contributions[:, -1]
        if not np.allclose(expected_values, expected_values[0], atol=1e-6):
            raise RuntimeError("SHAP base value unexpectedly varies across rows")
        return contributions[:, :-1], float(expected_values[0]), model_predictions

    import shap

    explainer = shap.TreeExplainer(model)
    exact_values = np.asarray(
        explainer.shap_values(transformed, check_additivity=True),
        dtype=float,
    )
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return exact_values, expected_value, model_predictions


def explain_raw_features(
    pipeline: Pipeline,
    features: pd.DataFrame,
    *,
    approximate: bool = True,
) -> ShapResult:
    """Compute additive SHAP values aggregated to the deployed raw feature contract."""
    transformed_values, expected_value, model_predictions = _xgboost_contributions(
        pipeline,
        features,
        approximate=approximate,
    )
    preprocessor, _ = _pipeline_parts(pipeline)
    transformed_names = preprocessor.get_feature_names_out()
    groups = transformed_feature_groups(transformed_names)

    raw_values = pd.DataFrame(index=features.index)
    for feature in MODEL_FEATURES:
        raw_values[feature] = transformed_values[:, groups[feature]].sum(axis=1)

    reconstructed = expected_value + raw_values[MODEL_FEATURES].sum(axis=1).to_numpy()
    max_error = float(np.max(np.abs(reconstructed - model_predictions)))
    method = "approximate_treeshap" if approximate else "exact_treeshap"
    return ShapResult(
        expected_value=expected_value,
        raw_values=raw_values,
        predicted_from_shap=reconstructed,
        max_reconstruction_error=max_error,
        method=method,
    )


def global_importance(raw_shap: pd.DataFrame) -> pd.DataFrame:
    """Rank deployed features by mean absolute raw-feature SHAP contribution."""
    missing = [feature for feature in MODEL_FEATURES if feature not in raw_shap.columns]
    if missing:
        raise ValueError(f"Missing SHAP columns: {missing}")

    mean_abs = raw_shap[MODEL_FEATURES].abs().mean()
    total = float(mean_abs.sum())
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_abs_shap_inr": [float(mean_abs[feature]) for feature in MODEL_FEATURES],
            "mean_shap_inr": [float(raw_shap[feature].mean()) for feature in MODEL_FEATURES],
        }
    )
    if total > 0.0:
        importance["importance_percent"] = importance["mean_abs_shap_inr"] / total * 100.0
    else:
        importance["importance_percent"] = 0.0
    return importance.sort_values("mean_abs_shap_inr", ascending=False).reset_index(drop=True)


def segment_importance(
    raw_shap: pd.DataFrame,
    metadata: pd.DataFrame,
    segment_masks: dict[str, pd.Series | np.ndarray],
) -> pd.DataFrame:
    """Calculate raw-feature SHAP importance inside predefined reliability segments."""
    if len(raw_shap) != len(metadata):
        raise ValueError("SHAP and metadata row counts must match")

    rows: list[dict[str, float | int | str]] = []
    for segment_name, mask in segment_masks.items():
        mask_array = np.asarray(mask, dtype=bool)
        if len(mask_array) != len(raw_shap):
            raise ValueError(f"Segment mask length mismatch: {segment_name}")
        if not mask_array.any():
            continue

        subset = raw_shap.loc[mask_array, MODEL_FEATURES]
        mean_abs = subset.abs().mean()
        total = float(mean_abs.sum())
        for feature in MODEL_FEATURES:
            value = float(mean_abs[feature])
            rows.append(
                {
                    "segment": segment_name,
                    "rows": int(mask_array.sum()),
                    "feature": feature,
                    "mean_abs_shap_inr": value,
                    "importance_percent": value / total * 100.0 if total > 0.0 else 0.0,
                    "mean_shap_inr": float(subset[feature].mean()),
                }
            )
    return pd.DataFrame(rows)


def categorical_effects(
    features: pd.DataFrame,
    raw_shap: pd.DataFrame,
    categorical_features: list[str],
    *,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Summarize average SHAP direction by observed categorical feature value."""
    if len(features) != len(raw_shap):
        raise ValueError("features and raw_shap row counts must match")
    if min_rows < 1:
        raise ValueError("min_rows must be positive")

    rows: list[dict[str, float | int | str]] = []
    for feature in categorical_features:
        if feature not in features.columns or feature not in raw_shap.columns:
            raise ValueError(f"Missing categorical effect feature: {feature}")
        values = features[feature].astype(str)
        for value, indices in values.groupby(values, observed=True).groups.items():
            positions = features.index.get_indexer(indices)
            positions = positions[positions >= 0]
            if len(positions) < min_rows:
                continue
            shap_values = raw_shap.iloc[positions][feature]
            rows.append(
                {
                    "feature": feature,
                    "value": str(value),
                    "rows": len(positions),
                    "mean_shap_inr": float(shap_values.mean()),
                    "mean_abs_shap_inr": float(shap_values.abs().mean()),
                    "median_shap_inr": float(shap_values.median()),
                }
            )
    return pd.DataFrame(rows)


def select_representative_index(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    error_column: str = "absolute_error",
) -> int:
    """Select a deterministic median-error case from one diagnostic segment."""
    mask_array = np.asarray(mask, dtype=bool)
    if len(mask_array) != len(frame):
        raise ValueError("Representative-case mask length does not match frame")
    segment = frame.loc[mask_array]
    if segment.empty:
        raise ValueError("Cannot choose a representative case from an empty segment")
    if error_column not in segment.columns:
        raise ValueError(f"Missing error column: {error_column}")

    target = float(segment[error_column].median())
    distance = (segment[error_column] - target).abs()
    candidates = segment.loc[distance == distance.min()].sort_values("record_id")
    return int(candidates.index[0])


def local_explanation_table(
    *,
    case_label: str,
    record_id: int,
    feature_row: pd.Series,
    shap_row: pd.Series,
    expected_value: float,
    predicted_price: float,
    actual_price: float,
) -> pd.DataFrame:
    """Create a ranked, display-ready local SHAP explanation table."""
    rows: list[dict[str, float | int | str]] = []
    for feature in MODEL_FEATURES:
        contribution = float(shap_row[feature])
        rows.append(
            {
                "case_label": case_label,
                "record_id": record_id,
                "actual_price": float(actual_price),
                "predicted_price": float(predicted_price),
                "expected_value": float(expected_value),
                "feature": feature,
                "feature_value": str(feature_row[feature]),
                "shap_value_inr": contribution,
                "direction": "increases_fare" if contribution >= 0.0 else "decreases_fare",
                "absolute_shap_inr": abs(contribution),
            }
        )
    result = pd.DataFrame(rows).sort_values("absolute_shap_inr", ascending=False)
    result["rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)
