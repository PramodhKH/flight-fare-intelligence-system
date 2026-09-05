"""Model-facing feature selection."""

from __future__ import annotations

import pandas as pd

from .schema import MODEL_FEATURES, TARGET_COLUMN


def select_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the deployment-aligned feature matrix and regression target."""
    missing = [c for c in MODEL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    return df[MODEL_FEATURES].copy(), df[TARGET_COLUMN].copy()
