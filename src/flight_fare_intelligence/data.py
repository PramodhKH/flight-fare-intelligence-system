"""Data loading and validation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    EXPECTED_CATEGORIES,
    EXPECTED_DAYS_LEFT_RANGE,
    EXPECTED_STRICT_ROUTES,
    EXPECTED_STRICT_ROWS,
    OPTIONAL_EXPORT_COLUMNS,
    RAW_REQUIRED_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when the flight booking dataset violates its contract."""


@dataclass(frozen=True)
class ValidationSummary:
    path: str
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    routes: int
    min_price: float
    max_price: float
    min_duration: float
    max_duration: float
    min_days_left: int
    max_days_left: int
    strict: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise DataValidationError(f"Expected a CSV file, received: {path.name}")
    return pd.read_csv(path)


def validate_dataset(
    df: pd.DataFrame, *, strict: bool = False, path: str = "<dataframe>"
) -> ValidationSummary:
    missing_columns = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    unexpected = [c for c in df.columns if c not in RAW_REQUIRED_COLUMNS + OPTIONAL_EXPORT_COLUMNS]
    if unexpected:
        raise DataValidationError(f"Unexpected columns: {unexpected}")

    if df[RAW_REQUIRED_COLUMNS].isna().any().any():
        counts = df[RAW_REQUIRED_COLUMNS].isna().sum()
        bad = counts[counts > 0].to_dict()
        raise DataValidationError(f"Missing values found: {bad}")

    for column, allowed in EXPECTED_CATEGORIES.items():
        observed = set(df[column].astype(str).unique())
        invalid = sorted(observed - allowed)
        if invalid:
            raise DataValidationError(f"Unexpected values in {column}: {invalid}")

    if not pd.api.types.is_numeric_dtype(df["duration"]):
        raise DataValidationError("duration must be numeric")
    if not pd.api.types.is_numeric_dtype(df["days_left"]):
        raise DataValidationError("days_left must be numeric")
    if not pd.api.types.is_numeric_dtype(df["price"]):
        raise DataValidationError("price must be numeric")

    if (df["duration"] <= 0).any():
        raise DataValidationError("duration must be > 0")
    if (df["price"] <= 0).any():
        raise DataValidationError("price must be > 0")

    min_days, max_days = EXPECTED_DAYS_LEFT_RANGE
    if not df["days_left"].between(min_days, max_days).all():
        raise DataValidationError(f"days_left must be between {min_days} and {max_days}")

    same_city = df["source_city"].eq(df["destination_city"])
    if same_city.any():
        raise DataValidationError(
            f"Found {int(same_city.sum())} rows with identical source and destination"
        )

    if "Unnamed: 0" in df.columns:
        export_index = df["Unnamed: 0"]
        if export_index.isna().any() or export_index.duplicated().any():
            raise DataValidationError("Unnamed: 0 must be a unique export index when present")

    routes = int(df[["source_city", "destination_city"]].drop_duplicates().shape[0])
    if strict:
        if len(df) != EXPECTED_STRICT_ROWS:
            raise DataValidationError(
                f"Strict validation expected {EXPECTED_STRICT_ROWS:,} rows; found {len(df):,}"
            )
        if routes != EXPECTED_STRICT_ROUTES:
            raise DataValidationError(
                f"Strict validation expected {EXPECTED_STRICT_ROUTES} directed routes; found {routes}"
            )
        if "Unnamed: 0" in df.columns:
            expected = pd.RangeIndex(start=0, stop=len(df), step=1)
            if not export_index.reset_index(drop=True).equals(
                pd.Series(expected, dtype=export_index.dtype)
            ):
                raise DataValidationError(
                    "Strict validation expected Unnamed: 0 to be a contiguous 0-based index"
                )

    return ValidationSummary(
        path=path,
        rows=len(df),
        columns=len(df.columns),
        missing_values=int(df[RAW_REQUIRED_COLUMNS].isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        routes=routes,
        min_price=float(df["price"].min()),
        max_price=float(df["price"].max()),
        min_duration=float(df["duration"].min()),
        max_duration=float(df["duration"].max()),
        min_days_left=int(df["days_left"].min()),
        max_days_left=int(df["days_left"].max()),
        strict=strict,
    )
