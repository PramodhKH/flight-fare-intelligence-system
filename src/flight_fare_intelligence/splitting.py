"""Leakage-aware, scenario-grouped train/validation/test splitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .analytics import BOOKING_HORIZON_BINS, BOOKING_HORIZON_LABELS
from .schema import MODEL_FEATURES, TARGET_COLUMN


@dataclass(frozen=True)
class SplitSummary:
    rows: dict[str, int]
    row_proportions: dict[str, float]
    scenarios: dict[str, int]
    scenario_overlap_count: int
    naive_row_random_scenario_overlap_count: int
    naive_row_random_rows_in_overlapping_scenarios: int
    strata: int
    minimum_scenarios_per_stratum: int
    target_mean: dict[str, float]
    target_median: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scenario_ids(df: pd.DataFrame) -> pd.Series:
    """Return a deterministic 64-bit hash for each deployed-feature vector."""
    return pd.util.hash_pandas_object(df[MODEL_FEATURES], index=False).astype("uint64")


def _strata(df: pd.DataFrame) -> pd.Series:
    horizon = pd.cut(
        df["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    )
    return (
        df["source_city"].astype(str)
        + ">"
        + df["destination_city"].astype(str)
        + "|"
        + df["class"].astype(str)
        + "|"
        + horizon.astype(str)
    )



def naive_row_random_overlap(
    df: pd.DataFrame,
    *,
    random_seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[int, int]:
    """Quantify exact-scenario overlap produced by a conventional row-random split."""
    scenario_id = _scenario_ids(df)
    row_index = pd.Series(range(len(df)))
    evaluation_fraction = validation_fraction + test_fraction
    train_rows, evaluation_rows = train_test_split(
        row_index,
        test_size=evaluation_fraction,
        random_state=random_seed,
    )
    test_share_of_evaluation = test_fraction / evaluation_fraction
    validation_rows, test_rows = train_test_split(
        evaluation_rows,
        test_size=test_share_of_evaluation,
        random_state=random_seed,
    )

    split = pd.Series(index=row_index.index, dtype="object")
    split.loc[train_rows] = "train"
    split.loc[validation_rows] = "validation"
    split.loc[test_rows] = "test"

    comparison = pd.DataFrame({"scenario_id": scenario_id, "split": split})
    split_counts = comparison.groupby("scenario_id")["split"].nunique()
    overlapping_ids = split_counts[split_counts > 1].index
    overlap_count = len(overlapping_ids)
    affected_rows = int(comparison["scenario_id"].isin(overlapping_ids).sum())
    return overlap_count, affected_rows

def create_split_assignments(
    df: pd.DataFrame,
    *,
    random_seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, SplitSummary]:
    """Create scenario-grouped, route/class/horizon-stratified split assignments.

    Identical deployed feature vectors are assigned wholly to one split. This prevents
    an exact production-input scenario from appearing in both training and evaluation.
    Stratification is performed at the unique-scenario level using directed route,
    cabin class, and booking-horizon bucket.
    """
    missing = [column for column in MODEL_FEATURES + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing split columns: {missing}")

    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive")

    scenario_id = _scenario_ids(df)
    scenario_frame = df[MODEL_FEATURES].copy()
    scenario_frame["scenario_id"] = scenario_id.values
    scenario_frame["stratum"] = _strata(df).values

    unique_scenarios = scenario_frame.drop_duplicates("scenario_id").copy()
    unique_feature_count = int(df[MODEL_FEATURES].drop_duplicates().shape[0])
    if len(unique_scenarios) != unique_feature_count:
        raise RuntimeError("Scenario hashing collision detected")

    min_scenarios = int(unique_scenarios.groupby("stratum", observed=True).size().min())
    if min_scenarios < 4:
        raise ValueError("At least four unique scenarios are required per stratum")

    evaluation_fraction = validation_fraction + test_fraction
    train_groups, evaluation_groups = train_test_split(
        unique_scenarios[["scenario_id", "stratum"]],
        test_size=evaluation_fraction,
        random_state=random_seed,
        stratify=unique_scenarios["stratum"],
    )

    test_share_of_evaluation = test_fraction / evaluation_fraction
    validation_groups, test_groups = train_test_split(
        evaluation_groups,
        test_size=test_share_of_evaluation,
        random_state=random_seed,
        stratify=evaluation_groups["stratum"],
    )

    split_map = pd.concat(
        [
            train_groups[["scenario_id"]].assign(split="train"),
            validation_groups[["scenario_id"]].assign(split="validation"),
            test_groups[["scenario_id"]].assign(split="test"),
        ],
        ignore_index=True,
    ).set_index("scenario_id")["split"]

    assignments = pd.DataFrame(
        {
            "record_id": df["Unnamed: 0"].values if "Unnamed: 0" in df.columns else df.index,
            "scenario_id": scenario_id.values,
        }
    )
    assignments["split"] = assignments["scenario_id"].map(split_map)

    if assignments["split"].isna().any():
        raise RuntimeError("Some rows did not receive a split assignment")

    scenario_split_counts = assignments.groupby("scenario_id")["split"].nunique()
    overlap_count = int((scenario_split_counts > 1).sum())
    if overlap_count:
        raise RuntimeError(f"Detected {overlap_count} scenarios spanning multiple splits")

    scored = assignments[["split"]].copy()
    scored[TARGET_COLUMN] = df[TARGET_COLUMN].to_numpy()

    row_counts = assignments["split"].value_counts().reindex(["train", "validation", "test"])
    scenario_counts = (
        assignments.drop_duplicates("scenario_id")["split"]
        .value_counts()
        .reindex(["train", "validation", "test"])
    )
    target_stats = scored.groupby("split")[TARGET_COLUMN].agg(["mean", "median"])
    naive_overlap_count, naive_affected_rows = naive_row_random_overlap(
        df,
        random_seed=random_seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    summary = SplitSummary(
        rows={key: int(value) for key, value in row_counts.items()},
        row_proportions={key: float(round(value / len(df), 6)) for key, value in row_counts.items()},
        scenarios={key: int(value) for key, value in scenario_counts.items()},
        scenario_overlap_count=overlap_count,
        naive_row_random_scenario_overlap_count=naive_overlap_count,
        naive_row_random_rows_in_overlapping_scenarios=naive_affected_rows,
        strata=int(unique_scenarios["stratum"].nunique()),
        minimum_scenarios_per_stratum=min_scenarios,
        target_mean={key: float(round(value, 2)) for key, value in target_stats["mean"].items()},
        target_median={key: float(round(value, 2)) for key, value in target_stats["median"].items()},
    )
    return assignments, summary
