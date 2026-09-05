"""Phase 2 market-intelligence analytics for the flight fare dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema import MODEL_FEATURES, TARGET_COLUMN

BOOKING_HORIZON_BINS = [0, 7, 14, 21, 35, 49]
BOOKING_HORIZON_LABELS = ["1-7", "8-14", "15-21", "22-35", "36-49"]


def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with route and booking-horizon analysis features."""
    required = {"source_city", "destination_city", "days_left", TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing analysis columns: {missing}")

    frame = df.copy()
    frame["route"] = frame["source_city"] + " → " + frame["destination_city"]
    frame["booking_horizon"] = pd.cut(
        frame["days_left"],
        bins=BOOKING_HORIZON_BINS,
        labels=BOOKING_HORIZON_LABELS,
        include_lowest=True,
    )
    return frame


def scenario_duplication_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Measure repeated deployed-feature scenarios and within-scenario fare variation."""
    missing = [column for column in MODEL_FEATURES + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for scenario analysis: {missing}")

    scenario_sizes = df.groupby(MODEL_FEATURES, dropna=False).size()
    repeated_groups = scenario_sizes[scenario_sizes > 1]

    scenario_prices = df.groupby(MODEL_FEATURES, dropna=False)[TARGET_COLUMN].agg(
        count="size", nunique="nunique", minimum="min", maximum="max"
    )
    repeated_prices = scenario_prices[scenario_prices["count"] > 1].copy()
    repeated_prices["price_range"] = repeated_prices["maximum"] - repeated_prices["minimum"]

    return {
        "rows_with_repeated_feature_vector": int(df.duplicated(MODEL_FEATURES).sum()),
        "unique_feature_vectors": int(df[MODEL_FEATURES].drop_duplicates().shape[0]),
        "repeated_feature_groups": len(repeated_groups),
        "maximum_rows_in_one_feature_group": int(repeated_groups.max()) if len(repeated_groups) else 1,
        "repeated_groups_with_price_variation": int((repeated_prices["nunique"] > 1).sum()),
        "median_within_group_price_range": float(repeated_prices["price_range"].median()),
        "mean_within_group_price_range": float(repeated_prices["price_range"].mean()),
        "maximum_within_group_price_range": float(repeated_prices["price_range"].max()),
    }


def _records(items: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in items[columns].to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            else:
                clean[key] = value
        result.append(clean)
    return result


def build_market_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact, JSON-serializable Phase 2 market summary."""
    frame = add_analysis_features(df)

    class_summary = (
        frame.groupby("class", observed=True)[TARGET_COLUMN]
        .agg(records="size", mean="mean", median="median", std="std")
        .reset_index()
    )
    class_summary[["mean", "median", "std"]] = class_summary[["mean", "median", "std"]].round(2)

    airline_summary = (
        frame.groupby(["class", "airline"], observed=True)[TARGET_COLUMN]
        .agg(records="size", mean="mean", median="median")
        .reset_index()
        .sort_values(["class", "mean"], ascending=[True, False])
    )
    airline_summary[["mean", "median"]] = airline_summary[["mean", "median"]].round(2)

    route_class = (
        frame.groupby(["class", "route"], observed=True)[TARGET_COLUMN]
        .agg(records="size", mean="mean", median="median")
        .reset_index()
    )
    route_class[["mean", "median"]] = route_class[["mean", "median"]].round(2)

    top_routes: dict[str, list[dict[str, Any]]] = {}
    bottom_routes: dict[str, list[dict[str, Any]]] = {}
    for cabin_class in sorted(frame["class"].unique()):
        subset = route_class[route_class["class"] == cabin_class].sort_values("mean", ascending=False)
        top_routes[cabin_class] = _records(subset.head(5), ["route", "records", "mean", "median"])
        bottom_routes[cabin_class] = _records(subset.tail(5), ["route", "records", "mean", "median"])

    horizon_summary = (
        frame.groupby(["class", "booking_horizon"], observed=True)[TARGET_COLUMN]
        .agg(records="size", mean="mean", median="median")
        .reset_index()
    )
    horizon_summary[["mean", "median"]] = horizon_summary[["mean", "median"]].round(2)
    horizon_summary["booking_horizon"] = horizon_summary["booking_horizon"].astype(str)

    stop_summary = (
        frame.groupby(["class", "stops"], observed=True)[TARGET_COLUMN]
        .agg(records="size", mean="mean", median="median")
        .reset_index()
    )
    stop_summary[["mean", "median"]] = stop_summary[["mean", "median"]].round(2)

    quantiles = frame[TARGET_COLUMN].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    return {
        "rows": len(frame),
        "routes": int(frame["route"].nunique()),
        "price_quantiles": {str(index): float(round(value, 2)) for index, value in quantiles.items()},
        "class_summary": _records(class_summary, ["class", "records", "mean", "median", "std"]),
        "airline_by_class": _records(
            airline_summary, ["class", "airline", "records", "mean", "median"]
        ),
        "top_routes_by_class": top_routes,
        "bottom_routes_by_class": bottom_routes,
        "booking_horizon_by_class": _records(
            horizon_summary,
            ["class", "booking_horizon", "records", "mean", "median"],
        ),
        "stops_by_class": _records(stop_summary, ["class", "stops", "records", "mean", "median"]),
        "scenario_duplication": scenario_duplication_summary(frame),
    }


def generate_eda_figures(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate the reproducible Phase 2 portfolio figures."""
    frame = add_analysis_features(df)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. Fare distribution by cabin class.
    figure, axis = plt.subplots(figsize=(8, 5))
    groups = [frame.loc[frame["class"] == name, TARGET_COLUMN] for name in ["Economy", "Business"]]
    axis.boxplot(groups, tick_labels=["Economy", "Business"], showfliers=False)
    axis.set_title("Fare Distribution by Cabin Class")
    axis.set_ylabel("Fare (INR)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_path / "phase2_fare_distribution_by_class.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)

    # 2. Daily booking-horizon behavior by cabin class.
    horizon = (
        frame.groupby(["days_left", "class"], observed=True)[TARGET_COLUMN]
        .median()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for cabin_class in ["Economy", "Business"]:
        subset = horizon[horizon["class"] == cabin_class]
        axis.plot(subset["days_left"], subset[TARGET_COLUMN], label=cabin_class)
    axis.set_title("Median Fare vs Days Until Departure")
    axis.set_xlabel("Days Until Departure")
    axis.set_ylabel("Median Fare (INR)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_path / "phase2_fare_vs_days_left.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)

    # 3. Median fare by airline and class.
    airline = (
        frame.groupby(["airline", "class"], observed=True)[TARGET_COLUMN]
        .median()
        .unstack("class")
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    airline.plot(kind="bar", ax=axis)
    axis.set_title("Median Fare by Airline and Cabin Class")
    axis.set_xlabel("Airline")
    axis.set_ylabel("Median Fare (INR)")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_path / "phase2_median_fare_by_airline_class.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)

    # 4. Stop-count effect by class.
    stop_order = ["zero", "one", "two_or_more"]
    stops = (
        frame.groupby(["stops", "class"], observed=True)[TARGET_COLUMN]
        .median()
        .unstack("class")
        .reindex(stop_order)
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    stops.plot(kind="bar", ax=axis)
    axis.set_title("Median Fare by Stops and Cabin Class")
    axis.set_xlabel("Stops")
    axis.set_ylabel("Median Fare (INR)")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_path / "phase2_median_fare_by_stops_class.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)

    # 5. Economy route heatmap.
    economy = frame[frame["class"] == "Economy"]
    economy_matrix = economy.pivot_table(
        index="source_city", columns="destination_city", values=TARGET_COLUMN, aggfunc="median"
    ).sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(economy_matrix.values, aspect="auto")
    axis.set_title("Median Economy Fare by Route")
    axis.set_xticks(range(len(economy_matrix.columns)), economy_matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(economy_matrix.index)), economy_matrix.index)
    figure.colorbar(image, ax=axis, label="Median Fare (INR)")
    figure.tight_layout()
    path = output_path / "phase2_route_heatmap_economy.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)

    # 6. Business route heatmap.
    business = frame[frame["class"] == "Business"]
    business_matrix = business.pivot_table(
        index="source_city", columns="destination_city", values=TARGET_COLUMN, aggfunc="median"
    ).sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(business_matrix.values, aspect="auto")
    axis.set_title("Median Business Fare by Route")
    axis.set_xticks(range(len(business_matrix.columns)), business_matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(business_matrix.index)), business_matrix.index)
    figure.colorbar(image, ax=axis, label="Median Fare (INR)")
    figure.tight_layout()
    path = output_path / "phase2_route_heatmap_business.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)

    return generated
