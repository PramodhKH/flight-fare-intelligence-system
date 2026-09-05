"""Presentation helpers that keep Streamlit rendering thin and testable."""

from __future__ import annotations

from typing import Any

import pandas as pd

DISPLAY_NAMES = {
    "Air_India": "Air India",
    "GO_FIRST": "Go First",
    "Early_Morning": "Early Morning",
    "Late_Night": "Late Night",
    "two_or_more": "2+ Stops",
    "zero": "Nonstop",
    "one": "1 Stop",
}


def pretty_label(value: Any) -> str:
    """Convert canonical backend category values to dashboard-friendly text."""
    text = str(value)
    return DISPLAY_NAMES.get(text, text.replace("_", " ").title())


def format_inr(value: float, *, decimals: int = 0) -> str:
    """Format a numeric fare using Indian-rupee display conventions."""
    if decimals:
        return f"₹{float(value):,.{decimals}f}"
    return f"₹{float(value):,.0f}"


def prediction_cards(payload: dict[str, Any]) -> dict[str, str]:
    """Flatten prediction payload into the five headline dashboard cards."""
    interval = payload["prediction_interval"]
    opportunity = payload["fare_opportunity"]
    reliability = payload["reliability"]
    guidance = payload["booking_guidance"]
    return {
        "predicted_fare": format_inr(payload["predicted_fare"]),
        "expected_range": (f"{format_inr(interval['lower'])} – {format_inr(interval['upper'])}"),
        "opportunity": f"{int(opportunity['fare_opportunity_score'])} / 100",
        "opportunity_label": pretty_label(opportunity["fare_position"]),
        "reliability": f"{int(reliability['score'])} / 100",
        "reliability_label": pretty_label(reliability["label"]),
        "guidance": pretty_label(guidance["recommendation"]),
    }


def shap_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Return sorted raw-feature SHAP drivers for a horizontal contribution chart."""
    rows = payload.get("explanation", {}).get("top_drivers", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["feature", "feature_value", "contribution_inr", "label"])
    frame = frame.copy()
    frame["label"] = frame.apply(
        lambda row: f"{pretty_label(row['feature'])}: {pretty_label(row['feature_value'])}",
        axis=1,
    )
    return frame.sort_values("contribution_inr")


def booking_curve_frame(payload: dict[str, Any], *, current_days_left: int) -> pd.DataFrame:
    """Normalize API booking-horizon rows and mark the active scenario."""
    frame = pd.DataFrame(payload.get("curve", []))
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "days_left",
                "predicted_fare",
                "historical_horizon_median",
                "is_current",
            ]
        )
    frame = frame.sort_values("days_left").copy()
    frame["is_current"] = frame["days_left"].astype(int).eq(int(current_days_left))
    return frame


def route_horizon_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize training-only route/class horizon analytics for charting."""
    frame = pd.DataFrame(payload.get("horizons", []))
    order = ["1-7", "8-14", "15-21", "22-35", "36-49"]
    if frame.empty:
        return pd.DataFrame(columns=["booking_horizon", "median", "q10", "q90", "rows"])
    frame["booking_horizon"] = pd.Categorical(
        frame["booking_horizon"], categories=order, ordered=True
    )
    return frame.sort_values("booking_horizon")


def what_if_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize counterfactual rows for tabular and chart display."""
    frame = pd.DataFrame(payload.get("scenarios", []))
    if frame.empty:
        return pd.DataFrame(columns=["changed_value", "predicted_fare", "difference_from_base"])
    frame = frame.copy()
    if "changed_value" not in frame.columns:
        changed_feature = str(payload.get("changed_feature", "value"))
        if changed_feature in frame.columns:
            frame["changed_value"] = frame[changed_feature]
    frame["display_value"] = frame["changed_value"].map(pretty_label)
    return frame.sort_values("predicted_fare")


def recommendation_copy(prediction: dict[str, Any]) -> str:
    """Return cautious product copy for the model-based booking guidance."""
    guidance = prediction["booking_guidance"]
    recommendation = str(guidance["recommendation"])
    change = float(guidance["expected_wait_change_percent"])
    if recommendation == "BUY_NOW":
        return (
            f"The model estimates fares may rise about {abs(change):.1f}% over the next "
            "7 days, while the current fare is comparatively attractive."
        )
    if recommendation == "WAIT_OR_MONITOR":
        return (
            f"The model estimates fares may soften about {abs(change):.1f}% over the next "
            "7 days. Monitor the market rather than treating this as a guaranteed forecast."
        )
    return (
        "Current value and booking-horizon signals are mixed. Monitor the fare rather than "
        "acting on a single model signal."
    )
