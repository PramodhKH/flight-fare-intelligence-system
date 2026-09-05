"""Phase 9 Streamlit decision-support dashboard."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from frontend.client import APIClientError, FlightFareAPIClient
from frontend.dashboard_data import (
    AIRLINES,
    CABIN_CLASSES,
    CITIES,
    DEPARTURE_PERIODS,
    STOP_OPTIONS,
    build_dashboard_bundle,
    default_scenario,
)
from frontend.viewmodels import (
    booking_curve_frame,
    format_inr,
    prediction_cards,
    pretty_label,
    recommendation_copy,
    route_horizon_frame,
    shap_frame,
    what_if_frame,
)

st.set_page_config(
    page_title="Flight Fare Intelligence System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
[data-testid="stAppViewContainer"] {background: #f7f9fc;}
[data-testid="stSidebar"] {background: #eef4ff; border-right: 1px solid #d8e4f7;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1600px;}
h1, h2, h3 {color: #17233c;}
.metric-card {
  background: white;
  border: 1px solid #e3eaf5;
  border-radius: 16px;
  padding: 18px 18px 15px 18px;
  min-height: 128px;
  box-shadow: 0 4px 18px rgba(35, 55, 90, 0.06);
}
.metric-title {font-size: 0.82rem; color: #60708d; font-weight: 700; margin-bottom: 8px;}
.metric-value {font-size: 1.55rem; color: #1769e0; font-weight: 800; line-height: 1.15;}
.metric-sub {font-size: 0.82rem; color: #6b778c; margin-top: 9px;}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #fff1df;
  color: #9b5c00;
  font-size: 0.76rem;
  font-weight: 700;
}
.panel {
  background: white;
  border: 1px solid #e3eaf5;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 4px 18px rgba(35, 55, 90, 0.05);
}
.status-ok {color: #0a8f55; font-weight: 700;}
.status-bad {color: #c33b32; font-weight: 700;}
.small-muted {color: #73809a; font-size: 0.84rem;}
.header-pill-row {display: flex; gap: 8px; justify-content: flex-start; flex-wrap: wrap;}
.header-pill-row-inline {margin-top: -0.15rem; margin-bottom: 0.9rem;}
.header-pill {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid #dce6f5;
  background: #ffffff;
  color: #50617d;
  font-size: 0.74rem;
  font-weight: 700;
  white-space: nowrap;
}
.header-pill-live {border-color: #bfe8d5; background: #edf9f3; color: #0a8f55;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def api_client() -> FlightFareAPIClient:
    """Create one reusable backend client per Streamlit server process."""
    return FlightFareAPIClient()


def metric_card(title: str, value: str, *, subtitle: str = "", badge: str = "") -> None:
    """Render a compact dashboard KPI card."""
    badge_html = (
        f'<div class="metric-sub"><span class="badge">{badge}</span></div>' if badge else ""
    )
    subtitle_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        (
            '<div class="metric-card">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f"{badge_html}{subtitle_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def scenario_from_sidebar() -> dict[str, Any]:
    """Render canonical input controls and return the validated API-shaped scenario."""
    defaults = default_scenario()
    st.sidebar.markdown("## ✈️ Flight Scenario")

    source = st.sidebar.selectbox(
        "From",
        CITIES,
        index=CITIES.index(str(defaults["source_city"])),
    )
    destinations = [city for city in CITIES if city != source]
    default_destination = str(defaults["destination_city"])
    destination_index = (
        destinations.index(default_destination) if default_destination in destinations else 0
    )
    destination = st.sidebar.selectbox("To", destinations, index=destination_index)
    airline = st.sidebar.selectbox(
        "Airline",
        AIRLINES,
        index=AIRLINES.index(str(defaults["airline"])),
        format_func=pretty_label,
    )
    cabin_class = st.sidebar.selectbox(
        "Class",
        CABIN_CLASSES,
        index=CABIN_CLASSES.index(str(defaults["class"])),
    )
    departure = st.sidebar.selectbox(
        "Departure period",
        DEPARTURE_PERIODS,
        index=DEPARTURE_PERIODS.index(str(defaults["departure_time"])),
        format_func=pretty_label,
    )
    stops = st.sidebar.selectbox(
        "Stops",
        STOP_OPTIONS,
        index=STOP_OPTIONS.index(str(defaults["stops"])),
        format_func=pretty_label,
    )
    duration = st.sidebar.number_input(
        "Duration (hours)",
        min_value=0.5,
        max_value=50.0,
        value=float(defaults["duration"]),
        step=0.25,
    )
    days_left = st.sidebar.slider(
        "Days before departure",
        min_value=1,
        max_value=49,
        value=int(defaults["days_left"]),
    )

    return {
        "airline": airline,
        "source_city": source,
        "destination_city": destination,
        "departure_time": departure,
        "stops": stops,
        "class": cabin_class,
        "duration": float(duration),
        "days_left": int(days_left),
    }


def load_dashboard(client: FlightFareAPIClient, scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch and retain a dashboard bundle without crashing the UI on API errors."""
    try:
        bundle = build_dashboard_bundle(client, scenario)
    except APIClientError as exc:
        st.error(str(exc))
        st.info("Start the backend in another terminal with `make api`, then retry.")
        return None
    st.session_state["dashboard_bundle"] = bundle
    st.session_state["dashboard_scenario"] = scenario
    return bundle


def render_header(client: FlightFareAPIClient) -> None:
    """Render the product title and status badges in normal page flow."""
    st.title("✈️ Flight Fare Intelligence System")
    st.caption("Explainable airfare prediction, uncertainty, and route intelligence")

    try:
        health = client.health()
        metadata = client.model_metadata()
        is_live = health.get("status") == "ok"
        champion = str(metadata.get("champion", "xgboost")).upper()
        coverage = float(metadata.get("nominal_interval_coverage", 0.9)) * 100.0
        live_class = "header-pill header-pill-live" if is_live else "header-pill"
        live_text = "● API Connected" if is_live else "● API Degraded"
        st.markdown(
            (
                '<div class="header-pill-row header-pill-row-inline">'
                f'<span class="{live_class}">{live_text}</span>'
                f'<span class="header-pill">{champion} · '
                f"{coverage:.0f}% conformal interval</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    except APIClientError:
        st.markdown(
            (
                '<div class="header-pill-row header-pill-row-inline">'
                '<span class="header-pill">● API Offline</span></div>'
            ),
            unsafe_allow_html=True,
        )


def render_metric_row(prediction: dict[str, Any]) -> None:
    cards = prediction_cards(prediction)
    columns = st.columns(5)
    with columns[0]:
        metric_card("Predicted Fare", cards["predicted_fare"])
    with columns[1]:
        metric_card("Expected Range", cards["expected_range"], subtitle="90% model interval")
    with columns[2]:
        metric_card(
            "Fare Opportunity Score",
            cards["opportunity"],
            badge=cards["opportunity_label"],
        )
    with columns[3]:
        metric_card(
            "Reliability",
            cards["reliability"],
            badge=cards["reliability_label"],
        )
    with columns[4]:
        metric_card(
            "Guidance",
            cards["guidance"],
            subtitle=recommendation_copy(prediction),
        )


def render_shap(prediction: dict[str, Any]) -> None:
    st.subheader("Why this fare?")
    frame = shap_frame(prediction)
    if frame.empty:
        st.info("No explanation payload was returned.")
        return
    colors = ["#159d6d" if value < 0 else "#ef6f6c" for value in frame["contribution_inr"]]
    figure = go.Figure(
        go.Bar(
            x=frame["contribution_inr"],
            y=frame["label"],
            orientation="h",
            marker_color=colors,
            text=[f"₹{value:+,.0f}" for value in frame["contribution_inr"]],
            textposition=[
                "inside" if value < 0 else "outside" for value in frame["contribution_inr"]
            ],
            cliponaxis=False,
        )
    )
    figure.update_layout(
        height=385,
        margin={"l": 145, "r": 125, "t": 10, "b": 35},
        xaxis_title="Impact on predicted fare (₹)",
        yaxis_title="",
        showlegend=False,
    )
    figure.update_yaxes(automargin=True)
    st.plotly_chart(figure, use_container_width=True)
    st.caption("SHAP explains the model estimate; it does not measure prediction certainty.")


def render_booking_horizon(bundle: dict[str, Any], scenario: dict[str, Any]) -> None:
    st.subheader("Booking-Horizon Counterfactual Intelligence")
    st.caption("How the model estimate changes when only `days_left` is varied.")
    frame = booking_curve_frame(
        bundle["booking_horizon"],
        current_days_left=int(scenario["days_left"]),
    )
    if frame.empty:
        st.info("Booking-horizon intelligence is unavailable for this scenario.")
        return
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["days_left"],
            y=frame["predicted_fare"],
            mode="lines+markers",
            name="Model what-if",
            line={"color": "#1769e0", "width": 2},
            marker={"size": 4},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["days_left"],
            y=frame["historical_horizon_median"],
            mode="lines",
            name="Historical route/class median",
            line={"color": "#8190aa", "width": 2, "dash": "dash", "shape": "hv"},
        )
    )
    current = frame.loc[frame["is_current"]]
    if not current.empty:
        current_days = int(current["days_left"].iloc[0])
        current_fare = float(current["predicted_fare"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=[current_days],
                y=[current_fare],
                mode="markers",
                name="Current scenario",
                marker={"size": 12, "color": "#f06449"},
            )
        )
        figure.add_annotation(
            x=current_days,
            y=current_fare,
            text=f"Current · {current_days} days · {format_inr(current_fare)}",
            showarrow=True,
            arrowhead=2,
            ax=45,
            ay=-42,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#d8e4f7",
            borderpad=6,
            font={"size": 11, "color": "#33445f"},
        )
    figure.update_layout(
        height=360,
        margin={"l": 10, "r": 20, "t": 10, "b": 20},
        xaxis_title="Days before departure",
        yaxis_title="Fare (₹)",
        hovermode="x unified",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "The solid line is a model counterfactual, not a guaranteed future-fare trajectory. "
        "The stepped dashed benchmark summarizes historical route/class booking-horizon medians."
    )


def render_route_analytics(bundle: dict[str, Any], prediction: dict[str, Any]) -> None:
    st.subheader("Route Analytics")
    route = bundle["route_analytics"]
    opportunity = prediction["fare_opportunity"]
    horizon = route_horizon_frame(route)

    row_one = st.columns(2)
    with row_one[0]:
        st.metric("Comparable median", format_inr(opportunity["benchmark_median"]))
    with row_one[1]:
        st.metric("Route", str(opportunity["benchmark_route"]).replace(">", " → "))

    row_two = st.columns(2)
    with row_two[0]:
        st.metric("Cabin", str(opportunity["benchmark_class"]))
    with row_two[1]:
        st.metric("Booking horizon", str(opportunity["benchmark_horizon"]))

    if not horizon.empty:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=horizon["booking_horizon"],
                y=horizon["q90"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=horizon["booking_horizon"],
                y=horizon["q10"],
                mode="lines",
                fill="tonexty",
                name="Historical 10th–90th percentile",
                line={"width": 0},
                fillcolor="rgba(23, 105, 224, 0.12)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=horizon["booking_horizon"],
                y=horizon["median"],
                mode="lines+markers",
                name="Historical median",
                line={"color": "#1769e0", "width": 3},
            )
        )
        figure.update_layout(
            height=300,
            margin={"l": 10, "r": 20, "t": 10, "b": 20},
            xaxis_title="Booking horizon",
            yaxis_title="Historical fare (₹)",
        )
        st.plotly_chart(figure, use_container_width=True)


def render_what_if(bundle: dict[str, Any]) -> None:
    st.subheader("What-if Simulator")
    st.caption("Vary one input while holding all other model inputs fixed.")
    tabs = st.tabs(["Airline", "Departure time", "Stops"])
    specs = [
        (tabs[0], "airline"),
        (tabs[1], "departure_time"),
        (tabs[2], "stops"),
    ]
    for tab, feature in specs:
        with tab:
            frame = what_if_frame(bundle["what_if"][feature])
            if frame.empty:
                st.info("No counterfactual values available.")
                continue
            figure = go.Figure(
                go.Bar(
                    x=frame["display_value"],
                    y=frame["predicted_fare"],
                    text=[format_inr(value) for value in frame["predicted_fare"]],
                    textposition="outside",
                    marker_color="#5f8fe8",
                )
            )
            figure.update_layout(
                height=300,
                margin={"l": 10, "r": 20, "t": 10, "b": 20},
                xaxis_title="",
                yaxis_title="Predicted fare (₹)",
            )
            st.plotly_chart(figure, use_container_width=True)
            best = frame.iloc[0]
            delta = float(best["difference_from_base"])
            st.success(
                f"Lowest model estimate: {best['display_value']} at "
                f"{format_inr(best['predicted_fare'])} ({delta:+,.0f} vs current)."
            )


def render_dashboard(bundle: dict[str, Any], scenario: dict[str, Any]) -> None:
    prediction = bundle["prediction"]
    render_metric_row(prediction)
    st.write("")
    left, right = st.columns([1, 1])
    with left:
        render_shap(prediction)
    with right:
        render_booking_horizon(bundle, scenario)
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        render_route_analytics(bundle, prediction)
    with right:
        render_what_if(bundle)
    st.warning(prediction["warning"], icon="⚠️")


def render_route_explorer(bundle: dict[str, Any], prediction: dict[str, Any]) -> None:
    st.header("🗺️ Route Explorer")
    render_route_analytics(bundle, prediction)
    st.info("All route benchmarks use training-only historical comparable fares.")


def render_fare_trends(bundle: dict[str, Any], scenario: dict[str, Any]) -> None:
    st.header("📈 Fare Trends")
    render_booking_horizon(bundle, scenario)
    route_horizon = route_horizon_frame(bundle["route_analytics"])
    if not route_horizon.empty:
        st.dataframe(route_horizon, use_container_width=True, hide_index=True)


def render_model_insights(client: FlightFareAPIClient, bundle: dict[str, Any]) -> None:
    st.header("🧠 Model Insights")
    prediction = bundle["prediction"]
    try:
        metadata = client.model_metadata()
        telemetry = client.telemetry()
    except APIClientError as exc:
        st.error(str(exc))
        return

    cols = st.columns(4)
    cols[0].metric("Champion", str(metadata["champion"]).upper())
    cols[1].metric("Nominal interval", f"{metadata['nominal_interval_coverage'] * 100:.0f}%")
    cols[2].metric("API requests", int(telemetry["total_requests"]))
    cols[3].metric("p95 API latency", f"{telemetry['p95_latency_ms']:.1f} ms")
    st.subheader("Current prediction drivers")
    render_shap(prediction)
    st.subheader("Production warnings")
    for warning in metadata["warnings"]:
        st.write(f"- {warning}")


def render_about() -> None:
    st.header("ℹ️ About")
    st.markdown(
        """
This portfolio system combines **XGBoost regression**, SHAP explainability, hierarchical
conformal uncertainty, route/class/booking-horizon benchmarks, counterfactual simulation,
FastAPI serving, and a Streamlit decision-support interface over **300,153 historical flight
records**.

**Important:** this is historical/model-based decision support. It does not have live airline
inventory and does not guarantee future ticket-price movement.
"""
    )
    st.code(
        "Streamlit UI → FastAPI → XGBoost + SHAP + Phase 7 Intelligence Bundle",
        language="text",
    )


def main() -> None:
    """Render the complete Phase 9 dashboard."""
    client = api_client()
    render_header(client)
    scenario = scenario_from_sidebar()
    predict_clicked = st.sidebar.button("Predict Fare", type="primary", use_container_width=True)

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Explore",
        ["Dashboard", "Route Explorer", "Fare Trends", "Model Insights", "About"],
    )
    st.sidebar.caption("Phase 9 · Streamlit + FastAPI · Production hardening")

    existing = st.session_state.get("dashboard_bundle")
    existing_scenario = st.session_state.get("dashboard_scenario")
    if predict_clicked or existing is None or existing_scenario != scenario:
        bundle = load_dashboard(client, scenario)
    else:
        bundle = existing

    if page == "About":
        render_about()
        return
    if bundle is None:
        return

    if page == "Dashboard":
        render_dashboard(bundle, scenario)
    elif page == "Route Explorer":
        render_route_explorer(bundle, bundle["prediction"])
    elif page == "Fare Trends":
        render_fare_trends(bundle, scenario)
    elif page == "Model Insights":
        render_model_insights(client, bundle)


if __name__ == "__main__":
    main()
