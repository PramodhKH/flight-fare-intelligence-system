"""Production inference facade for fare prediction and decision intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .explainability import explain_raw_features
from .intelligence import (
    ComparableFareIndex,
    booking_guidance,
    booking_horizon_label,
    counterfactual_table,
    days_left_curve,
    fare_opportunity,
    predict_scenario,
    scenario_frame,
)
from .schema import MODEL_FEATURES
from .uncertainty import SegmentConformalCalibrator, comparative_reliability_score

MODEL_WARNING = (
    "Historical/model-based decision support only. The service does not have live airline "
    "inventory and cannot guarantee future fare movement."
)
RELIABILITY_WARNING = (
    "Reliability is comparative uncertainty, not a probability that the prediction is correct."
)
COUNTERFACTUAL_WARNING = (
    "Counterfactuals vary one model input while holding others fixed; they are not guaranteed "
    "future-price trajectories."
)


@dataclass
class FareIntelligenceEngine:
    """Load and serve the locked Phase 4 model plus Phase 7 intelligence artifacts."""

    model: Any
    calibrator: SegmentConformalCalibrator
    comparable_index: ComparableFareIndex
    reliability_reference: np.ndarray
    bundle_metadata: dict[str, Any]

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str | Path = "models/phase4_champion.joblib",
        intelligence_path: str | Path = "models/phase7_intelligence_bundle.joblib",
    ) -> FareIntelligenceEngine:
        """Load deployment artifacts from disk with contract checks."""
        model_path = Path(model_path)
        intelligence_path = Path(intelligence_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing champion model artifact: {model_path}")
        if not intelligence_path.exists():
            raise FileNotFoundError(f"Missing intelligence bundle: {intelligence_path}")

        model = joblib.load(model_path)
        bundle = joblib.load(intelligence_path)
        required = {
            "calibrator",
            "comparable_fare_index",
            "reliability_reference",
            "metadata",
        }
        missing = sorted(required - set(bundle))
        if missing:
            raise ValueError(f"Intelligence bundle is missing keys: {missing}")

        return cls(
            model=model,
            calibrator=bundle["calibrator"],
            comparable_index=bundle["comparable_fare_index"],
            reliability_reference=np.asarray(bundle["reliability_reference"], dtype=float),
            bundle_metadata=dict(bundle["metadata"]),
        )

    def metadata(self) -> dict[str, Any]:
        """Return deployment metadata safe for the public model endpoint."""
        return {
            "phase": 8,
            "champion": "xgboost",
            "model_features": list(MODEL_FEATURES),
            "nominal_interval_coverage": float(self.bundle_metadata.get("coverage", 0.90)),
            "uncertainty_calibration_rows": int(
                self.bundle_metadata.get("calibration_rows", len(self.reliability_reference))
            ),
            "test_set_scored": bool(self.bundle_metadata.get("test_set_scored", False)),
            "warnings": [MODEL_WARNING, RELIABILITY_WARNING, COUNTERFACTUAL_WARNING],
        }

    def predict(
        self,
        scenario: dict[str, Any],
        *,
        include_explanation: bool = True,
        include_guidance: bool = True,
        top_explanation_features: int = 5,
    ) -> dict[str, Any]:
        """Return point prediction, uncertainty, value context, and optional decision support."""
        features = scenario_frame(scenario)
        point = predict_scenario(self.model, scenario)
        interval = self.calibrator.interval_frame(features, np.array([point])).iloc[0]
        opportunity = fare_opportunity(point, scenario, self.comparable_index)

        relative_width = float(interval["interval_width"] / max(point, 1_000.0))
        reliability_score, reliability_label, reliability_percentile = (
            comparative_reliability_score(relative_width, self.reliability_reference)
        )

        result: dict[str, Any] = {
            "scenario": {feature: scenario[feature] for feature in MODEL_FEATURES},
            "predicted_fare": round(point, 2),
            "prediction_interval": {
                "lower": round(float(interval["prediction_lower"]), 2),
                "upper": round(float(interval["prediction_upper"]), 2),
                "width": round(float(interval["interval_width"]), 2),
                "calibration_level": str(interval["calibration_level"]),
                "calibration_rows": int(interval["calibration_rows"]),
            },
            "fare_opportunity": _round_mapping(opportunity),
            "reliability": {
                "score": reliability_score,
                "label": reliability_label,
                "relative_uncertainty_percentile": round(reliability_percentile, 4),
                "warning": RELIABILITY_WARNING,
            },
            "warning": MODEL_WARNING,
        }

        if include_guidance:
            curve = days_left_curve(self.model, scenario)
            guidance = booking_guidance(
                current_days_left=int(scenario["days_left"]),
                current_fare=point,
                opportunity_score=int(opportunity["fare_opportunity_score"]),
                curve=curve,
            )
            result["booking_guidance"] = _round_mapping(guidance)

        if include_explanation:
            explanation = explain_raw_features(self.model, features, approximate=True)
            shap_row = explanation.raw_values.iloc[0]
            ranked = sorted(
                MODEL_FEATURES,
                key=lambda feature: abs(float(shap_row[feature])),
                reverse=True,
            )[:top_explanation_features]
            result["explanation"] = {
                "method": explanation.method,
                "base_value": round(float(explanation.expected_value), 2),
                "reconstruction_error": round(float(explanation.max_reconstruction_error), 6),
                "top_drivers": [
                    {
                        "feature": feature,
                        "feature_value": str(scenario[feature]),
                        "contribution_inr": round(float(shap_row[feature]), 2),
                        "direction": (
                            "increases_fare"
                            if float(shap_row[feature]) >= 0.0
                            else "decreases_fare"
                        ),
                    }
                    for feature in ranked
                ],
            }
        return result

    def batch_predict(
        self,
        scenarios: list[dict[str, Any]],
        *,
        include_explanations: bool = False,
    ) -> list[dict[str, Any]]:
        """Run bounded batch inference without expensive guidance curves."""
        return [
            self.predict(
                scenario,
                include_explanation=include_explanations,
                include_guidance=False,
            )
            for scenario in scenarios
        ]

    def what_if(
        self,
        scenario: dict[str, Any],
        *,
        feature: str,
        values: list[Any],
    ) -> dict[str, Any]:
        """Compare one-feature-at-a-time counterfactual fare estimates."""
        base_fare = predict_scenario(self.model, scenario)
        table = counterfactual_table(self.model, scenario, feature=feature, values=values)
        table["difference_from_base"] = table["predicted_fare"] - base_fare
        table["difference_from_base_percent"] = (
            table["difference_from_base"] / max(base_fare, 1_000.0) * 100.0
        )
        rows = []
        for record in table.to_dict(orient="records"):
            rows.append(_round_mapping(record))
        return {
            "base_predicted_fare": round(base_fare, 2),
            "changed_feature": feature,
            "scenarios": rows,
            "warning": COUNTERFACTUAL_WARNING,
        }

    def routes(self) -> list[dict[str, str]]:
        """List directed routes represented by the historical benchmark index."""
        route_names = sorted({key[0] for key in self.comparable_index.values})
        rows = []
        for route in route_names:
            source, destination = route.split(">", maxsplit=1)
            rows.append({"route": route, "source_city": source, "destination_city": destination})
        return rows

    def route_analytics(
        self,
        *,
        source_city: str,
        destination_city: str,
        cabin_class: str,
    ) -> dict[str, Any]:
        """Return training-only historical fare summaries across booking horizons."""
        route = f"{source_city}>{destination_city}"
        horizon_order = ["1-7", "8-14", "15-21", "22-35", "36-49"]
        horizons: list[dict[str, Any]] = []
        for horizon in horizon_order:
            fares = self.comparable_index.values.get((route, cabin_class, horizon))
            if fares is None or len(fares) == 0:
                continue
            horizons.append(
                {
                    "booking_horizon": horizon,
                    "rows": len(fares),
                    "q10": round(float(np.quantile(fares, 0.10)), 2),
                    "median": round(float(np.median(fares)), 2),
                    "mean": round(float(np.mean(fares)), 2),
                    "q90": round(float(np.quantile(fares, 0.90)), 2),
                }
            )
        if not horizons:
            raise KeyError(f"No route analytics available for {route} / {cabin_class}")
        return {
            "route": route,
            "class": cabin_class,
            "horizons": horizons,
            "source": "training-only historical comparable fares",
        }

    def booking_horizon_curve(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Return model counterfactual and historical medians for all 1..49 days."""
        model_curve = days_left_curve(self.model, scenario)
        route = f"{scenario['source_city']}>{scenario['destination_city']}"
        cabin_class = str(scenario["class"])
        historical_medians: dict[str, float] = {}
        for horizon in ["1-7", "8-14", "15-21", "22-35", "36-49"]:
            fares = self.comparable_index.values.get((route, cabin_class, horizon))
            if fares is not None and len(fares) > 0:
                historical_medians[horizon] = float(np.median(fares))

        rows = []
        for record in model_curve.to_dict(orient="records"):
            days = int(record["days_left"])
            horizon = booking_horizon_label(days)
            rows.append(
                {
                    "days_left": days,
                    "predicted_fare": round(float(record["predicted_fare"]), 2),
                    "historical_horizon_median": round(historical_medians[horizon], 2),
                }
            )
        return {
            "route": route,
            "class": cabin_class,
            "curve": rows,
            "warning": COUNTERFACTUAL_WARNING,
        }


def _round_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Round floating values for stable API serialization while preserving other types."""
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (float, np.floating)):
            result[key] = round(float(value), 4)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        else:
            result[key] = value
    return result
