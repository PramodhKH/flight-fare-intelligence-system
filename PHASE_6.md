# Phase 6 — Explainable Fare Engine with SHAP

## Objective

Explain the locked Phase 4/5 **XGBoost champion** without changing its parameters and without scoring the sealed test split. Phase 6 converts the model from a high-performing black-box regressor into an additive fare-explanation engine suitable for later API and dashboard use.

The central questions are:

- Which deployed features drive fare predictions globally?
- Do the dominant drivers change for Economy, Business, last-minute, and high-fare cases?
- How does booking horizon affect the model's predicted fare contribution?
- Can an individual prediction be decomposed into understandable INR contributions while preserving model additivity?
- Do Phase 5 failure segments have identifiable model drivers that can inform Phase 7 uncertainty and reliability design?

## Explainability policy

Phase 6 remains diagnostic:

- the XGBoost champion is loaded from `models/phase4_champion.joblib`;
- no hyperparameters are changed;
- validation is the only explained evaluation split;
- all 45,009 test rows remain sealed;
- SHAP is aggregated back to the eight deployed raw features;
- global/segment explanations use XGBoost approximate TreeSHAP across the **full validation split** for scalable coverage;
- deterministic representative local cases use exact TreeSHAP;
- additivity is checked numerically before outputs are accepted.

The XGBoost model consumes a dense one-hot representation internally. Phase 6 therefore sums the SHAP values of all transformed columns belonging to the same deployed input:

```text
one-hot airline columns      → airline
one-hot source columns       → source_city
one-hot destination columns  → destination_city
one-hot departure columns    → departure_time
one-hot stops columns        → stops
one-hot class columns        → class
duration                     → duration
days_left                    → days_left
```

This preserves the additive model decomposition while producing explanations that match the user-facing feature contract.

## Global explanation design

The entire 44,954-row validation split is explained. Global importance is ranked by:

```text
mean(abs(SHAP contribution))
```

in INR. This measures the typical magnitude by which each raw feature moves a prediction away from the model's baseline value.

A reference run found the following ordering:

| Feature | Mean absolute SHAP | Share of total raw-feature importance |
| --- | ---: | ---: |
| class | ~₹19.7k | ~69.1% |
| duration | ~₹2.13k | ~7.5% |
| days_left | ~₹1.81k | ~6.4% |
| airline | ~₹1.77k | ~6.2% |
| source_city | ~₹1.20k | ~4.2% |

The exact local values generated on the user's environment are authoritative. The reference result nevertheless confirms that cabin class dominates the aggregate model because Economy and Business occupy very different fare regimes.

This is why Phase 6 does not stop at one global ranking. Segment explanations are required to avoid allowing the class effect to hide the model's behavior inside each fare regime.

## Segment-level explanations

Raw-feature SHAP importance is recomputed inside five predefined segments:

```text
overall validation
Economy
Business
1–7 days before departure
₹80k+ observed fares
```

Reference diagnostics show distinct driver patterns:

### Economy

`class` remains the largest baseline separator, while **days_left** becomes the most important secondary feature. This is consistent with the strong booking-horizon behavior established in Phase 2 and the elevated last-minute errors measured in Phase 5.

### Business

The largest secondary drivers become **duration** and **airline**, indicating that Business pricing variation is driven more strongly by itinerary/market structure after the cabin-class premium is established.

### Last-minute bookings

`days_left` becomes a much larger positive contributor. In the reference run, 1–7 day cases had a mean absolute `days_left` SHAP contribution of roughly ₹4.8k, materially above its global importance.

### High-fare tail

For ₹80k+ cases, the strongest reference drivers after Business class were **duration** and **source city**, with route/airline context also contributing materially. This provides an explanation layer for the Phase 5 finding that high-value Business fares are the champion's least reliable region.

Importantly, SHAP explains what the model is using; it does **not** prove causal effects in airline pricing.

## Booking-horizon effect

The generated `phase6_days_left_shap_dependence.png` plot shows the model's contribution from `days_left` across the validation split.

A reference run showed a clear directional pattern:

```text
1 day left   → strongly positive fare contribution
2–7 days     → positive contribution
~3 weeks     → contribution turns negative on average
30–49 days   → generally negative contribution
```

For example, the mean reference SHAP contribution for one day remaining was approximately +₹8.5k, while 49 days remaining was approximately -₹1.7k. These are model explanations, not guaranteed future price changes.

This finding directly supports Phase 7's booking-horizon intelligence and counterfactual what-if simulator.

## Duration effect

Duration is the second-largest global SHAP driver in the reference run. Its relationship is nonlinear rather than simply "longer flight = more expensive." Very short itineraries carry a strong negative contribution relative to the global model baseline, medium-duration itineraries tend to increase predicted fare, and the contribution changes again in the longest-duration tail.

That nonlinear pattern helps explain why Linear Regression performed poorly inside the class-specific fare regimes in Phase 3.

## Local explanations

Phase 6 chooses four deterministic cases rather than manually selecting attractive examples:

```text
representative Economy
representative Business
representative 1–7 day booking
representative ₹80k+ fare
```

Within each segment, the case whose absolute validation error is closest to the segment median is selected, with record ID used as a deterministic tie-breaker.

Exact TreeSHAP then decomposes the prediction:

```text
model baseline
+ class contribution
+ airline contribution
+ route-component contributions
+ departure-period contribution
+ stops contribution
+ duration contribution
+ days-left contribution
= predicted fare
```

A reference high-fare case demonstrates the intended output style:

```text
Actual fare:      ~₹105.8k
Predicted fare:   ~₹96.4k

Business class:   strong positive contribution
2 days left:      strong positive contribution
Mumbai origin:    positive contribution
Vistara:          positive contribution
Bangalore dest.:  positive contribution
long duration:    positive contribution
```

The model still underpredicts that case, illustrating an important distinction: a prediction can be explainable while still being uncertain or biased. Phase 7 will address that reliability gap with uncertainty-aware outputs rather than treating SHAP as confidence.

## Additivity checks

Phase 6 rejects explanations if the SHAP reconstruction differs from the model prediction by more than ₹1.

The reference execution produced sub-₹1 reconstruction differences for both:

- approximate full-validation TreeSHAP; and
- exact representative local TreeSHAP.

This ensures that the raw-feature aggregation remains faithful to the XGBoost prediction.

## Generated outputs

Running `make phase6` creates:

```text
reports/metrics/phase6_explainability_summary.json

reports/predictions/phase6_global_feature_importance.csv
reports/predictions/phase6_segment_feature_importance.csv
reports/predictions/phase6_categorical_effects.csv
reports/predictions/phase6_validation_shap.csv
reports/predictions/phase6_numeric_dependence.csv
reports/predictions/phase6_local_explanations.csv

reports/figures/phase6_global_shap_importance.png
reports/figures/phase6_segment_driver_heatmap.png
reports/figures/phase6_days_left_shap_dependence.png
reports/figures/phase6_duration_shap_dependence.png
reports/figures/phase6_representative_economy_shap.png
reports/figures/phase6_representative_business_shap.png
reports/figures/phase6_representative_last_minute_shap.png
reports/figures/phase6_representative_high_fare_shap.png
```

All generated outputs remain local and ignored by Git. The explanation code, tests, methodology, and configuration are version-controlled.

## Prerequisite

Phase 6 intentionally explains the already-locked champion. It expects the local Phase 4 model and Phase 5 reliability report to exist:

```bash
make phase5
```

Then run:

```bash
make phase6
```

## Completion gate

Phase 6 is complete only when:

1. the active project environment resolves NumPy, Pandas, scikit-learn, XGBoost, CatBoost, and SHAP;
2. strict raw-data validation passes;
3. the Phase 5 report confirms XGBoost remains champion and the test set is sealed;
4. all 44,954 validation rows receive raw-feature SHAP explanations;
5. approximate full-validation additivity error remains below ₹1;
6. global and reliability-segment feature importance outputs are generated;
7. booking-horizon and duration dependence diagnostics are generated;
8. four deterministic representative cases receive exact local explanations;
9. exact local additivity error remains below ₹1;
10. the test split remains unscored;
11. all automated tests pass; and
12. Ruff reports no lint violations.
