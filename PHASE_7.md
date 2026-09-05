# Phase 7 — Fare Intelligence, Uncertainty & Decision Engine

## Objective

Phase 7 converts the locked XGBoost fare estimator into a **decision-intelligence layer**. The model is still not retrained, and the sealed Phase 2 test set remains untouched. The new layer answers four questions that a point prediction alone cannot:

1. **What fare range is plausible?**
2. **Is the estimated fare cheap or expensive relative to comparable historical flights?**
3. **How reliable is this estimate compared with other calibrated predictions?**
4. **How does the model's expected fare change under booking-horizon and other what-if scenarios?**

The resulting outputs are designed for the Phase 8 FastAPI service and Phase 9 Streamlit application.

## Evaluation discipline

The Phase 2 split remains authoritative:

```text
Train       210,190
Validation   44,954
Test         45,009  ← still sealed
```

Phase 7 does **not** consume the test split. Instead, the existing validation split is deterministically partitioned at the `scenario_id` level into two new roles:

```text
Validation calibration subset  ~50%
Validation evaluation subset   ~50%
```

All rows sharing the same Phase 2 production scenario stay on the same side of this Phase 7 partition. The calibration half constructs uncertainty rules; the evaluation half measures empirical interval coverage. This partition is only for uncertainty engineering and does not alter the original train/validation/test assignments.

## 1. Tail-aware asymmetric conformal intervals

A single symmetric `prediction ± RMSE` interval would be inappropriate because Phase 5 showed that error varies substantially by cabin class, booking horizon, and fare level. The extreme high-fare tail is also strongly asymmetric: expensive fares are usually **underpredicted**, not equally over- and underpredicted.

Phase 7 therefore uses a hierarchical residual-calibration system. For each prediction, it attempts to calibrate residual bounds using the most specific supported context:

```text
class + booking horizon + predicted fare band
                 ↓ fallback
class + predicted fare band
                 ↓ fallback
class + booking horizon
                 ↓ fallback
class
                 ↓ fallback
global residual distribution
```

The normal target is **90% empirical coverage**. Signed residual quantiles are used rather than an absolute-error radius, which permits asymmetric lower and upper bounds.

### High-fare Business guardrail

Phase 5 established a pre-existing reliability risk: fares above ₹80k were underpredicted more than 90% of the time. Phase 7 therefore adds a deliberately conservative upper-tail guardrail when the model predicts a **Business** fare in the `60–80k` or `80k+` band.

For those risk segments:

- the upper residual bound uses the **99.5th calibration percentile**;
- a smaller minimum calibration sample of 50 is permitted;
- the lower bound remains aligned with the standard interval policy.

This is not hidden model retuning. It is an explicit uncertainty policy motivated by the already-documented Phase 5 failure mode. The consequence is intentionally wider intervals when the model enters a region where it previously showed systematic underprediction.

A reference execution produced:

```text
Nominal coverage:              90.0%
Evaluation coverage:           ~90.5%
Median interval width:         ~₹3.7k
Mean interval width:           ~₹7.5k
High-fare ₹80k+ coverage:      ~91.0%
Last-minute 1–7 day coverage:  ~90.9%
```

Exact values generated in the user's environment are authoritative.

## 2. Fare Opportunity Score

The Fare Opportunity Score answers a different question from the prediction interval:

> Where does this estimated fare sit relative to historically comparable flights?

Comparable flights are defined using only the training split:

```text
directed route + cabin class + booking horizon
```

For example:

```text
Delhi → Mumbai
Economy
8–14 days before departure
```

The model-estimated fare is ranked against the empirical historical fare distribution for that comparison group.

```text
Fare Opportunity Score = 100 − historical fare percentile
```

A lower-priced fare therefore receives a higher opportunity score.

The user-facing bands are:

| Score | Position |
| ---: | --- |
| 80–100 | EXCELLENT VALUE |
| 60–79 | GOOD VALUE |
| 40–59 | TYPICAL |
| 20–39 | ABOVE TYPICAL |
| 0–19 | EXPENSIVE |

The engine also returns the comparison-group median, number of historical comparables, percentile, absolute difference from the median, and percentage difference from the median.

This score is contextual, not a claim that the model has found a live airline discount.

## 3. Comparative reliability score

Phase 5 demonstrated that raw training-context counts are **not** a valid standalone confidence measure. Phase 7 therefore does not equate “more rows” with “more confidence.”

Instead, reliability is derived from the calibrated prediction interval:

```text
relative uncertainty = interval width / predicted fare
```

That relative width is ranked against the same quantity across the calibration distribution. Narrower-than-usual intervals receive higher reliability scores.

```text
HIGH    score ≥ 70
MEDIUM  score 40–69
LOW     score < 40
```

The score is intentionally described as **comparative uncertainty**, not “probability the prediction is correct.” This distinction must be preserved in the API, dashboard, README, and resume narrative.

## 4. Booking-horizon intelligence

Phase 6 showed that `days_left` becomes especially influential near departure. Phase 7 operationalizes that insight with a counterfactual curve.

For one flight scenario, all inputs are held constant while `days_left` is varied from 1 through 49:

```text
same route
same airline
same cabin class
same stops
same departure period
same duration
only days_left changes
```

This produces a model-based booking-horizon curve that can later drive the Streamlit slider and `/what-if` API endpoint.

The project is explicit that this is **counterfactual model behavior learned from historical cross-sectional data**, not a longitudinal forecast of a specific flight's future listed price.

## 5. Buy Now / Monitor / Wait-or-Monitor guidance

The decision engine compares the current model estimate with the model's predicted fares over the next seven days closer to departure.

The policy uses three states:

```text
BUY_NOW
MONITOR
WAIT_OR_MONITOR
```

A change is considered material when the median model-predicted fare over the wait window differs from the current estimate by at least **5%**.

The Fare Opportunity Score is used as a second signal so that the engine does not automatically recommend buying an already expensive fare merely because the model expects it to become even more expensive.

Examples of policy logic:

```text
Expected fare rises materially
+ current fare is not above-typical
→ BUY_NOW

Expected fare falls materially
+ current fare is not already excellent value
→ WAIT_OR_MONITOR

Signals conflict or change is small
→ MONITOR
```

Every recommendation carries the following interpretation:

> Historical/model-based decision support only. It does not use live airline inventory and does not guarantee future price movement.

## 6. General what-if simulation

The Phase 7 utility layer supports one-feature-at-a-time counterfactuals across deployed inputs. The reference demo exports comparisons for:

- airline;
- departure period;
- stops; and
- all 1–49 `days_left` values.

Candidate airline/departure/stops values used in the generated demo are restricted to values historically observed for the same route and class, reducing obviously unsupported what-if combinations.

## 7. Route and booking-horizon intelligence artifacts

Two reusable historical analytics tables are generated for Phase 8/9:

### Route benchmarks

For every directed route × class × booking-horizon group:

```text
rows
q10
q25
median
mean
q75
q90
```

### Booking-horizon intelligence

For every route × class × exact `days_left` value:

```text
rows
historical median fare
historical mean fare
```

These artifacts allow the eventual application to distinguish the ML estimate from the historical market context surrounding that estimate.

## Reference demo

The deterministic reference execution selected a real validation Economy scenario close to the portfolio example:

```text
Vistara
Delhi → Mumbai
Morning
1 stop
Economy
~12–13 days before departure
```

A reference run produced an estimated fare of roughly ₹11.7k with a calibrated interval of about ₹9.3k–₹15.4k. Its contextual fare score placed it above the median for comparable route/class/horizon flights, while the booking-horizon model indicated upward pressure closer to departure. Because those signals conflicted, the decision policy returned `MONITOR` rather than forcing a Buy/Wait answer.

That behavior is intentional: ambiguity should remain visible instead of being converted into false certainty.

## Generated outputs

Running `make phase7` creates:

```text
models/phase7_intelligence_bundle.joblib

reports/metrics/phase7_intelligence_summary.json

reports/predictions/phase7_uncertainty_evaluation.csv
reports/predictions/phase7_interval_coverage_by_segment.csv
reports/predictions/phase7_route_benchmarks.csv
reports/predictions/phase7_booking_horizon_intelligence.csv
reports/predictions/phase7_demo_days_left_curve.csv
reports/predictions/phase7_demo_counterfactuals.csv

reports/figures/phase7_interval_coverage_by_class.png
reports/figures/phase7_interval_width_by_booking_horizon.png
reports/figures/phase7_fare_opportunity_score_distribution.png
reports/figures/phase7_reliability_score_distribution.png
reports/figures/phase7_demo_booking_horizon_curve.png
```

Generated data, figures, reports, and model artifacts remain ignored by Git. The code, tests, configuration, methodology, and documentation are version-controlled.

## Intelligence bundle

`models/phase7_intelligence_bundle.joblib` contains the reusable objects needed by Phase 8:

```text
uncertainty calibrator
historical comparable-fare index
reliability reference distribution
calibration metadata
```

The XGBoost prediction model itself remains the locked Phase 4 champion artifact.

## Completion gate

Phase 7 is complete only when:

1. the Phase 6 report confirms XGBoost is still the champion and the test set is sealed;
2. the 44,954 validation rows are scenario-preservingly partitioned into calibration and uncertainty-evaluation subsets;
3. no scenario appears on both Phase 7 validation subsets;
4. the uncertainty calibrator is fit without touching the test split;
5. overall empirical interval coverage is close to the 90% target;
6. the Phase 5 high-fare failure region receives explicit conservative uncertainty treatment;
7. fare-opportunity scores use training-only comparable fare distributions;
8. reliability scores are interval-based rather than support-count-based;
9. route and booking-horizon intelligence artifacts are generated;
10. one-feature-at-a-time counterfactual simulation is reproducible;
11. Buy/Monitor/Wait-or-Monitor guidance includes an explicit non-guarantee disclaimer;
12. all automated tests pass; and
13. Ruff reports no lint violations.

## Next phase

Phase 8 will expose the Phase 4 prediction model, Phase 6 explanation engine, and Phase 7 intelligence bundle through a validated **FastAPI** service supporting single prediction, batch inference, route analytics, what-if simulation, model metadata, and health endpoints.

## Automatic code-quality correction

Phase gates now run Ruff's safe auto-fixes and formatter before the final lint and test gates.
This automatically corrects safe issues such as import sorting, unused-import cleanup when Ruff
marks it safe, and formatting. Ruff fixes marked unsafe are intentionally not applied
automatically; the final lint check will fail instead of silently changing program semantics.
Run `make autofix` at any time (`make lint-fix` remains an alias). Tests run after formatting so
they validate the corrected source.
