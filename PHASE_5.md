# Phase 5 — Model Reliability, Segment Error Analysis & Robustness

## Objective

Stress-test the locked Phase 4 **XGBoost champion** without retuning it and without scoring the sealed test split. Phase 5 asks a different question from Phase 4:

> Where is the model reliable, where does it fail, and are those failures driven by fare level, cabin class, booking horizon, route, itinerary structure, or sparse market context?

The Phase 4 validation set remains the only scored evaluation set. The 45,009-row test split stays sealed for later final champion evaluation.

## Reliability policy

Phase 5 is deliberately diagnostic rather than optimization-driven:

- no XGBoost hyperparameters are changed;
- no new model family is introduced;
- no decision is informed by the test split;
- the runner-up Random Forest is used only as a **locked robustness comparator**;
- RMSE remains the primary reliability metric, supported by MAE, R², MAPE, bias, median absolute error, P90 absolute error, and underprediction rate.

This prevents repeated validation-set tuning while still challenging the Phase 4 champion.

## Segment dimensions

The champion is evaluated across:

- cabin class;
- airline;
- directed route;
- booking horizon;
- stop count;
- departure period;
- actual-fare band; and
- training market-context support.

Fare bands are:

```text
< ₹10k
₹10k–20k
₹20k–40k
₹40k–60k
₹60k–80k
₹80k+
```

Booking-horizon buckets remain identical to Phase 2:

```text
1–7
8–14
15–21
22–35
36–49 days
```

## Market-context support

Exact deployed-feature vectors cannot appear across train/validation/test because Phase 2 groups them into one split. For robustness analysis, Phase 5 therefore measures a broader categorical market context:

```text
airline
+ source city
+ destination city
+ class
+ departure period
+ stops
```

For each validation row, the system counts how many training rows share that context and labels support as:

```text
unseen
low:      1–25 rows
medium:   26–100 rows
high:     101+ rows
```

This provides a defensible sparsity diagnostic without violating the scenario-grouped split.

## Reference findings from the locked Phase 4 validation predictions

The user's Phase 4 XGBoost run produced **₹2,677.55 RMSE, ₹1,401.91 MAE, 0.9861 R², and 10.39% MAPE** on 44,954 validation rows.

### Cabin-class reliability

| Class | Rows | RMSE | MAE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Economy | 30,937 | ₹1,475.78 | ₹836.24 | 0.8465 | 12.84% |
| Business | 14,017 | ₹4,264.46 | ₹2,650.40 | 0.8914 | 4.97% |

Business fares have materially larger absolute errors, while Economy has the higher percentage error because its fares are lower.

### Booking-horizon reliability

| Days before departure | Rows | RMSE | MAE |
| --- | ---: | ---: | ---: |
| 1–7 | 4,814 | **₹4,433.99** | ₹2,705.82 |
| 8–14 | 6,408 | ₹3,040.06 | ₹1,737.01 |
| 15–21 | 6,761 | ₹2,586.24 | ₹1,394.55 |
| 22–35 | 13,591 | **₹2,074.90** | ₹1,095.27 |
| 36–49 | 13,380 | ₹2,198.74 | **₹1,087.48** |

Last-minute bookings are substantially harder to estimate than flights booked several weeks in advance. This is directly relevant to the later booking-horizon intelligence and uncertainty layers.

### High-fare tail failure

The most important Phase 5 finding is systematic underprediction in the extreme fare tail.

For validation fares at or above **₹80,000**:

```text
rows:                  315
RMSE:                  ₹12,966
MAE:                   ₹10,578
mean bias:            -₹10,170
underprediction rate:  92.1%
```

For the 14 validation fares at or above **₹100,000**:

```text
RMSE:                  ₹20,883
MAE:                   ₹19,722
mean bias:            -₹19,722
underprediction rate: 100.0%
```

The negative mean bias means the model predicts below the observed fare on average. The very-high-fare segment is small, so its exact metrics must not be overgeneralized, but the directional failure is strong enough to preserve as a known limitation.

### Sparse-context robustness

The broad market-context analysis does **not** support the hypothesis that sparse categorical support is the main source of error:

```text
low support (1–25 training rows):       RMSE ≈ ₹1,714
medium support (26–100 rows):           RMSE ≈ ₹1,975
high support (101+ rows):               RMSE ≈ ₹2,859
unseen contexts:                        4 validation rows only
```

The higher error in high-support contexts is largely consistent with those contexts containing more complex and higher-value fare regimes. Support count therefore cannot be interpreted as a standalone confidence score. Phase 7 will combine uncertainty and contextual evidence more carefully.

## Champion vs runner-up robustness challenge

Random Forest remains the strongest Phase 4 alternative, so Phase 5 compares the two **without retuning either model**.

Reference diagnostics show XGBoost retains the lower RMSE:

| Stress segment | XGBoost RMSE | Random Forest RMSE |
| --- | ---: | ---: |
| Overall validation | **₹2,677.55** | ₹2,815.82 |
| Economy | **₹1,475.78** | ₹1,519.83 |
| Business | **₹4,264.46** | ₹4,508.93 |
| 1–7 days left | **₹4,433.99** | ₹4,783.13 |
| ₹80k+ fare | **₹12,966.26** | ₹13,990.01 |
| ₹100k+ fare | **₹20,882.64** | ₹20,974.25 |

Random Forest still has slightly better overall MAE/MAPE, as established in Phase 4, but XGBoost remains the stronger choice under the project's RMSE-first reliability policy and also retains the better deployment-size/latency tradeoff from Phase 4.

## Interpretation

Phase 5 does **not** indicate that the champion should be replaced. Instead it establishes three concrete constraints for later system design:

1. **Fare magnitude matters.** Absolute error rises sharply for high-value Business fares and the ₹80k+ tail.
2. **Booking horizon matters.** Last-minute flights are materially harder to predict.
3. **Raw support count is insufficient as confidence.** Sparse market contexts are not automatically less accurate, so the future reliability score must use uncertainty and error behavior rather than record count alone.

The high-fare underprediction pattern becomes a priority case for Phase 6 SHAP explanations and Phase 7 uncertainty/reliability design.

## Generated outputs

Running `make phase5` creates:

```text
reports/metrics/phase5_reliability_summary.json

reports/predictions/phase5_segment_metrics.csv
reports/predictions/phase5_stress_tests.csv
reports/predictions/phase5_champion_vs_runner_up.csv
reports/predictions/phase5_validation_diagnostics.csv
reports/predictions/phase5_worst_errors.csv

reports/figures/phase5_rmse_by_fare_band.png
reports/figures/phase5_bias_by_fare_band.png
reports/figures/phase5_rmse_by_class.png
reports/figures/phase5_rmse_by_booking_horizon.png
reports/figures/phase5_rmse_by_training_support.png
reports/figures/phase5_route_rmse_heatmap.png
```

All generated artifacts remain local and ignored by Git. The code, tests, configuration, and methodology are version-controlled.

## Prerequisite

Phase 5 intentionally uses the already-locked Phase 4 artifacts instead of silently retraining the champion. Run Phase 4 first if those local artifacts do not exist:

```bash
make phase4
```

Then run:

```bash
make phase5
```

## Completion gate

Phase 5 is complete only when:

1. the project environment/import check passes;
2. strict raw-data validation passes;
3. the Phase 4 XGBoost champion artifact and benchmark report exist;
4. Phase 4 confirms the test set was not scored;
5. the complete validation reliability frame is generated;
6. all defined segment and stress diagnostics are generated;
7. XGBoost is compared with the locked Random Forest runner-up without retuning;
8. the test set remains sealed;
9. all automated tests pass; and
10. Ruff reports no lint violations.
