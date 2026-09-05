# Phase 2 — Flight Market Intelligence, EDA & Leakage-Safe Splitting

## Objective

Turn the validated 300,153-row dataset into a reproducible market-intelligence layer and establish the definitive evaluation split used by every modeling phase that follows.

## Why this phase matters

The dataset is highly structured rather than i.i.d. in the simplistic sense. Cabin class dominates the fare scale, routes have materially different price distributions, and booking horizon has a strong nonlinear relationship with Economy fares. In addition, repeated production-input scenarios exist in the raw data.

A naive row-random split can place the exact same deployed feature vector in both training and evaluation. Phase 2 prevents that by grouping identical production-input scenarios before splitting.

## Definitive split design

- Train: 70%
- Validation: 15%
- Test: 15%
- Random seed: 42
- Grouping unit: exact deployed feature vector
- Stratification unit: directed route × cabin class × booking-horizon bucket
- Booking-horizon buckets: 1–7, 8–14, 15–21, 22–35, 36–49 days

The deployed feature vector is:

`airline, source_city, destination_city, departure_time, stops, class, duration, days_left`

All rows with an identical deployed feature vector remain in the same split.

## Canonical findings

- Dataset rows: 300,153
- Directed routes: 30
- Unique deployed-feature scenarios: 289,719
- Rows repeating an already-observed deployed feature vector: 10,434
- Repeated feature groups: 9,963
- Repeated feature groups with more than one observed fare: 6,599
- Median fare spread inside repeated scenarios: ₹537
- Maximum observed fare spread inside a repeated scenario: ₹60,309

A seeded conventional 70/15/15 row-random split would place **4,679 exact production scenarios across multiple splits**, affecting 9,674 rows. The definitive grouped split reduces this overlap to zero.

This is the principal reason the project uses scenario-grouped evaluation rather than a naive row-random split.

### Fare scale is strongly class-dependent

- Economy median fare: approximately ₹5.8K
- Business median fare: approximately ₹53.2K

Aggregate metrics will therefore always be supplemented with class-specific and segment-specific diagnostics in later phases.

### Booking horizon is especially important for Economy

Average Economy fares are substantially higher inside the final week before departure than 36–49 days out. Business fares also rise close to departure, but the relationship is less dramatic relative to their overall price scale.

### Route effects are material

The most and least expensive routes differ materially within each cabin class. Route/class-conditioned analysis is therefore part of the project’s intelligence layer rather than being treated as a generic categorical encoding problem.

## Generated outputs

Running Phase 2 writes:

- `reports/metrics/phase2_market_summary.json`
- `reports/metrics/phase2_split_summary.json`
- `data/processed/phase2_split_assignments.csv`
- six EDA figures under `reports/figures/`

Generated data, figures, and JSON metrics remain ignored by Git. The reproducible code and this phase record are committed.

## Definitive split result on the canonical dataset

- Train: 210,190 rows (70.03%) across 202,803 unique scenarios
- Validation: 44,954 rows (14.98%) across 43,458 unique scenarios
- Test: 45,009 rows (15.00%) across 43,458 unique scenarios
- Stratification cells: 300
- Minimum unique scenarios in any stratum: 209
- Exact scenario overlap across splits: **0**
- Median fare in all three splits: ₹7,425
- Mean fare: train ₹20,881.00; validation ₹20,900.67; test ₹20,919.09

The near-identical target distributions are a validation signal only; target values are not used as model inputs.

## Validation gate

Phase 2 is considered complete only when:

1. strict Phase 1 data validation passes;
2. the Phase 2 analytics pipeline completes;
3. split scenario overlap is exactly zero;
4. the automated test suite passes; and
5. Ruff reports no lint violations.
