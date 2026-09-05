# Phase 1 - Foundation and Data Contract

## Decisions locked

- Project name: **Flight Fare Intelligence System**
- Task: supervised tabular regression
- Canonical dataset: 300,153 rows
- Deployed features: airline, source city, destination city, departure period, stops, class, duration, days left
- Target: price (INR)
- Analysis-only fields: flight number, arrival period
- Ignored source export field: `Unnamed: 0`
- Planned models: Linear Regression -> Random Forest -> XGBoost -> CatBoost
- Planned deployment: saved sklearn-compatible pipeline/artifact -> FastAPI -> Streamlit

## Why `flight` is not a deployed feature

The raw data contains 1,561 flight codes. Flight identifiers can strongly memorize schedule-specific patterns and would force the UI/API to require a high-cardinality operational code. The deployment contract instead focuses on attributes a traveler can intentionally vary in a what-if scenario. Flight number remains available for EDA and diagnostic experiments.

## Why `arrival_time` is not a deployed feature

The intended what-if system is centered on departure choice, route, stops, duration, cabin class, and booking horizon. Keeping arrival period out of the required model contract makes inference simpler and prevents the UI from demanding an additional schedule field not present in the stated product experience. It remains available for exploratory analysis.

## Phase 1 exit criteria

- package-style repository exists
- original reference PDF preserved under `legacy/`
- raw CSV is local but Git-ignored
- strict validator passes the canonical 300,153-row dataset
- deployment feature contract is explicitly tested
- unit tests and linting pass
- CI scaffold covers Python 3.11 and 3.12
