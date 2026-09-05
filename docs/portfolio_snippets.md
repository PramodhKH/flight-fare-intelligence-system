# Portfolio Snippets

## GitHub repository description

Explainable airfare decision-intelligence platform built on 300K+ flight records with XGBoost, conformal uncertainty, SHAP, route analytics, FastAPI, and Streamlit.

## Resume — compact two-bullet version

- Built an end-to-end **Flight Fare Intelligence System** over **300K+ flight records**, benchmarking Linear Regression, Random Forest, XGBoost, and CatBoost with leakage-safe grouped splits; selected XGBoost at **₹2.67K held-out RMSE / 0.986 R²** while reducing RMSE ~60% versus the linear baseline.
- Engineered a production decision-support stack with **90.28% empirical coverage for nominal 90% conformal intervals**, SHAP explanations, Fare Opportunity and reliability scoring, counterfactual booking guidance, **FastAPI + Streamlit**, Docker, CI, telemetry, and a **72-test** automated suite.

## Resume — single-bullet version

- Developed an explainable airfare decision-intelligence platform on **300K+ records**, achieving **₹2.67K held-out RMSE / 0.986 R²** with XGBoost and productionizing conformal uncertainty, SHAP, contextual fare scoring, counterfactual guidance, FastAPI, Streamlit, Docker, CI, and 72 automated tests.

## LinkedIn project description

Built a production-oriented Flight Fare Intelligence System from 300,153 historical flight records. The project uses a leakage-safe scenario-grouped evaluation design and benchmarks Linear Regression, Random Forest, XGBoost, and CatBoost before selecting XGBoost. The final untouched 45,009-row test set achieved ₹2,670 RMSE, 0.9862 R², and 10.38% MAPE.

Beyond point prediction, the system adds SHAP explanations, 90% conformal prediction intervals, Fare Opportunity and reliability scores, route/class/booking-horizon analytics, and model-based Buy Now / Monitor / Wait guidance. The complete intelligence layer is exposed through FastAPI and a Streamlit dashboard, with Docker, CI, request telemetry, and automated regression/integration testing.

## Interview talking points

1. **Leakage:** identical production scenarios were kept in one split; a normal row-random split would have leaked thousands of repeated scenarios across train/evaluation boundaries.
2. **Model selection:** XGBoost won on RMSE and production tradeoff, while Random Forest had slightly better MAE/MAPE but was much larger and slower.
3. **Reliability:** high aggregate R² did not hide segment failures; Business, last-minute, and ₹80K+ fares were explicitly stress-tested.
4. **Uncertainty:** conformal intervals were calibrated without touching the test set and later achieved ~90.3% coverage on the untouched final holdout.
5. **Explainability vs certainty:** SHAP explains the model estimate; it does not measure confidence, so explanation and uncertainty are separate layers.
6. **Product thinking:** prediction is contextualized with historical comparables and counterfactual decisions instead of being presented as an isolated number.
