# Phase 9 final UI fix

This patch only updates `frontend/app.py`.

Changes:
- Moves `API Connected` and the XGBoost/conformal status pill into normal page flow under the subtitle so Streamlit's top toolbar cannot clip them.
- Keeps the status row left-aligned and responsive.
- Moves negative SHAP contribution labels inside their bars and increases chart margins/height to avoid overlap with feature labels.
- No model, API, uncertainty, SHAP values, recommendation logic, or data-split behavior is changed.
