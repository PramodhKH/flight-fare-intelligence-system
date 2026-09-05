#!/usr/bin/env python
"""Verify the project ML, explainability, and API stack in the active interpreter."""

from __future__ import annotations

import json
import sys

import catboost
import fastapi
import httpx
import numpy
import pandas
import plotly
import pydantic
import shap
import sklearn
import streamlit
import uvicorn
import xgboost


def main() -> None:
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
            "shap": shap.__version__,
            "fastapi": fastapi.__version__,
            "pydantic": pydantic.__version__,
            "httpx": httpx.__version__,
            "uvicorn": uvicorn.__version__,
            "streamlit": streamlit.__version__,
            "plotly": plotly.__version__,
        },
        "ide_note": "VS Code should use <project>/.venv/bin/python on macOS/Linux.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
