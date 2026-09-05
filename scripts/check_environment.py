#!/usr/bin/env python
"""Verify the project ML/explainability stack resolves in the active interpreter."""

from __future__ import annotations

import json
import sys

import catboost
import numpy
import pandas
import shap
import sklearn
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
        },
        "ide_note": "VS Code should use <project>/.venv/bin/python on macOS/Linux.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
