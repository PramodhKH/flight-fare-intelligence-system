PYTHON ?= python
DATA ?= data/raw/Flight_Booking.csv

.PHONY: install envcheck validate test format autofix lint-fix lint quality phase1 phase2 phase3 phase4 phase5 phase6 phase7

install:
	$(PYTHON) -m pip install -e ".[dev]"

envcheck:
	$(PYTHON) scripts/check_environment.py

validate:
	$(PYTHON) scripts/validate_data.py $(DATA) --strict

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m ruff format src scripts tests

autofix:
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests

# Backward-compatible alias.
lint-fix: autofix

lint:
	$(PYTHON) -m ruff check src scripts tests

quality: autofix lint test

phase1: validate quality

phase2: validate
	$(PYTHON) scripts/run_phase2.py $(DATA) --seed 42
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest

phase3: validate
	$(PYTHON) scripts/run_phase2.py $(DATA) --seed 42
	$(PYTHON) scripts/run_phase3.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest

phase4: envcheck validate
	$(PYTHON) scripts/run_phase2.py $(DATA) --seed 42
	$(PYTHON) scripts/run_phase3.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/benchmark_family.py $(DATA) --family random_forest --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/benchmark_family.py $(DATA) --family xgboost --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/benchmark_family.py $(DATA) --family catboost --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/run_phase4.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest


phase5: envcheck validate
	$(PYTHON) scripts/run_phase5.py $(DATA) --assignments data/processed/phase2_split_assignments.csv
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest

phase6: envcheck validate
	$(PYTHON) scripts/run_phase6.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest


phase7: envcheck validate
	$(PYTHON) scripts/run_phase7.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) -m ruff check src scripts tests --fix-only
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest
