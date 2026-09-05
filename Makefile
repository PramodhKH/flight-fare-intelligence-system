PYTHON ?= python
DATA ?= data/raw/Flight_Booking.csv

.PHONY: install envcheck validate test lint phase1 phase2 phase3 phase4 phase5

install:
	$(PYTHON) -m pip install -e ".[dev]"

envcheck:
	$(PYTHON) scripts/check_environment.py

validate:
	$(PYTHON) scripts/validate_data.py $(DATA) --strict

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests

phase1: validate test lint

phase2: validate
	$(PYTHON) scripts/run_phase2.py $(DATA) --seed 42
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check src scripts tests

phase3: validate
	$(PYTHON) scripts/run_phase2.py $(DATA) --seed 42
	$(PYTHON) scripts/run_phase3.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check src scripts tests

phase4: envcheck validate
	$(PYTHON) scripts/run_phase2.py $(DATA) --seed 42
	$(PYTHON) scripts/run_phase3.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/benchmark_family.py $(DATA) --family random_forest --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/benchmark_family.py $(DATA) --family xgboost --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/benchmark_family.py $(DATA) --family catboost --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) scripts/run_phase4.py $(DATA) --assignments data/processed/phase2_split_assignments.csv --seed 42
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check src scripts tests


phase5: envcheck validate
	$(PYTHON) scripts/run_phase5.py $(DATA) --assignments data/processed/phase2_split_assignments.csv
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check src scripts tests
