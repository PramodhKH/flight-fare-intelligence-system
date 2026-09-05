PYTHON ?= python
DATA ?= data/raw/Flight_Booking.csv

.PHONY: install validate test lint phase1 phase2

install:
	$(PYTHON) -m pip install -e ".[dev]"

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
