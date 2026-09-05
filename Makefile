PYTHON ?= python
DATA ?= data/raw/Flight_Booking.csv
RUFF_TARGETS = src scripts tests api frontend

.PHONY: install envcheck validate test format autofix lint-fix lint quality \
	phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9 phase10 api dashboard stack

install:
	$(PYTHON) -m pip install -e ".[dev]"

envcheck:
	$(PYTHON) scripts/check_environment.py

validate:
	$(PYTHON) scripts/validate_data.py $(DATA) --strict

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m ruff format $(RUFF_TARGETS)

autofix:
	$(PYTHON) -m ruff check $(RUFF_TARGETS) --fix-only
	$(PYTHON) -m ruff format $(RUFF_TARGETS)

# Backward-compatible alias.
lint-fix: autofix

lint:
	$(PYTHON) -m ruff check $(RUFF_TARGETS)

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

phase8: envcheck validate
	$(PYTHON) -m scripts.run_phase8
	$(PYTHON) -m ruff check src scripts tests api --fix-only
	$(PYTHON) -m ruff format src scripts tests api
	$(PYTHON) -m ruff check src scripts tests api
	$(PYTHON) -m pytest

phase9: envcheck validate
	$(PYTHON) -m scripts.run_phase9
	$(PYTHON) -m ruff check $(RUFF_TARGETS) --fix-only
	$(PYTHON) -m ruff format $(RUFF_TARGETS)
	$(PYTHON) -m ruff check $(RUFF_TARGETS)
	$(PYTHON) -m pytest

phase10: envcheck validate
	$(PYTHON) -m scripts.run_phase9
	$(PYTHON) -m scripts.run_phase10 $(DATA)
	$(PYTHON) -m ruff check $(RUFF_TARGETS) --fix-only
	$(PYTHON) -m ruff format $(RUFF_TARGETS)
	$(PYTHON) -m ruff check $(RUFF_TARGETS)
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	FLIGHT_FARE_API_URL=$${FLIGHT_FARE_API_URL:-http://127.0.0.1:8000} \
		$(PYTHON) -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501

stack:
	docker compose up --build
