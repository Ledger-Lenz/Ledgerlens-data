.PHONY: install lint format test run run-testnet check-env typecheck

VENV_BIN := $(abspath .venv/bin)
ifeq ($(wildcard $(VENV_BIN)/python),)
  PYTHON := python3
  PIP := pip3
  RUFF := ruff
  BLACK := black
  PYTEST := pytest
else
  PYTHON := $(VENV_BIN)/python
  PIP := $(VENV_BIN)/pip
  RUFF := $(VENV_BIN)/ruff
  BLACK := $(VENV_BIN)/black
  PYTEST := $(VENV_BIN)/pytest
endif

install:
	$(PIP) install -r requirements.txt
	$(PIP) install ruff black

lint:
	$(RUFF) check .
	$(BLACK) --check .

format:
	$(RUFF) check --fix .
	$(BLACK) .

test:
	$(PYTEST) -q

run:
	$(PYTHON) run_pipeline.py

check-env:
	@$(PYTHON) scripts/check_env.py

run-testnet:
	@echo "Starting LedgerLens pipeline against Stellar Testnet..."
	@export STELLAR_NETWORK=TESTNET && \
	export HORIZON_URL=https://horizon-testnet.stellar.org && \
	export SOROBAN_RPC_URL=https://soroban-testnet.stellar.org && \
	$(PYTHON) -m scripts.stream
