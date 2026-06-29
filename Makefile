.PHONY: install lint format test run scale-workers load-test
.PHONY: install lint format test run typecheck load-test

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
	python run_pipeline.py

scale-workers:
	@if [ -z "$(N)" ]; then \
		echo "Error: N is required. Usage: make scale-workers N=4"; \
		exit 1; \
	fi
	python -m scripts.kafka_workers --num-workers $(N)
	$(PYTHON) run_pipeline.py

# ---------------------------------------------------------------------------
# Load testing — disabled by default, enabled by RUN_LOAD_TESTS=1
#
# Usage:
#   make load-test                       # in-process, 500 tps, 120s
#   make load-test LOAD_RATE=1000        # custom rate
#   make load-test LOAD_KAFKA=1          # use live Kafka broker
#   RUN_LOAD_TESTS=1 make load-test      # enable in CI
#
# Results are written to reports/load_test_results.json.
# Pass/fail criteria (from docs/load_testing.md):
#   p99 latency < 10s at >= 500 tps
#   worker memory < 1024 MB
# ---------------------------------------------------------------------------
LOAD_RATE     ?= 500
LOAD_DURATION ?= 120
LOAD_RAMP     ?= 30
LOAD_OUTPUT   ?= reports/load_test_results.json
LOAD_KAFKA    ?= 0

load-test:
ifeq ($(RUN_LOAD_TESTS),1)
	@echo "==> Running load test at $(LOAD_RATE) tps for $(LOAD_DURATION)s..."
	@mkdir -p reports
	$(PYTHON) scripts/load_test_pipeline.py \
		--rate $(LOAD_RATE) \
		--duration $(LOAD_DURATION) \
		--ramp-time $(LOAD_RAMP) \
		--output $(LOAD_OUTPUT) \
		--fail-on-threshold \
		$(if $(filter 1,$(LOAD_KAFKA)),,--no-kafka)
else
	@echo "Load tests are disabled by default."
	@echo "Run: RUN_LOAD_TESTS=1 make load-test"
	@echo "Or:  make load-test LOAD_RATE=500  (and set RUN_LOAD_TESTS=1)"
endif
