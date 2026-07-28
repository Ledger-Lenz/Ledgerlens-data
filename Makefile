.PHONY: install lock lock-check lint format test run scale-workers typecheck mutation-test

VENV_BIN := $(abspath .venv/bin)
UV ?= uv
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

PY_TAG := $(shell $(PYTHON) -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
PLATFORM := $(shell $(PYTHON) -c 'import sys; print("macos" if sys.platform == "darwin" else "linux" if sys.platform.startswith("linux") else "unsupported")')
LOCKFILE := requirements/requirements-$(PLATFORM)-$(PY_TAG).txt

install: lock-check
	@test -f "$(LOCKFILE)" || (echo "No lockfile for $(PLATFORM) $(PY_TAG); supported: macOS/Linux on Python 3.11 or 3.12" && exit 1)
	$(PIP) install --require-hashes -r $(LOCKFILE)
	$(PIP) check

lock:
	$(UV) pip compile \
		--python-version 3.11 \
		--python-platform linux \
		--no-strip-extras \
		--generate-hashes \
		--upgrade \
		--custom-compile-command "make lock" \
		--output-file requirements/requirements-linux-py311.txt \
		requirements.in
	$(UV) pip compile \
		--python-version 3.12 \
		--python-platform linux \
		--no-strip-extras \
		--generate-hashes \
		--upgrade \
		--custom-compile-command "make lock" \
		--output-file requirements/requirements-linux-py312.txt \
		requirements.in
	$(UV) pip compile \
		--python-version 3.11 \
		--python-platform macos \
		--no-strip-extras \
		--generate-hashes \
		--upgrade \
		--custom-compile-command "make lock" \
		--output-file requirements/requirements-macos-py311.txt \
		requirements.in
	$(UV) pip compile \
		--python-version 3.12 \
		--python-platform macos \
		--no-strip-extras \
		--generate-hashes \
		--upgrade \
		--custom-compile-command "make lock" \
		--output-file requirements/requirements-macos-py312.txt \
		requirements.in
	$(PYTHON) scripts/validate_lockfiles.py --stamp

lock-check:
	$(PYTHON) scripts/validate_lockfiles.py

lint:
	$(RUFF) check .
	$(BLACK) --check .

format:
	$(RUFF) check --fix .
	$(BLACK) .

test:
	$(PYTEST) -q

fuzz:
	@echo "Running fuzz tests for 60 seconds each..."
	timeout 65 python tests/fuzz/fuzz_avro_codec.py tests/fuzz/corpus/ -max_len=10000 -timeout=10 || true
	timeout 65 python tests/fuzz/fuzz_horizon_response.py tests/fuzz/corpus/ -max_len=50000 -timeout=10 || true
	@echo "Fuzz testing complete."

test-e2e:
	@echo "Running end-to-end integration tests (requires LEDGERLENS_INTEGRATION_TESTS=1)..."
	LEDGERLENS_INTEGRATION_TESTS=1 $(PYTEST) tests/integration/test_full_pipeline_e2e.py -v --timeout=120

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
# Mutation testing — enforces ≥80% mutation score on the core scoring path
#
# Usage:
#   make mutation-test              # run and enforce threshold
#   make mutation-test THRESHOLD=70 # override threshold (for debugging)
#
# Runtime target: < 15 minutes in CI (--paths-to-mutate limits scope).
# Mutated files are never written to disk; mutmut restores originals after
# each probe, so no mutated code is persisted.
# ---------------------------------------------------------------------------
MUTATION_THRESHOLD ?= 80
MUTATION_PATHS = detection/benford_engine.py,detection/feature_engineering.py,detection/model_inference.py

mutation-test:
	@echo "==> Running mutation tests on core scoring path..."
	@echo "    Targets: $(MUTATION_PATHS)"
	@echo "    Threshold: $(MUTATION_THRESHOLD)%"
	mutmut run \
		--paths-to-mutate "$(MUTATION_PATHS)" \
		--runner "python -m pytest -x -q --timeout=30 -m 'not integration and not slow' \
			tests/test_benford.py \
			tests/test_benford_ci.py \
			tests/test_feature_engineering.py \
			tests/test_model_inference.py" \
		--no-progress || true
	@echo "==> Mutation results:"
	mutmut results || true
	$(PYTHON) scripts/check_mutation_score.py --threshold $(MUTATION_THRESHOLD)
