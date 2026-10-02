.PHONY: install lint format test run scale-workers typecheck mutation-test threshold-sweep anonymization-check check-env check-schema-compatibility check-review-gates ops-check ops-validate static-analysis benchmark verify-lockfile regenerate-lockfile dependency-risk-report dependency-risk-report-osv partition-write partition-read retention-scan snapshot-freeze snapshot-list snapshot-verify run-compare run-compare-all check-cycles probe-deps probe-deps-json validate-readme validate-readme-warn validate-notebooks validate-notebooks-strict validate-notebooks-ci validate-all check-integrity dead-path-report env-docs env-docs-check migrate migrate-status migrate-dry-run new-migration onboard onboard-fix onboard-json maturity-report check-stale-maturity check-deps scan-secrets
.ONESHELL:

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

test-fast:
	$(PYTEST) -q -m "not integration and not slow" --ignore=tests/fuzz

check-env-example:
	$(PYTHON) -m scripts.check_env_example

# ---------------------------------------------------------------------------
# Source package integrity checks (Issue #540)
#
# Filesystem + AST sweep for missing __init__.py, unresolved merge conflict
# markers, syntax errors, and empty modules. Runs automatically before every
# `pytest` session (see tests/conftest.py::pytest_sessionstart) — this
# target is for running it standalone, independent of the test suite.
# ---------------------------------------------------------------------------
check-integrity:
	$(PYTHON) scripts/check_package_integrity.py

# ---------------------------------------------------------------------------
# Dead-path detection for retired modules (Issue #547)
#
# Reports source modules with no inbound Python import and no reference in
# Makefile/CI/docs — candidates for removal. Read-only, never deletes code.
# ---------------------------------------------------------------------------
dead-path-report:
	$(PYTHON) scripts/detect_dead_paths.py

# ---------------------------------------------------------------------------
# Environment contract docs generated from config schemas (Issue #544)
#
# `env-docs` regenerates docs/environment_contract.md from config.py.
# `env-docs-check` fails (without writing) if the committed doc has drifted
# from config.py — wire this into CI to keep the contract doc honest.
# ---------------------------------------------------------------------------
env-docs:
	$(PYTHON) scripts/generate_env_contract_docs.py

env-docs-check:
	$(PYTHON) scripts/generate_env_contract_docs.py --check

# ---------------------------------------------------------------------------
# Review gates for high-risk data and model changes
#
#   make check-schema-compatibility   # Avro wire-schema compatibility
#   make check-review-gates           # high-risk path acknowledgement
#
# check-schema-compatibility is the exact command CI runs.
#
# check-review-gates reports what CI would decide but always exits 0, since
# locally there is usually no pull-request body yet. Unlike CI — which sees a
# pushed branch — it also counts uncommitted and untracked files, so it is
# useful before you commit. Reads the body from PR_BODY_FILE when supplied:
#
#   make check-review-gates PR_BODY_FILE=/tmp/body.md
#
# To reproduce a CI failure exactly, drop --dry-run:
#
#   python scripts/check_review_gates.py \
#       --changed-paths-from <file> --pr-body-file <file>
# ---------------------------------------------------------------------------
BASE ?= main
PR_BODY_FILE ?=

check-schema-compatibility:
	$(PYTHON) scripts/check_schema_compatibility.py

check-feature-labels:
	$(PYTHON) scripts/check_feature_label_consistency.py

check-report-schemas:
	$(PYTHON) scripts/check_report_schema_compatibility.py

check-review-gates:
	@{ \
		git diff --name-only $(BASE)...HEAD; \
		git diff --name-only HEAD; \
		git ls-files --others --exclude-standard; \
	} | sort -u > .changed_paths.tmp
	@$(PYTHON) scripts/check_review_gates.py \
		--changed-paths-from .changed_paths.tmp \
		$(if $(PR_BODY_FILE),--pr-body-file $(PR_BODY_FILE),--pr-body "") \
		--dry-run
	@rm -f .changed_paths.tmp

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

# Validate environment configuration contracts (config/contracts.py) without
# starting the service. Usage:
#   make check-env MODE=api        # validate one runtime mode
#   make check-env                 # validate every known runtime mode
check-env:
	$(PYTHON) -m scripts.check_env $(if $(MODE),--mode $(MODE),--all)

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
# Issue #964: also scope to score_normaliser and risk_propagation
MUTATION_PATHS_EXTENDED = detection/benford_engine.py,detection/score_normaliser.py,detection/risk_propagation.py

mutation-test:
	@echo "==> Running mutation tests on core scoring path..."
	@echo "    Targets: $(MUTATION_PATHS)"
	@echo "    Threshold: $(MUTATION_THRESHOLD)%"
	mutmut run \
		--paths-to-mutate "$(MUTATION_PATHS)" \
		--runner "python -m pytest -x -q --timeout=30 -m 'not integration and not slow' \
			tests/test_benford.py \
			tests/test_benford_ci.py \
			tests/test_mutation_coverage.py \
			tests/test_feature_engineering.py \
			tests/test_model_inference.py" \
		--no-progress || true
	@echo "==> Mutation results:"
	mutmut results || true
	$(PYTHON) scripts/check_mutation_score.py --threshold $(MUTATION_THRESHOLD)

# ---------------------------------------------------------------------------
# Import cycle detection (Issue #546)
#
# Usage:
#   make check-cycles                  # check all packages
#   make check-cycles PACKAGES="detection ingestion"  # specific packages
#
# Exit codes: 0 = clean, 2 = cycles found.
# ---------------------------------------------------------------------------
CYCLE_PACKAGES ?=

check-cycles:
	@echo "==> Checking for import cycles..."
	@if [ -n "$(CYCLE_PACKAGES)" ]; then \
		$(PYTHON) scripts/check_import_cycles.py --packages $(CYCLE_PACKAGES); \
	else \
		$(PYTHON) scripts/check_import_cycles.py; \
	fi

# ---------------------------------------------------------------------------
# Module dependency / layering rules (config/module_boundaries.yml, Issue #791)
#
# Usage:
#   make check-boundaries                  # check all packages
#   make check-boundaries PACKAGE=detection  # scope to one package
#
# Exit codes: 0 = within declared boundaries, 1 = one or more violations.
# ---------------------------------------------------------------------------

check-boundaries:
	@echo "==> Checking module dependency / layering boundaries..."
	$(PYTHON) scripts/check_module_dependencies.py $(if $(PACKAGE),--package $(PACKAGE),)

# ---------------------------------------------------------------------------
# Optional dependency probes (Issue #542)
#
# Usage:
#   make probe-deps                   # probe all groups
#   make probe-deps PROBE_GROUPS="gnn kafka"   # probe specific groups
#   make probe-deps-json              # JSON output
#
# Exit codes: 0 = all available, 2 = some missing.
# ---------------------------------------------------------------------------
PROBE_GROUPS ?=

probe-deps:
	@echo "==> Probing optional dependencies..."
	@if [ -n "$(PROBE_GROUPS)" ]; then \
		$(PYTHON) -m utils.dependency_probe --groups $(PROBE_GROUPS); \
	else \
		$(PYTHON) -m utils.dependency_probe; \
	fi

probe-deps-json:
	$(PYTHON) -m utils.dependency_probe --json

# ---------------------------------------------------------------------------
# README examples validation (Issue #548)
#
# Usage:
#   make validate-readme              # validate README.md only
#   make validate-readme DOCS="README.md docs/"  # include docs/ directory
#   make validate-readme-warn         # warn-only mode (never fails CI)
#
# Exit codes: 0 = all valid, 2 = broken references found.
# ---------------------------------------------------------------------------
DOCS ?= README.md

validate-readme:
	@echo "==> Validating README bash examples..."
	$(PYTHON) scripts/validate_readme_examples.py --docs $(DOCS)

validate-readme-warn:
	$(PYTHON) scripts/validate_readme_examples.py --docs $(DOCS) --warn-only

# ---------------------------------------------------------------------------
# Notebook validation (Issue #549)
#
# Usage:
#   make validate-notebooks           # structure checks (default)
#   make validate-notebooks-strict    # structure + outputs + strict markers
#   make validate-notebooks-ci        # full CI gate (outputs + exec counts + strict)
#
# Exit codes: 0 = pass, 2 = failures found.
# ---------------------------------------------------------------------------

validate-notebooks:
	@echo "==> Validating notebooks (structure)..."
	$(PYTHON) scripts/validate_notebooks.py

validate-notebooks-strict:
	$(PYTHON) scripts/validate_notebooks.py --strict

validate-notebooks-ci:
	@echo "==> Validating notebooks (CI gate: outputs + execution count + strict)..."
	$(PYTHON) scripts/validate_notebooks.py --check-outputs --check-execution-count --strict

# ---------------------------------------------------------------------------
# validate-all — run all 4 validation tools in sequence
#
# Usage:  make validate-all
# ---------------------------------------------------------------------------

validate-all: check-cycles probe-deps validate-readme validate-notebooks
	@echo "==> All validation checks complete."

# ---------------------------------------------------------------------------
# Repository health diagnostics (Issue #477)
#
# Usage:
#   make diagnose                    # Run all diagnostic checks (text output)
#   make diagnose-json               # JSON output for CI/automation
#   make diagnose CATEGORIES="environment dependencies"  # Filter by category
#
# Exit codes: 0 = healthy, 1 = failures detected
# ---------------------------------------------------------------------------

CATEGORIES ?=

diagnose:
	@echo "==> Running repository health diagnostics..."
	$(PYTHON) -m scripts.diagnose $(if $(CATEGORIES),--categories $(CATEGORIES),)

diagnose-json:
	$(PYTHON) -m scripts.diagnose --json

# ---------------------------------------------------------------------------
# Issue #527 — Partitioned dataset helpers
#
# Usage:
#   make partition-write DATA=data/synthetic_dataset.parquet STRATEGY=time PERIOD=month
#   make partition-read  ROOT=data/partitioned
# ---------------------------------------------------------------------------
DATA ?= data/synthetic_dataset.parquet
STRATEGY ?= time
PERIOD ?= month
PARTITION_ROOT ?= data/partitioned

partition-write:
	@echo "==> Writing partitioned dataset from $(DATA) (strategy=$(STRATEGY), period=$(PERIOD))"
	$(PYTHON) - <<-'EOF'
	import pandas as pd
	from pathlib import Path
	from data.partitioning import PartitionedDatasetWriter, TimePartitionStrategy, PairPartitionStrategy, WalletPartitionStrategy

	strat_name = "$(STRATEGY)"
	strategy = TimePartitionStrategy(period="$(PERIOD)") if strat_name == "time" else (PairPartitionStrategy() if strat_name == "pair" else WalletPartitionStrategy())
	df = pd.read_parquet("$(DATA)")
	writer = PartitionedDatasetWriter(root=Path("$(PARTITION_ROOT)"), strategy=strategy)
	result = writer.write(df)
	print(f"Written {sum(result.values())} rows across {len(result)} partition(s).")
	for k, n in sorted(result.items()): print(f"  {k}: {n} rows")
	EOF

partition-read:
	@echo "==> Listing partitions under $(PARTITION_ROOT)"
	$(PYTHON) - <<-'EOF'
	from pathlib import Path
	from data.partitioning import PartitionedDatasetReader
	reader = PartitionedDatasetReader(Path("$(PARTITION_ROOT)"))
	parts = reader.list_partitions()
	print(f"Found {len(parts)} partition(s):")
	for p in parts: meta = reader.get_metadata(p); rows = meta.row_count if meta else "?"; print(f"  {p}: {rows} rows")
	EOF

# ---------------------------------------------------------------------------
# Issue #530 — Data retention scan / purge
#
# Usage:
#   make retention-scan              # dry-run using default policies
#   make retention-scan DRY_RUN=0   # live purge
# ---------------------------------------------------------------------------
DRY_RUN ?= 1

retention-scan:
	@echo "==> Running retention scan (dry_run=$(DRY_RUN))"
	$(PYTHON) - <<-'EOF'
	from data.retention import FileRetentionManager, DEFAULT_POLICIES
	from pathlib import Path
	dry = "$(DRY_RUN)" != "0"
	manager = FileRetentionManager(
	    policies=DEFAULT_POLICIES,
	    base_dir=Path("."),
	    audit_log_path=Path("reports/retention_audit.ndjson"),
	)
	report = manager.run(dry_run=dry)
	print(report.summary())
	EOF

# ---------------------------------------------------------------------------
# Issue #533 — Dataset snapshot helpers
#
# Usage:
#   make snapshot-freeze SOURCE=data/synthetic_dataset.parquet LABEL=pre-training
#   make snapshot-list
#   make snapshot-verify SNAPSHOT_ID=snap_20240601T000000_pre-training_abc12345
# ---------------------------------------------------------------------------
SOURCE ?= data/synthetic_dataset.parquet
LABEL ?= manual
SNAPSHOT_ROOT ?= data/snapshots
SNAPSHOT_ID ?=

snapshot-freeze:
	@echo "==> Freezing snapshot of $(SOURCE) (label=$(LABEL))"
	$(PYTHON) - <<-'EOF'
	from pathlib import Path
	from data.reproducibility import DatasetSnapshot
	snap = DatasetSnapshot(snapshot_root=Path("$(SNAPSHOT_ROOT)"))
	m = snap.freeze(Path("$(SOURCE)"), label="$(LABEL)")
	print(f"Snapshot created: {m.snapshot_id}")
	print(f"  sha256   : {m.sha256}")
	print(f"  rows     : {m.row_count}")
	print(f"  columns  : {m.columns}")
	EOF

snapshot-list:
	@echo "==> Listing snapshots under $(SNAPSHOT_ROOT)"
	$(PYTHON) - <<-'EOF'
	from pathlib import Path
	from data.reproducibility import SnapshotRegistry
	reg = SnapshotRegistry(Path("$(SNAPSHOT_ROOT)"))
	manifests = reg.list_all()
	print("No snapshots found.") if not manifests else [print(f"  {m.snapshot_id}  label={m.label!r}  rows={m.row_count}  sha256={m.sha256[:12]}...") for m in manifests]
	EOF

snapshot-verify:
	@if [ -z "$(SNAPSHOT_ID)" ]; then echo "Error: SNAPSHOT_ID is required.  Usage: make snapshot-verify SNAPSHOT_ID=<id>"; exit 1; fi
	@echo "==> Verifying snapshot $(SNAPSHOT_ID)"
	$(PYTHON) - <<-'EOF'
	from pathlib import Path
	from data.reproducibility import DatasetSnapshot
	snap = DatasetSnapshot(snapshot_root=Path("$(SNAPSHOT_ROOT)"))
	ok = snap.verify("$(SNAPSHOT_ID)")
	print("Integrity OK" if ok else "Integrity FAILED")
	EOF

# ---------------------------------------------------------------------------
# Issue #534 — Model run comparison
#
# Usage:
#   make run-compare BASELINE=run_20240601 CANDIDATE=run_20240701 RUNS_DIR=models
#   make run-compare-all RUNS_DIR=models
# ---------------------------------------------------------------------------
BASELINE ?=
CANDIDATE ?=
RUNS_DIR ?= models
COMPARE_TOLERANCE ?= 0.01

run-compare:
	@if [ -z "$(BASELINE)" ] || [ -z "$(CANDIDATE)" ]; then \
		echo "Error: BASELINE and CANDIDATE are required."; \
		echo "Usage: make run-compare BASELINE=run_A CANDIDATE=run_B RUNS_DIR=models"; \
		exit 1; \
	fi
	@echo "==> Comparing runs $(BASELINE) → $(CANDIDATE) (tolerance=$(COMPARE_TOLERANCE))"
	$(PYTHON) - <<-'EOF'
	from pathlib import Path
	from evaluation.run_comparator import ModelRunComparator
	comp = ModelRunComparator(runs_dir=Path("$(RUNS_DIR)"), metrics_filename="metrics.json")
	report = comp.compare("$(BASELINE)", "$(CANDIDATE)", regression_tolerance=float("$(COMPARE_TOLERANCE)"))
	print(report.summary())
	out = Path("reports/run_comparison_$(BASELINE)_vs_$(CANDIDATE).json")
	report.save(out)
	print(f"Report saved to {out}")
	EOF

run-compare-all:
	@echo "==> Comparing all consecutive runs under $(RUNS_DIR)"
	$(PYTHON) - <<-'EOF'
	from pathlib import Path
	from evaluation.run_comparator import ModelRunComparator
	comp = ModelRunComparator(runs_dir=Path("$(RUNS_DIR)"), metrics_filename="metrics.json")
	reports = comp.compare_all(regression_tolerance=float("$(COMPARE_TOLERANCE)"))
	print("No consecutive run pairs found.") if not reports else [print(r.summary()) for r in reports]
	EOF

# ---------------------------------------------------------------------------
# Static analysis gate — mypy + bandit + radon (issue #545)
# ---------------------------------------------------------------------------
static-analysis:
	@echo "==> Running repository-wide static analysis gate..."
	$(PYTHON) scripts/static_analysis_gate.py

# Run only mypy (fast, no subprocess)
typecheck:
	$(PYTHON) -m mypy detection ingestion streaming ci_metrics benchmarks utils config.py

# ---------------------------------------------------------------------------
# Benchmark datasets — run detector benchmarks (issue #537)
# ---------------------------------------------------------------------------
benchmark:
	@echo "==> Running detector benchmark suite..."
	$(PYTHON) -m benchmarks.datasets
	@echo "To run benchmarks against a detector, see benchmarks/runner.py"

# ---------------------------------------------------------------------------
# Lockfile verification (issue #541)
# ---------------------------------------------------------------------------
verify-lockfile:
	@echo "==> Verifying installed environment matches requirements.lock..."
	$(PYTHON) scripts/verify_lockfile.py

regenerate-lockfile:
	@echo "==> Regenerating requirements.lock from current environment..."
	$(PYTHON) scripts/verify_lockfile.py --generate

# ---------------------------------------------------------------------------
# Dependency risk reporting (Issue #478)
#
# The default report is deterministic and offline. Use dependency-risk-report-osv
# when network access is available to enrich it with OSV.dev advisories.
# ---------------------------------------------------------------------------
dependency-risk-report:
	$(PYTHON) scripts/dependency_risk_report.py

dependency-risk-report-osv:
	$(PYTHON) scripts/dependency_risk_report.py --osv

# ---------------------------------------------------------------------------
# Threshold sweep — run threshold diagnostics on a backtest dataset
#
# Usage:
#   make threshold-sweep DATASET=data/backtest.parquet
#   make threshold-sweep DATASET=data/backtest.parquet OUTPUT=reports/
# ---------------------------------------------------------------------------
DATASET ?= data/backtest.parquet
SWEEP_OUTPUT ?= reports/

threshold-sweep:
	@echo "==> Running threshold sweep diagnostics..."
	@echo "    Dataset: $(DATASET)"
	@echo "    Output:  $(SWEEP_OUTPUT)"
	$(PYTHON) -m evaluation.backtest $(DATASET) $(SWEEP_OUTPUT) --sweep
	@echo "==> Threshold sweep complete. Report in $(SWEEP_OUTPUT)/backtest_report.json"

# ---------------------------------------------------------------------------
# Anonymization check — ensure shared example data is free of PII
#
# Usage:
#   make anonymization-check
# ---------------------------------------------------------------------------
anonymization-check:
	@echo "==> Running anonymization checks..."
	$(PYTHON) scripts/check_anonymization.py --target data tests/fixtures tests/fuzz/corpus
	@echo "==> Anonymization check complete."

ops-check:
	python -m cli.main healthcheck

ops-validate:
	python -m cli.main validate-artifacts

# ---------------------------------------------------------------------------
# Repository maturity report (Issue #958)
#
# Usage:
#   make maturity-report             # write docs/maturity_report.md
#   make maturity-report BADGES=docs/badges   # also write per-module badges
#   make check-stale-maturity        # CI stale-entry check (fast)
#
# Exit codes for check-stale-maturity: 0 = clean, 1 = dangling references.
# ---------------------------------------------------------------------------
MATURITY_OUTPUT ?= docs/maturity_report.md
BADGES ?=

maturity-report:
	@echo "==> Generating repository maturity report..."
	$(PYTHON) scripts/generate_maturity_report.py \
		--output $(MATURITY_OUTPUT) \
		$(if $(BADGES),--badges $(BADGES),)
	@echo "==> Report written to $(MATURITY_OUTPUT)"

check-stale-maturity:
	@echo "==> Checking repo_maturity.yaml for stale path references..."
	$(PYTHON) scripts/generate_maturity_report.py --check-stale --quiet

# ---------------------------------------------------------------------------
# Module dependency / boundary enforcement alias (Issue #957)
#
# ``check-deps`` is the CI-facing name; ``check-boundaries`` also works
# (see the existing target above).
#
# Usage:
#   make check-deps
#   make check-deps PACKAGE=detection
# ---------------------------------------------------------------------------
check-deps:
	@echo "==> Enforcing module layering rules (config/module_boundaries.yml)..."
	$(PYTHON) scripts/check_module_dependencies.py $(if $(PACKAGE),--package $(PACKAGE),)

# ---------------------------------------------------------------------------
# Secret-detection scan (Issue #956)
#
# Usage:
#   make scan-secrets                  # scan default target paths
#   make scan-secrets SCAN_PATHS="config/ .env.example"
#
# Exit codes: 0 = clean, 1 = secret-shaped values found, 2 = crash.
# ---------------------------------------------------------------------------
SCAN_PATHS ?=

scan-secrets:
	@echo "==> Running secret-detection scan..."
	$(PYTHON) scripts/scan_secrets.py $(if $(SCAN_PATHS),--paths $(SCAN_PATHS),)
