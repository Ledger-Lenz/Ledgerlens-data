# Developer Environment Quickstart Guide Implementation

## Summary

This PR implements the Developer Environment Quickstart Guide as requested in issue #266. It provides a cohesive zero-to-hero guide for new contributors, covering the complete setup process from cloning the repository to running the pipeline against Stellar Testnet.

## Changes

### 1. New Documentation: `docs/developer_quickstart.md`

A comprehensive quickstart guide that covers:
- **Security warnings** prominently warning against using mainnet keypairs for testing and committing `.env` files
- **Prerequisites** with required tools and versions
- **Step 1: Clone and Install** - basic repository setup
- **Step 2: Configure Environment Variables** - setting up `.env` with required variables
- **Step 3: Validate Your Environment** - using `make check-env`
- **Step 4: Generate Training Data and Train Models** - for local development
- **Step 5: Run Docker Compose (Optional)** - Kafka backend setup
- **Step 6: Run Against Testnet** - `make run-testnet` usage
- **Step 7: Verify a Test Wallet's Risk Score** - scoring commands
- **Step 8: Running the Test Suite** - unit and integration tests
- **Troubleshooting section** covering the 5 most common setup errors:
  1. ModuleNotFoundError after `make install`
  2. Missing `WATCHED_ASSET_PAIRS` error
  3. Docker Compose port conflicts
  4. Testnet Friendbot rate limiting (429 errors)
  5. Missing trained models

### 2. New Makefile Target: `make run-testnet`

Added to `Makefile`:
```makefile
run-testnet:
	@echo "Starting LedgerLens pipeline against Stellar Testnet..."
	@export STELLAR_NETWORK=TESTNET && \
	export HORIZON_URL=https://horizon-testnet.stellar.org && \
	export SOROBAN_RPC_URL=https://soroban-testnet.stellar.org && \
	$(PYTHON) -m scripts.stream
```

This target:
- Sets `STELLAR_NETWORK=TESTNET` to target the test network
- Uses Testnet Horizon and Soroban RPC endpoints
- Starts the streaming pipeline (`scripts/stream.py`) with Testnet defaults

### 3. New Makefile Target: `make check-env`

Added to `Makefile`:
```makefile
check-env:
	@$(PYTHON) scripts/check_env.py
```

Along with the new script `scripts/check_env.py` that validates required environment variables and produces clear, human-readable error messages (not Python tracebacks) for any missing configuration.

### 4. CI Tests for `make check-env`

Added to `.github/workflows/ci.yml`:
```yaml
- name: Test check-env target exits 0 with vars set
  run: |
    export WATCHED_ASSET_PAIRS=USDC:GA5Z...
    export RISK_SCORE_DB_URL=sqlite:///test.db
    export MODEL_DIR=./models
    make check-env

- name: Test check-env target exits non-zero with missing var
  run: |
    unset WATCHED_ASSET_PAIRS
    export RISK_SCORE_DB_URL=sqlite:///test.db
    export MODEL_DIR=./models
    if make check-env; then
      echo "Expected failure but check-env passed"
      exit 1
    fi
    echo "check-env correctly failed with missing WATCHED_ASSET_PAIRS"
```

### 5. Unit Tests: `tests/test_check_env.py`

Tests verify:
- `check-env` exits 0 with all required env vars set
- `check-env` exits non-zero with specific missing var and expected error message
- `check-env` reports multiple missing vars in one run
- Error messages are human-readable (no Python traceback)
- Whitespace-only values are treated as missing

## Security Considerations

The quickstart guide includes prominent warnings:
- Never use mainnet keypairs for testing
- Never commit `.env` files to version control
- Testnet and mainnet configs must be kept strictly separate

## Validation

All tests pass:
```
tests/test_check_env.py::test_check_env_passes_with_all_required_vars_set PASSED
tests/test_check_env.py::test_check_env_fails_with_missing_watched_asset_pairs PASSED
tests/test_check_env.py::test_check_env_fails_with_missing_risk_score_db_url PASSED
tests/test_check_env.py::test_check_env_fails_with_missing_model_dir PASSED
tests/test_check_env.py::test_check_env_reports_multiple_missing_vars PASSED
tests/test_check_env.py::test_check_env_handles_whitespace_as_missing PASSED
```

closes #266