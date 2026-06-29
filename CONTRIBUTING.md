# Contributing to ledgerlens-data

Thanks for your interest in contributing to LedgerLens! This repo holds the
data ingestion and fraud-detection layer — see the README's
[Organization Map](README.md#organization-map) for how it fits with the
other LedgerLens repos.

## Security

Before implementing changes that touch API endpoints, model loading, training data, or database persistence, review the [Security Threat Model](docs/security_threat_model.md) for STRIDE analysis and attack surface identification. High-risk components may require security architect review.

## Development setup

```bash
git clone https://github.com/<org>/ledgerlens-data.git
cd ledgerlens-data
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env  # then edit as needed
```

## Running checks locally

```bash
make lint     # ruff + black --check
make format   # ruff --fix + black
make test     # pytest (unit tests only — no network)
```

Optionally install the pre-commit hooks so checks run automatically:

```bash
pip install pre-commit
pre-commit install
```

### Unit tests vs integration tests

`make test` runs `pytest tests/` and **never** hits the Testnet. All tests
under `tests/integration/` are automatically skipped unless
`LEDGERLENS_INTEGRATION_TESTS=1` is set.

To run the live Testnet integration tests locally:

```bash
# 1. Deploy the contract (once per testnet reset / keypair rotation)
python -m scripts.testnet_setup \
    --wasm-path ledgerlens_score.wasm \
    --wasm-sha256 <sha256-from-release> \
    --salt ci-testnet

# 2. Run integration tests
export LEDGERLENS_INTEGRATION_TESTS=1
export $(grep -v '^#' .env.testnet | xargs)
pytest tests/integration/ -v --timeout=120
```

See [`tests/integration/README.md`](tests/integration/README.md) for full
setup instructions, required environment variables, WASM version details,
and Testnet fee estimates.

The `testnet-integration.yml` CI workflow runs these tests on a weekly
schedule (Sundays 03:00 UTC) and on manual `workflow_dispatch` — it does
**not** run on pull requests so it never blocks a PR merge.

## Pull requests

- Keep PRs focused on a single logical change.
- Add or update tests for any behavior change.
- Run `make lint` and `make test` before opening a PR — CI runs the same
  checks on Python 3.11 and 3.12.
- If you change a shared contract (`RiskScore` shape, asset pair ID format,
  feature schema — see the README's "Shared Contracts" section), call that
  out in the PR description so consuming repos (`ledgerlens-core`,
  `ledgerlens-api`, `ledgerlens-contract`, `ledgerlens-dashboard`) can be
  updated.

## Security

See [`docs/security_threat_model.md`](docs/security_threat_model.md) for the comprehensive STRIDE-based threat model. Key mitigations:

- **Model integrity:** Ed25519 signatures on `metrics.json`; SHA-256 verification of `.joblib` files
- **Label poisoning:** HMAC-SHA256 on annotations; baseline distribution tracking
- **Model inversion:** Gaussian-mechanism DP on SHAP explanations; per-wallet query budgeting
- **Byzantine robustness:** Trimmed-mean ensemble voting
- **Credential security:** Never commit signing keys or API credentials to version control

All security-relevant PRs must reference the threat model and document which mitigations are affected.

## Code style

- Formatting/linting is enforced by `ruff` and `black` (see
  `pyproject.toml`). Line length is 100.
- Favor small, composable functions following the existing module layout:
  `ingestion/` for data acquisition, `detection/` for scoring logic,
  `tests/` mirrors both.
- New feature columns added to `detection/feature_engineering.py` must be
  documented in the README's feature tables and accounted for in
  `detection/model_training.py::FEATURE_COLUMNS_EXCLUDE` handling.

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include the asset
pair, wallet, and time window if reporting a detection accuracy issue —
that's usually enough to reproduce a Benford/feature calculation locally.
