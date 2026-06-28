# Developer Environment Quickstart Guide

This guide takes you from a fresh clone to a running local pipeline with a test wallet being scored against the Stellar Testnet.

## ⚠️ Security Warning

**Before you begin:**
- **Never use mainnet keypairs for testing.** Testnet and mainnet configurations must be kept strictly separate.
- **Never commit `.env` files** to version control. The `.env` file may contain sensitive secrets.
- Environment variables and config files for testnet (`STELLAR_NETWORK=TESTNET`) and mainnet (`STELLAR_NETWORK=PUBLIC`) must be kept separate. Use `.env.testnet` for testing and never mix credentials between environments.

---

## Prerequisites

### Required Tools

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| Python | 3.11 | Core runtime |
| pip | 23.0+ | Package management |
| Docker | 24.0+ | Containerized services |
| Docker Compose | 2.20+ | Multi-container orchestration |
| make | GNU Make 4.0+ | Task automation |
| git | 2.30+ | Version control |

### Required Accounts & Resources

- **Stellar Testnet account** (generated automatically via Friendbot if needed)
- **Stellar Soroban CLI** (optional, for manual contract deployment)
- **Friendbot access** (free Testnet XLM distribution at https://friendbot.stellar.org)

---

## Step 1: Clone and Install

```bash
git clone https://github.com/Ledger-Lenz/Ledgerlens-data.git
cd Ledgerlens-data
```

Install Python dependencies:

```bash
make install
```

This installs all requirements from `requirements.txt` and sets up `ruff`/`black` for linting and formatting.

---

## Step 2: Configure Environment Variables

Copy the example environment file and configure it for your environment:

```bash
cp .env.example .env
```

Edit `.env` to set the required variables. The **minimum required variables** are:

```bash
# Required for basic operation
WATCHED_ASSET_PAIRS=USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN,XLM:native
```

For Testnet integration testing, you'll need additional variables (see Step 4).

> **Note:** If you only need local development with synthetic data, the defaults in `.env.example` work for most purposes.

---

## Step 3: Validate Your Environment

Before running the pipeline, validate all required environment variables:

```bash
make check-env
```

This command checks for required variables and produces clear, human-readable error messages for any missing configuration. It exits with code 0 if all variables are set, or non-zero with descriptive errors.

Expected output with all variables set:

```
✓ WATCHED_ASSET_PAIRS is set
✓ RISK_SCORE_DB_URL is set
✓ MODEL_DIR is set
Environment validation passed.
```

---

## Step 4: Generate Training Data and Train Models

For local development and testing, generate a synthetic dataset:

```bash
python -m scripts.generate_synthetic_dataset --output data/synthetic_dataset.parquet
python -m detection.model_training --data-path data/synthetic_dataset.parquet
```

This creates trained models in `./models` which are required for the scoring pipeline.

---

## Step 5: Run Docker Compose (Optional - Kafka Backend)

The default streaming backend uses SSE (Server-Sent Events). For the Kafka-based backend:

```bash
# Start Zookeeper, Kafka, Prometheus, and Grafana
docker-compose up -d

# Check services are healthy
docker-compose ps
```

The Kafka stack includes:
- **Zookeeper** (port 2181)
- **Kafka broker** (port 9092)
- **Prometheus** (port 9090) - Metrics collection
- **Grafana** (port 3000) - Dashboard (admin password from `GRAFANA_ADMIN_PASSWORD` env var)

To scale scorer workers:

```bash
docker-compose up --scale ledgerlens-scorer=3
```

---

## Step 6: Run Against Testnet

The `run-testnet` target starts Docker services and runs the streaming pipeline against Stellar Testnet with appropriate defaults:

```bash
make run-testnet
```

This command:
1. Sets `STELLAR_NETWORK=TESTNET` to target the test network
2. Uses `HORIZON_URL=https://horizon-testnet.stellar.org` (Testnet Horizon)
3. Uses `SOROBAN_RPC_URL=https://soroban-testnet.stellar.org` (Testnet Soroban RPC)
4. Starts the streaming pipeline (`scripts/stream.py`) with Testnet defaults

**Prerequisites for `run-testnet`:**
- Docker services must be running (`docker-compose up -d`)
- `WATCHED_ASSET_PAIRS` must be set in your `.env` file

---

## Step 7: Verify a Test Wallet's Risk Score

After the pipeline has processed some trades, you can score a specific wallet:

```bash
# Score a wallet against a specific asset pair
python -m scripts.score_wallet \
    --wallet G... \
    --pair "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
```

Or generate a forensic report:

```bash
python -m scripts.score_wallet \
    --wallet G... \
    --pair "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native" \
    --report --report-format markdown
```

---

## Step 8: Run the Test Suite

```bash
make test
```

This runs all unit tests (no network access). Integration tests require additional setup.

### Running Integration Tests

Integration tests hit the live Stellar Testnet and require a deployed contract:

```bash
# Deploy the contract (once per testnet reset)
python -m scripts.testnet_setup \
    --wasm-path ledgerlens_score.wasm \
    --wasm-sha256 <sha256-from-release> \
    --salt dev-testnet

# Run integration tests
export LEDGERLENS_INTEGRATION_TESTS=1
export $(grep -v '^#' .env.testnet | xargs)
pytest tests/integration/ -v --timeout=120
```

---

## Troubleshooting: Common Setup Errors

### 1. `ModuleNotFoundError` after `make install`

**Error:**
```
ModuleNotFoundError: No module named 'stellar_sdk'
```

**Solution:** Ensure you're using the virtual environment:
```bash
source .venv/bin/activate
make install
```

Or run via the venv Python directly:
```bash
.venv/bin/python -m scripts.stream
```

### 2. `WATCHED_ASSET_PAIRS is not set` error

**Error:**
```
LedgerLens configuration errors:
- WATCHED_ASSET_PAGS is not set.
```

**Solution:** Copy `.env.example` to `.env` and ensure `WATCHED_ASSET_PAIRS` is set:
```bash
cp .env.example .env
# Edit .env to include:
WATCHED_ASSET_PAIRS=USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN,XLM:native
```

Then validate:
```bash
make check-env
```

### 3. Docker Compose fails to start (port already in use)

**Error:**
```
Error: Bind for 0.0.0.0:9092 failed: port is already allocated
```

**Solution:** Either stop the conflicting service or change the port mapping in `docker-compose.yml`:
```bash
# Find what's using the port
sudo lsof -i :9092
# Stop it or modify docker-compose.yml ports section
```

### 4. Testnet Friendbot rate limited (429 error)

**Error:**
```
Friendbot rate-limited (429), retrying in 5s...
```

**Solution:** Wait 10 seconds before retrying. The setup script handles retries automatically. For repeated failures, generate a new keypair:
```bash
# Generate a new random keypair
python -c "from stellar_sdk import Keypair; print(Keypair.random().secret)"
```

### 5. No trained models found

**Error:**
```
No trained models found in ./models. Run 'python -m detection.model_training' first.
```

**Solution:** Generate synthetic data and train models:
```bash
python -m scripts.generate_synthetic_dataset --output data/synthetic_dataset.parquet
python -m detection.model_training --data-path data/synthetic_dataset.parquet
```

### 6. `.env` file accidentally committed

**Error:** Secret key leaked in git history.

**Solution:** Remove the file and rotate secrets:
```bash
# Remove from tracking (keep local copy)
git rm --cached .env 2>/dev/null || true
echo ".env" >> .gitignore
# Rotate any exposed secrets immediately
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `make install` | Install Python dependencies |
| `make check-env` | Validate required environment variables |
| `make run-testnet` | Start streaming pipeline against Testnet |
| `make test` | Run unit test suite |
| `docker-compose up -d` | Start Kafka stack |
| `docker-compose down` | Stop Kafka stack |
| `make lint` | Run code linting |
| `make format` | Auto-format code |

---

## Next Steps

- Read the [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Explore the [docs/](../docs/) directory for architecture details
- Join the LedgerLens community discussions for support