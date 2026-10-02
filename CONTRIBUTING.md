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
make check-env-example  # verify .env.example covers every config.py variable
```

### Module boundary enforcement (Issue #957)

This repo enforces a strict layering rule between its top-level packages:
`foundation` (utils, config) → `domain` (detection, ingestion, streaming, …) → `entrypoint` (api, scripts).
A lower-layer package must never import from a higher-layer one.  The rules
are declared in `config/module_boundaries.yml` and enforced by:

```bash
make check-deps           # check all packages
make check-deps PACKAGE=detection  # check only one package
```

This check also runs as a required step in CI (`Check module dependency rules`
in `.github/workflows/ci.yml`).  A PR that introduces a boundary violation
will fail CI — fix it by restructuring the import or moving shared code to
`utils/` or `config/`.

To deliberately test that the check catches violations, temporarily add a
`from api import ...` line in any `detection/` file, run `make check-deps`,
and observe the violation message, then revert.

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

### Running a subset of tests

With 300+ files under `tests/`, running the full suite on every iteration is
slow. Use these `pytest` invocations to scope a run down while you iterate:

```bash
# A single test file
pytest tests/test_benford.py

# A single test function (-k matches by substring)
pytest tests/test_benford.py -k test_chi_square_statistic

# Everything except the slower integration and fuzz suites
pytest tests/ --ignore=tests/integration --ignore=tests/fuzz
```

`pyproject.toml` defines these pytest markers (`-m 'not integration and not
slow'` to exclude both):

| Marker | Meaning |
|---|---|
| `integration` | Live Testnet integration tests — deselect with `-m "not integration"` (also skipped automatically unless `LEDGERLENS_INTEGRATION_TESTS=1` is set, see above) |
| `slow` | Tests that run PPO training — deselect with `-m "not slow"` |
| `concurrency` | Concurrency validation tests for streaming workers — included by default |

`tests/fuzz/` is a separate atheris-based fuzzing suite, not run by plain
`pytest`; see [`tests/fuzz/README.md`](tests/fuzz/README.md) and `make fuzz`.

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

### API contract compatibility

Any PR that touches `api/app.py` or anything under `contracts/` is gated by
[`scripts/check_api_compatibility.py`](scripts/check_api_compatibility.py),
which diffs the public API surface against the committed baseline and fails
the build on a breaking change. The check runs in CI via
[`.github/workflows/api-compatibility.yml`](.github/workflows/api-compatibility.yml)
on every pull request that modifies those paths.

Run it locally before opening such a PR:

```bash
python scripts/check_api_compatibility.py
```

#### Intentionally shipping a breaking change

A breaking change is only allowed when it is deliberate and acknowledged:

1. **Bump the API version.** Update the version constant in `api/app.py`
   (and the matching entry in `contracts/`) so the new surface is published
   under a new major/minor version.
2. **Regenerate the baseline.** Re-run the checker with the update flag to
   record the new contract as the accepted baseline:

   ```bash
   python scripts/check_api_compatibility.py --update-baseline
   ```

   Commit the regenerated baseline alongside the version bump so the CI gate
   sees the change as acknowledged rather than accidental.
3. **Notify downstream consumers.** Call out the break in the PR description
   and in the `CHANGELOG.md` entry, and notify the consuming repos
   (`ledgerlens-core`, `ledgerlens-api`, `ledgerlens-contract`,
   `ledgerlens-dashboard`) so they can pin or migrate before the release.

Without the version bump and regenerated baseline, the CI job fails and the
PR cannot merge.

### Changelog entries

Every PR that touches high-impact paths must include a `CHANGELOG.md` entry
under `## [Unreleased]`. The entry is enforced by
[`scripts/validate_changelog.py`](scripts/validate_changelog.py) in CI.

**Required format (Keep a Changelog):**

```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Behavior change description

### Fixed
- Bug fix description
```

- Entries must start with `- ` and be grouped under one of the standard
  subsections: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
  `Security`.
- If your PR only touches documentation, CI configuration, or tests, a
  changelog entry is not required — but you still need to check the
  "Added a CHANGELOG.md entry, or this PR is exempt" box in the PR template.
- The CI job
  [`.github/workflows/changelog-validation.yml`](.github/workflows/changelog-validation.yml)
  runs `python scripts/validate_changelog.py --check-pr` on every pull
  request and fails the build if a high-impact path changed without a
  matching entry.

## Metrics labeling guidelines

Metrics emitted through `monitoring/metrics_collector.py` must keep label
cardinality bounded. Unbounded label values (raw wallet addresses,
transaction IDs, block hashes, free-form user input) multiply the number of
time series and can overload the metrics backend and blow up storage cost.

**Rules for new metrics:**

- **Never** use a raw identifier as a label value. This includes wallet
  addresses, transaction IDs/hashes, block numbers, request IDs, and any
  other per-event unique value.
- Prefer a small, fixed set of label values (e.g. `asset_pair`, `chain`,
  `status`, `severity`). If a label can take more than a few dozen distinct
  values, it is a cardinality risk.
- To attribute a metric to a specific entity, use a bounded bucket instead
  of the raw value — e.g. hash the identifier into a fixed number of shards
  (`wallet_shard="0".."15"`) or use a coarse category (`wallet_type`).
- Keep the total number of label combinations per metric small. A metric
  with labels `a` (10 values) and `b` (10 values) already produces 100
  series; adding a high-cardinality label multiplies that by the number of
  distinct values.
- When in doubt, emit the detail as a log line or a structured event rather
  than a metric label.

**Enforcement:**

- The runtime guardrail in `monitoring/metrics_collector.py` flags or
  rejects emissions whose label values look like high-cardinality
  identifiers (long hex/base58 strings, UUIDs, etc.).
- A CI check scans new metric-emission code for known high-cardinality-risk
  patterns (label values sourced directly from user/transaction
  identifiers) and fails the build when one is introduced.

If a metric genuinely needs a high-cardinality dimension, open an issue to
discuss an aggregation strategy before adding it.

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
- **Adding a new top-level module?** Also add a corresponding pattern to
  [`.github/CODEOWNERS`](.github/CODEOWNERS) — see
  [`.github/review-checklists.md`](.github/review-checklists.md) for the
  expected review-gate entry and an example.
- New feature columns added to `detection/feature_engineering.py` must be
  documented in the README's feature tables and accounted for in
  `detection/model_training.py::FEATURE_COLUMNS_EXCLUDE` handling.
- **Adding a new ML feature?** Follow the end-to-end guide in
  [`docs/contributor_feature_guide.md`](docs/contributor_feature_guide.md).
  It covers naming conventions, function signatures, range validation,
  dataset card updates, SHAP integration, and required test patterns —
  with a complete worked example using `counterparty_variance`.
- **Adding or renaming a feature? Three places must stay in sync (Issue #946):**
  1. `detection/feature_engineering.py` — add/rename the feature column.
  2. `data/feature_dictionary.md` — add a new `### N.M · \`feature_name\`` entry
     with Formula, Range, Empirical p1–p99, High/Low value, and Mutable fields
     (copy the template from any existing entry in the same section).
  3. `reporting/feature_labels.py` — if the feature should appear in narrative
     reports, add a plain-English label to `FEATURE_LABELS`.
  Run `python scripts/check_feature_label_consistency.py` locally (or
  `make check-feature-labels`) to confirm all three are consistent before
  opening a PR — the CI will also enforce this automatically.
- **Replacing a trained model artifact?** Run `make validate-artifacts`
  before committing — see
  [`docs/artifact_backward_compatibility.md`](docs/artifact_backward_compatibility.md)
  for the backward compatibility rules enforced against archived versions.
- **Deprecating a public function or class?** Use the `@deprecated` decorator
  documented in
  [`docs/deprecation_policy.md`](docs/deprecation_policy.md) and run
  `make check-deprecations` so removal versions are tracked and enforced.
- Run `make validate-docs` after editing any doc linked from this file — it
  checks for dead links, missing headings, and broken code examples.

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include the asset
pair, wallet, and time window if reporting a detection accuracy issue —
that's usually enough to reproduce a Benford/feature calculation locally.

## Mutation testing

LedgerLens uses [mutmut](https://github.com/boxed/mutmut) to measure test
*effectiveness*, not just coverage. A mutation score of **≥ 80%** is
enforced in CI on the core scoring path:

- `detection/benford_engine.py`
- `detection/feature_engineering.py`
- `detection/model_inference.py`

### Running mutation tests locally

```bash
# Full run — same as CI (may take 10–15 minutes)
make mutation-test

# Run only and inspect results
mutmut run \
  --paths-to-mutate "detection/benford_en

/* … truncated 2671 chars — edit only what you need near the top … */
# Show a summary of all mutation outcomes
mutmut results

# Check whether the score meets the 80% threshold
python scripts/check_mutation_score.py --threshold 80
```

### Interpreting the results

| Status | Meaning |
|---|---|
| `ok` | Mutation **killed** — at least one test caught the change ✓ |
| `survived` | Mutation **survived** — the test suite didn't detect the logic error ✗ |
| `suspicious` | Tests passed but with timing/output differences — treated as killed |
| `timeout` | Test run timed out — treated as killed |
| `ba_error` | mutmut could not apply the mutation — excluded from the score |

The **mutation score** is `killed / (killed + survived) × 100`. The CI step
fails when this drops below 80%.

### Investigating surviving mutations

```bash
# Show the diff for a specific surviving mutation (ID from `mutmut results`)
mutmut show <ID>

# Apply the mutation locally, run tests manually, then restore
mutmut apply <ID>
pytest tests/test_benford.py -v    # add a test that catches this case
mutmut unapply <ID>

# Re-run only the surviving mutations (much faster after fixing tests)
mutmut rerun
```

### Which mutation operators matter most for a fraud-detection ML pipeline

1. **Relational operators** (`>` ↔ `>=`, `<` ↔ `<=`): threshold comparisons in
   `bft_trimmed_mean`, `_has_consensus`, `MAD_NONCONFORMITY_THRESHOLD`, and
   `ML_FLAG_THRESHOLD` are the highest-risk off-by-one sites.
2. **Arithmetic operators** (`+` ↔ `-`, `*` ↔ `/`): the chi-square and Z-score
   formulas in `benford_engine.py` contain squared differences and
   square-root normalisation that silently produce wrong scores when mutated.
3. **Boolean literals and conditions** (`True`/`False` flips, `and`/`or` swaps):
   the `diverged`, `consensus_failure`, and `benford_flag` guards must be
   tested explicitly with boundary-value inputs.
4. **Return values** (mutating the returned constant 0.0, 1.0, etc.): empty-input
   fallback paths in feature functions often return sentinel zeros that tests
   must assert are *exactly* zero, not just non-negative.

### Security note

mutmut applies mutations in-process using Python AST manipulation and
restores the original file after every test run. **Mutated code is never
committed, never persisted to `models/`, and never reaches the network.**
The CI job runs in a dedicated `mutation-test` job isolated from the
regular `test` matrix.


## Updating offline stubs when real integration behaviour changes

`integrations/offline_stubs.py::StubContractClient` is an in-memory drop-in
for `LedgerLensContractClient`.  It is used in local development, unit tests,
and CI jobs where a live Testnet keypair and deployed contract are unavailable.

### Why stubs drift

When the real `LedgerLensContractClient` changes — a new required field in a
response, a renamed method, a changed exception type — `StubContractClient`
can silently diverge.  A diverged stub means tests pass locally but fail
against the real contract (or vice versa), hiding integration bugs until they
surface in production.

### Detecting drift automatically

A nightly CI job (`.github/workflows/stub_drift_check.yml`) runs
`StubDriftDetector.compare_scenarios()` against the canonical set of test
scenarios defined in `run_contract_test_scenarios()`.  If it fails:

1. Open the failing workflow run and read the diff output — it lists every
   scenario and field that diverged.
2. Check the `integrations/contract_client.py` Git log for recent changes to
   `submit_score`, `get_score`, or related methods.
3. Update `StubContractClient` to match the real method signature/return shape.
4. Update `run_contract_test_scenarios()` if a new scenario is needed.
5. Re-run `pytest tests/test_stub_drift.py` locally to confirm alignment.

### Updating the stub step-by-step

When you change `LedgerLensContractClient`:

1. **Update `StubContractClient`** — mirror the same method signature change
   in `integrations/offline_stubs.py`.  The stub must behave identically for
   the happy path (same return keys, same exception types on error paths).

2. **Update `run_contract_test_scenarios()`** — if your change adds a new
   scenario (e.g., a new method or a new required argument), add a
   corresponding test scenario to `run_contract_test_scenarios()` in
   `offline_stubs.py`.

3. **Run the drift detection test locally:**
   ```bash
   pytest tests/test_stub_drift.py -v
   ```

4. **Review `tests/test_offline_stubs.py`** — add a test for any new stub
   behaviour (happy path + error path).

5. **Include both files in your PR** — changes to the real client and its stub
   should travel together in the same commit.

### Running the drift check manually

```bash
python -m pytest tests/test_stub_drift.py -v

# Or run the scenario comparison directly:
python - <<'EOF'
from integrations.offline_stubs import StubContractClient, StubDriftDetector, run_contract_test_scenarios

stub_results = run_contract_test_scenarios(client=StubContractClient())
# Replace the second argument with a real client call in a live environment:
real_results = run_contract_test_scenarios(client=StubContractClient())

StubDriftDetector().compare_scenarios(stub_results, real_results)
print("No drift detected.")
EOF
```

### Adding a new scenario

To add a new test scenario to the drift-check baseline:

1. Add the scenario function inside `run_contract_test_scenarios()` following
   the existing pattern.
2. Add a corresponding test in `tests/test_stub_drift.py`.
3. Verify that `StubContractClient` correctly handles the new scenario.

The `StubDriftDetector` compares scenarios structurally — it checks whether
both sides agree on `ok`, `error_type`, and `result_keys`.  It does not
compare field *values* (those are covered by `tests/test_offline_stubs.py`).
