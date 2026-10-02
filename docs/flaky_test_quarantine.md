# Flaky-test Detection and Quarantine — Issue #966

Flaky tests erode trust in CI by randomly failing on unrelated PRs.  This
document describes the detection workflow, the quarantine mechanism, and the
process for resolving quarantined tests.

## What is a Flaky Test?

A **flaky test** passes on some runs and fails on others with no code change.
Common causes: timing dependencies, global mutable state, network calls, random
seeds, file-system ordering.

## Detection Workflow

The `.github/workflows/flaky-tests.yml` workflow runs the test suite **twice
in parallel** on every push and PR.  After both runs complete, it compares the
two JUnit XML result files with `scripts/detect_flaky_tests.py`.

A test is flagged as **flaky** if it:
- **Passed** in run 1 and **failed** in run 2, or
- **Failed** in run 1 and **passed** in run 2.

Consistently failing tests are genuine failures, not flaky tests.

## Quarantine Mechanism

### Marking a test as quarantined

```python
import pytest

@pytest.mark.quarantine
def test_my_flaky_test():
    # This test will still run but won't block merges.
    ...
```

### What quarantine does

| Behaviour | Normal test | Quarantined test |
|---|---|---|
| Runs in CI | ✅ | ✅ (in dedicated job) |
| Failure blocks merges | ✅ | ❌ (non-blocking) |
| Appears in report | ✅ | ✅ |
| Tracked in registry | — | ✅ |
| Surfaces in monthly audit | — | ✅ (if >30 days) |

Quarantined tests run in the `run-quarantined` CI job.  That job always exits
with code 0 so PR authors can merge even if a quarantined test fails.  The
failure is visible in the GitHub Actions job summary.

### Running quarantined tests locally

```bash
# Run only quarantined tests
pytest -m quarantine -v

# Run the full suite excluding quarantined (mirrors normal CI)
pytest -m "not integration and not slow and not quarantine"

# View quarantine report
python scripts/detect_flaky_tests.py \
    --audit-only \
    --registry reports/flaky/quarantine_registry.json
```

## Quarantine Registry

The quarantine registry (`reports/flaky/quarantine_registry.json`) records
every test that has been detected as flaky, together with:

- `quarantined_since` — ISO-8601 timestamp when first detected
- `flaky_run_count` — number of times flakiness was confirmed
- `last_seen` — most recent detection timestamp

The registry is uploaded as a CI artifact (`quarantine-registry`) with a
90-day retention window and summarised in the Actions job summary after every
workflow run.

## Monthly Staleness Audit

A scheduled job (`quarantine-audit`, runs on the 1st of every month at 01:00
UTC) surfaces all quarantined tests older than **30 days**.  These require
attention: either fix the underlying flakiness or explicitly document why the
test remains quarantined.

```
[40 days]  tests/test_streaming_pipeline.py::test_ws_reconnect  (flaky count: 3)
[35 days]  tests/test_kafka_worker.py::test_offset_commit_race  (flaky count: 2)
```

## Resolution Process

1. **Investigate** — run the flagged test in a loop locally:
   ```bash
   for i in $(seq 1 20); do pytest tests/test_my_module.py::test_my_test -x; done
   ```
2. **Fix** — address the root cause (add mock, fix global state, set seed, etc.).
3. **Remove the marker** — once fixed, remove `@pytest.mark.quarantine`.
4. **Verify** — confirm the test passes consistently over 5+ runs.
5. **Remove from registry** — delete the entry from
   `reports/flaky/quarantine_registry.json` or let the monthly audit confirm
   it is no longer appearing.

## Detection Script

```bash
# Compare two JUnit XML runs and update the registry
python scripts/detect_flaky_tests.py \
    --run1 reports/flaky/run1.xml \
    --run2 reports/flaky/run2.xml \
    --registry reports/flaky/quarantine_registry.json \
    --output reports/flaky/flaky_report.json

# Monthly audit (show stale entries)
python scripts/detect_flaky_tests.py \
    --audit-only \
    --registry reports/flaky/quarantine_registry.json \
    --stale-days 30
```

## Intentional Flaky Fixture

`tests/test_flaky_quarantine.py` contains an intentionally-flaky test
(`test_intentionally_flaky_for_detection_verification`) to verify that the
detection mechanism works.  It is marked `@pytest.mark.quarantine` so it never
blocks merges.  The CI workflow sets `LEDGERLENS_TEST_SIMULATE_FLAKY=1` in run-2
to trigger a failure, then compares the two results and confirms detection.
