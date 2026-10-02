# Typed Deployment Mode Fixtures

`config/deployment_modes.py` (Issue #543) gives each supported deployment
mode — `local`, `testnet`, `production` — a typed, reusable, validated
fixture instead of hand-rolled `.env` files that drift out of sync with
what `Config.validate()` actually requires.

## Why

`Config` (config.py) is a flat, env-var-driven surface with 500+
attributes. Nothing previously enforced that a given deployment mode set a
*coherent* combination of them — e.g. that `production` never accidentally
ships with `HORIZON_DEV_MODE=True`, or that `testnet` sets an on-chain
contract ID when on-chain submission is required. Contributors discovered
missing/incoherent values only at runtime.

## Usage

```python
from config.deployment_modes import DeploymentMode, apply_deployment_mode

with apply_deployment_mode(DeploymentMode.TESTNET) as fixture:
    # Config now reflects the testnet fixture's overrides, and has already
    # been validated with Config.validate(require_onchain=fixture.require_onchain).
    run_pipeline()
# Every overridden attribute is restored to its prior value on exit.
```

In tests, use the equivalent pytest fixtures registered in
`tests/conftest.py`:

```python
def test_something(testnet_deployment_config):
    assert Config.STELLAR_NETWORK == "TESTNET"
```

## Adding a new mode

Add one `DeploymentModeFixture` entry to `DEPLOYMENT_MODE_FIXTURES` in
`config/deployment_modes.py`. Every consumer — tests, scripts,
`apply_deployment_mode` — picks it up automatically; there is nothing else
to wire up.

## Validation

`apply_deployment_mode(..., validate=True)` (the default) calls
`Config.validate()` with the fixture's `require_onchain` flag immediately
after applying overrides. A fixture that doesn't produce a valid
configuration raises `DeploymentModeValidationError` naming the mode and
the underlying `Config.validate()` failure — the diagnostic points
directly at which fixture is inconsistent and why, instead of surfacing as
an unrelated runtime error downstream.

## Local validation commands

```bash
pytest tests/test_deployment_modes.py -v
```

## Graceful-shutdown guarantee (streaming workers)

Kafka workers drain in-flight messages on SIGTERM before exiting: every consumed
message is either fully processed with its offset committed, or not committed at
all and redelivered. Keep `terminationGracePeriodSeconds` >
`KAFKA_DRAIN_TIMEOUT_SECONDS` (default 30s) during rolling deploys. See
`docs/stream_replay_runbook.md#graceful-shutdown--in-flight-draining-892`.
## Liveness vs readiness probes (Issue #902)

`streaming.health_check.start_health_server(port)` exposes:

- `GET /livez` (alias `/health`) — liveness: worker heartbeats only. No external
  dependencies, so a Kafka outage never causes a restart.
- `GET /readyz` — readiness: every dependency registered via
  `register_dependency()` (e.g. `kafka_broker_check`, `model_artifact_check`,
  a feature-store ping) is checked live with a `READINESS_CHECK_TIMEOUT_SECONDS`
  timeout (default 2s). Returns 503 while any dependency is down and flips back
  to 200 automatically when it recovers — no restart required.

```yaml
livenessProbe:
  httpGet: { path: /livez, port: 8080 }
  periodSeconds: 10
  failureThreshold: 6        # ~1 min of stalled heartbeats before restart
readinessProbe:
  httpGet: { path: /readyz, port: 8080 }
  periodSeconds: 5
  timeoutSeconds: 3          # > READINESS_CHECK_TIMEOUT_SECONDS
  failureThreshold: 2        # pull from traffic quickly
  successThreshold: 1        # rejoin as soon as dependencies recover
```
