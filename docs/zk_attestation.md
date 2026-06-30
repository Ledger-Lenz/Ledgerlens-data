# zk Attestation Design

This repository now supports a V1 hash-commitment flow for on-chain score submissions.

## V1

The attestor computes three public values from the wallet submission:

1. `trade_data_hash` from a canonical serialization of the public trade set.
2. `model_version_hash` from the model parameters committed at deployment time.
3. `commitment = SHA-256(wallet, trade_data_hash, model_version_hash, score)`.

The contract client can submit the usual `RiskScore` fields plus the commitment metadata. The raw `submit_score` method remains available as a fallback when attestation is not required.

## Reproducibility

The commitment is deterministic because trade rows and columns are canonicalized before hashing. Re-running the same score computation over the same trade set yields the same receipt.

## V2 zkVM path

Future Risc Zero integration should keep the same public receipt shape, but replace the hash-commitment builder with a guest program that consumes:

- `wallet`
- `trade_data_hash`
- `model_version_hash`
- `score`

The guest should emit the public receipt values above and a proof artifact that Soroban can verify before accepting the attested score.


---

## Soroban Event Listener & Audit Trail

### Overview

`integrations/soroban_event_listener.py` subscribes to events emitted by the
`ledgerlens-score` Soroban contract and builds an **on-chain audit trail** of
how risk scores are consumed by third-party applications (wallets, exchange
UIs, dApps). This enables compliance reporting on when flagged wallets were
queried and whether consumers acted on current or stale data.

### Contract event schema

The `ledgerlens-score` contract emits three event types. Each event follows
the standard Soroban event structure: a **topic** vector (first element is the
event name symbol, subsequent elements are key parameters) and a **value** map.

#### `score_read`

Emitted whenever a consumer calls `get_score(wallet, asset_pair)`.

| Field | Source | Type | Description |
|---|---|---|---|
| `event_type` | topic[0] | symbol | `"score_read"` |
| `wallet` | topic[1] | address | Wallet whose score was read |
| `consumer` | value.consumer | address | Account that called `get_score` |
| `score` | value.score | u32 | Score value returned to the consumer |
| `asset_pair` | value.asset_pair | string | Asset pair identifier |

#### `score_updated`

Emitted whenever `submit_score(...)` writes a new score for a wallet.

| Field | Source | Type | Description |
|---|---|---|---|
| `event_type` | topic[0] | symbol | `"score_updated"` |
| `wallet` | topic[1] | address | Wallet whose score changed |
| `score` | value.score | u32 | New score value |
| `asset_pair` | value.asset_pair | string | Asset pair identifier |

#### `threshold_updated`

Emitted when the on-chain alert threshold is changed by an admin.

| Field | Source | Type | Description |
|---|---|---|---|
| `event_type` | topic[0] | symbol | `"threshold_updated"` |
| `old_threshold` | value.old_threshold | u32 | Previous threshold value |
| `new_threshold` | value.new_threshold | u32 | New threshold value |

### DB schema (`contract_event` table)

```
contract_event
├── id                    INTEGER  PRIMARY KEY
├── event_type            VARCHAR(32)   NOT NULL  -- score_read | score_updated | threshold_updated
├── ledger_sequence       BIGINT        NOT NULL
├── event_timestamp       DATETIME      NOT NULL
├── wallet_id_hash        VARCHAR(64)   NOT NULL  -- HMAC-SHA256 of raw wallet address
├── score                 INTEGER       NULLABLE
├── consumer_address_hash VARCHAR(64)   NULLABLE  -- HMAC-SHA256 of raw consumer address
├── asset_pair            VARCHAR(128)  NULLABLE
├── old_threshold         INTEGER       NULLABLE
├── new_threshold         INTEGER       NULLABLE
├── raw_payload           TEXT          NULLABLE  -- full JSON for forward-compat
└── ingested_at           DATETIME      NOT NULL
```

A companion `event_watermark` table stores the last processed ledger per
contract, enabling crash-safe resume without re-processing old events.

### Privacy: address hashing

Raw Stellar G-addresses are **never stored** in the DB or written to log
lines. Both `wallet_id` and `consumer_address` are replaced with
HMAC-SHA256 digests keyed by `EVENT_HMAC_SECRET` (set via environment
variable). This allows joining events for the same wallet across audit
queries without exposing the raw address to log aggregation systems.

### Stale score consumption alert

When a `score_read` event arrives for a wallet whose score has since changed
by more than `STALE_SCORE_ALERT_THRESHOLD` (default **20 points**), an alert
is fired via `streaming.alert_dispatcher.AlertDispatcher`. The alert payload
includes:

```json
{
  "score": 82,
  "consumed_score": 55,
  "delta": 27,
  "stale_consumption": true
}
```

This signals that a consuming application read an outdated score, which may
affect a compliance or risk-management decision. The threshold is configurable:

```bash
STALE_SCORE_ALERT_THRESHOLD=15 python -m integrations.soroban_event_listener
```

### Horizon vs Soroban RPC delivery

Two event backends are supported, selected via `EVENT_BACKEND`:

| Backend | Source | Notes |
|---|---|---|
| `rpc` (default) | Soroban RPC `getEvents` | Lowest latency; requires Soroban RPC endpoint |
| `horizon` | Horizon `/accounts/{contract_id}/effects` | Broader availability; wraps Soroban events under `data.topic` |

During Stellar protocol upgrades, Horizon and Soroban RPC may temporarily
diverge (Horizon lags by a few ledgers while Horizon catches up to the new
XDR encoding). The listener handles this transparently: the watermark is only
advanced after successful parsing and persistence, so if a backend becomes
temporarily unavailable the next poll retries from the last committed ledger
without any data loss.

To switch backends at runtime, set `EVENT_BACKEND=horizon` and restart.

### Running the listener

```bash
# Start with Soroban RPC (default):
python -m integrations.soroban_event_listener

# Use Horizon fallback:
EVENT_BACKEND=horizon python -m integrations.soroban_event_listener
```

Or embed it in a pipeline:

```python
from integrations.soroban_event_listener import SorobanEventListener
from streaming.alert_dispatcher import AlertDispatcher

dispatcher = AlertDispatcher(channel="stdout")
listener = SorobanEventListener(
    contract_id="C...",
    dispatcher=dispatcher,
    current_score_fn=lambda wallet_hash: score_store.get_by_hash(wallet_hash),
)
listener.start_background()
```

### Synthetic event fixture

`tests/fixtures/soroban_events.json` contains realistic synthetic event
payloads covering all three event types plus the Horizon effects wrapper
format. These fixtures are used by `tests/test_soroban_event_listener.py`
and can be used during local development to test parsing without a live
Soroban network:

```python
import json
from integrations.soroban_event_listener import parse_contract_event

with open("tests/fixtures/soroban_events.json") as f:
    fixtures = json.load(f)

event = parse_contract_event(fixtures["score_read"])
print(event.event_type, event.score, event.wallet_id_hash)
```
