# WebSocket Streaming Policies

## Slow-consumer policy (#893)

Each connection has an outbound queue bounded by `WS_CLIENT_QUEUE_DEPTH`.
Producers never block, so a slow client cannot delay delivery to others.

| `WS_SLOW_CONSUMER_POLICY` | Behavior when the queue is full |
|---|---|
| `drop_oldest` (default) | Oldest queued message is evicted and a `{"type": "dropped"}` notice is queued. If `WS_SLOW_CONSUMER_MAX_DROPS > 0`, the client is disconnected once drops exceed it. |
| `disconnect` | Connection is closed immediately with close code `4008 slow_consumer`. |

Metrics: `ws_client_queue_depth` (histogram, observed per enqueue),
`ws_slow_consumer_events_total{policy,action}` (`action` = `drop_oldest` | `disconnect`),
`ws_messages_dropped_total`.

## Token refresh (#894)

Send before the current token's `exp`:

```json
{"type": "refresh_token", "token": "<new JWT>"}
```

The new token must be valid and have the same `sub`. On success the server
replies `{"type": "token_refreshed", "exp": <unix ts>}` and permissions are
updated from the new token's scope. The initial authentication flow is unchanged.

| Close code | Reason | Meaning |
|---|---|---|
| `4001` | `token_expired` | No valid refresh before `exp + WS_TOKEN_EXPIRY_GRACE_SECONDS`. |
| `4003` | `token_refresh_rejected` | Refresh token invalid, subject mismatch, or sent after expiry. |
| `4008` | `slow_consumer` | Slow-consumer policy disconnect (see above). |
| `1008` | varies | Initial auth failure, capacity, or abuse block. |

## Reputation-based adaptive rate limiting (#895)

Each client ID has a reputation score in `[-1, 1]` (new clients start at `0`).
Every compliant request adds `0.01`; any rate-limit or wallet-targeting
violation sets it to `-1`. The score decays toward `0` with half-life
`WS_ABUSE_REPUTATION_HALF_LIFE_SECONDS`, so good standing expires and penalties
recover after a cooldown. Reputation survives reconnects.

Effective limit = `WS_ABUSE_MAX_REQUESTS_PER_MINUTE × multiplier`:

- score `0` → `WS_ABUSE_NEW_CLIENT_MULTIPLIER` (default `1.0`)
- score `1` → `WS_ABUSE_MAX_REPUTATION_MULTIPLIER` (default `3.0`)
- score `-1` → half the new-client multiplier

Wallet-targeting detection is never relaxed by reputation, and the rate ceiling
is capped at the max multiplier, so abuse is still throttled regardless of history.
