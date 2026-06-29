# LedgerLens REST API

The LedgerLens REST API exposes wallet risk scores over HTTP. It is served by `api/app.py` (FastAPI).

## Base URL

```
http://localhost:8000
```

Serve locally with:

```bash
uvicorn api.app:app --reload
```

The OpenAPI spec is available at `/openapi.json` and Swagger UI at `/docs`.

## Authentication

Every request (except `/v1/health`) requires an `X-API-Key` header:

```
X-API-Key: <your-api-key>
```

API keys are stored as **bcrypt hashes** in `config.API_KEYS` (env var `API_KEYS`, comma-separated). Plaintext keys are never logged.

Generate a hashed key:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'mysecret', bcrypt.gensalt()).decode())"
```

A missing or invalid key returns `401 Unauthorized`.

## Rate Limiting

Each API key is rate-limited to `API_RATE_LIMIT_RPM` requests per minute (default: 60). Exceeding the limit returns `429 Too Many Requests`.

## Endpoints

### `GET /v1/health`

Liveness and readiness check. No authentication required.

**Response:**

```json
{
  "status": "ok",
  "db": "ok",
  "model": "ok"
}
```

`status` is `"degraded"` when the database is unavailable.

---

### `GET /v1/wallets/{address}/scores`

Paginated risk score history for a wallet. Results are in **descending timestamp order**. Pagination is cursor-based (keyed on `score_id`) for consistent results under concurrent writes.

**Path parameters:**

| Parameter | Description |
|---|---|
| `address` | Stellar account ID (must match `G[A-Z2-7]{55}`) |

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start_ts` | integer | Unix timestamp lower bound |
| `end_ts` | integer | Unix timestamp upper bound |
| `asset_pair` | string | Filter by asset pair (e.g. `USDC:GA5Z.../XLM:native`) |
| `min_score` | integer (0–100) | Exclude scores below this threshold |
| `cursor` | integer | `score_id` from the previous page's `next_cursor` |
| `limit` | integer (1–200) | Page size, default 50 |

**Response:**

```json
{
  "items": [
    {
      "score_id": 42,
      "wallet": "G...",
      "asset_pair": "USDC:.../XLM:native",
      "score": 85,
      "benford_flag": true,
      "ml_flag": true,
      "confidence": 91,
      "propagated_risk": null,
      "ring_id": null,
      "updated_at": "2026-06-27T20:00:00+00:00"
    }
  ],
  "next_cursor": 41,
  "total": 1
}
```

Pass `?cursor=<next_cursor>` to fetch the next page. `next_cursor` is `null` on the last page.

**Example:**

```bash
curl -H "X-API-Key: mysecret" \
  "http://localhost:8000/v1/wallets/G.../scores?min_score=70&limit=20"
```

---

### `GET /v1/wallets/{address}/latest`

Latest risk score and top-3 contributing SHAP features.

**Query parameters:**

| Parameter | Description |
|---|---|
| `asset_pair` | Optional; narrows to a specific trading pair |

**Response:**

```json
{
  "wallet": "G...",
  "asset_pair": "USDC:.../XLM:native",
  "score": 85,
  "benford_flag": true,
  "ml_flag": true,
  "confidence": 91,
  "top_features": [
    {"feature": "benford_chi_square_1h", "shap_value": 0.34},
    {"feature": "counterparty_concentration", "shap_value": 0.28},
    {"feature": "round_trip_frequency", "shap_value": 0.21}
  ]
}
```

Returns `404` when no score exists for the wallet.

**Example:**

```bash
curl -H "X-API-Key: mysecret" \
  "http://localhost:8000/v1/wallets/G.../latest"
```

## Versioning

The API uses URL versioning (`/v1/`). New optional fields may be added to responses without a version bump. Removing or renaming fields requires a new major version (`/v2/`). Clients should ignore unknown JSON keys for forward compatibility.
