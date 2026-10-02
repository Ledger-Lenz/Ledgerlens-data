# Asset Metadata: Staleness and Trust Tiers

`ingestion/asset_metadata_fetcher.py` supplies asset circulating supply, which
token-velocity features and forensic reports depend on. When the primary source
is down, it falls back to lower-trust data. It never presents that data as
current: every value comes back as an `AssetMetadataRecord` that says where it
came from and how old it is.

```python
from ingestion.asset_metadata_fetcher import get_asset_metadata

record = get_asset_metadata(
    "USDC", issuer,
    horizon_url=config.HORIZON_URL,
    redis_client=redis,
    alternate_horizon_url="https://horizon.example-mirror.org",  # optional
)
record.circulating_supply  # float | None
record.trust_tier          # TrustTier.PRIMARY / CACHE / ALTERNATE / UNAVAILABLE
record.fetched_at          # when the value was fetched from its source
record.age_seconds         # seconds since fetched_at
record.is_stale            # age >= 1 hour (the cache TTL)
record.is_degraded         # not fresh primary data: tier != primary, or stale
```

`get_asset_supply()` still returns only the number, for callers that don't need
provenance. Anything that shows supply-derived values to a person or a model
should use `get_asset_metadata()` and pass the record along.

---

## Fallback chain

| Step | Condition | Returned tier |
|---|---|---|
| 1 | Cached primary data younger than the TTL (1 hour) | `primary` |
| 2 | Otherwise fetch from the primary Horizon (`horizon_url`) | `primary` |
| 3 | Primary unavailable and cached primary data younger than 7 days | `cache` (always `is_stale=True`) |
| 4 | Primary unavailable, no usable cache, `alternate_horizon_url` answers | `alternate` |
| 5 | Nothing available | `unavailable` (`circulating_supply=None`) |

The primary counts as **unavailable** when it can't be reached or returns an
unusable response (network error, malformed JSON, non-numeric amount). A
primary that answers but has no positive supply for the asset is a valid
`primary` answer with `circulating_supply=None`, and does not trigger fallback.

Cache entries (Redis when a client is given, plus an in-process copy) store the
supply, `fetched_at` and `source`. They are kept for the whole 7-day stale
window, and freshness is judged from `fetched_at`. Only primary data is cached.
Alternate-source values are returned but never written back, so they can never
come back later labelled as `cache`.

---

## Trust tiers

| Tier | Meaning | How to treat it |
|---|---|---|
| `primary` | Fresh data from the primary Horizon source | Current |
| `cache` | Last known primary value, served while the primary is down; older than the TTL | Usable, but say it is stale and how old it is |
| `alternate` | From a secondary, lower-trust source | Usable, but say it did not come from the primary |
| `unavailable` | No value | Supply-derived features are `NaN` |

---

## Where it is surfaced

- **Forensic reports** (`ForensicReportGenerator.generate(..., asset_metadata=record)`):
  the report gets an `asset_metadata` field (supply plus all provenance
  fields), which is covered by `report_sha256`. The Markdown report gets an
  "Asset Metadata Provenance" section, with a warning when the data is
  degraded.
- **Feature pipeline** (`build_extended_feature_vector(..., asset_metadata=record)`):
  the record's supply feeds the token-velocity features, and
  `asset_metadata_trust_tier`, `asset_metadata_age_seconds` and
  `asset_metadata_is_stale` are added next to them.
- **Importer registry**: `AssetMetadataFetcherRegistry.get_asset_metadata()`.

Tests: `pytest tests/test_asset_metadata_fetcher.py -v`
