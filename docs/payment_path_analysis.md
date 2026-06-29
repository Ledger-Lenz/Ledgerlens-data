# Payment Path Analysis for Multi-Hop Wash Trade Detection

## Overview

Sophisticated wash traders on the Stellar DEX route trades through multi-hop payment paths (using Stellar's `path_payment_strict_send` and `path_payment_strict_receive` operations) to obfuscate the connection between buyer and seller wallets. A direct trade between wallet A and wallet B is easy to detect; the same economic transaction routed through 3 intermediate wallets and 2 asset conversions appears as 6 separate trades distributed across multiple accounts.

This module reconstructs these multi-hop flows and attributes them to the originating wallets, enabling detection of coordinated wash trading patterns that would be invisible when examining individual trades in isolation.

## The Problem: Multi-Hop Obfuscation

### Direct Trade (Easily Detected)
```
Wallet A ←→ Wallet B
```
Single trade pair at SDEX order book. High concentration, obvious round-trip patterns.

### Multi-Hop Path Payment (Obfuscated)
```
Wallet A
  ↓ (send 100 USDC)
  → Intermediate 1
  ↓ (USDC → XLM)
  → Intermediate 2
  ↓ (XLM → USDT)
  → Wallet A
  (receive 88 USDT)
```

The same economic transaction now appears as:
- 3 payment operations across 3 wallets
- 2 asset conversions
- Path length: 2 intermediate hops

Each intermediate wallet may have legitimate trading patterns that mask the coordinated wash-trading arrangement at the top level.

## Architecture

### Payment Path Reconstruction

`ingestion/payment_path_analyzer.py` provides the core reconstruction logic:

```python
def reconstruct_path_flow(
    path_payment_op: dict,
    all_operations: pd.DataFrame | None = None,
) -> ReconstructedPathFlow | None
```

**Input**: A Stellar Horizon path payment operation dict with fields:
- `type`: `"path_payment_strict_send"` or `"path_payment_strict_receive"`
- `source_account`: Wallet initiating the payment
- `destination_account`: Wallet receiving the payment
- `asset_path`: Array of intermediate assets (length ≤ 6)
- `amount` or `amount_sent` / `destination_amount`: Trading amounts
- `transaction_id`, `created_at`: Metadata

**Output**: `ReconstructedPathFlow` with:
- `source_wallet`: Originating wallet
- `destination_wallet`: Final recipient
- `source_amount`: Exact amount sent from source
- `destination_amount`: Exact amount received by destination
- `hop_count`: Number of intermediate asset conversions
- `path_payment_ids`: Transaction IDs along the path
- `execution_time`: Timestamp
- `is_round_trip`: `True` if source == destination

### Round-Trip Detection

```python
def compute_path_payment_round_trip_frequency(
    wallet: str,
    path_flows: list[ReconstructedPathFlow],
    time_window_hours: int = 24,
) -> float
```

Measures the fraction of a wallet's effective volume (after path reconstruction) that completes round-trips within a specified time window.

**Formula**:
```
round_trip_frequency = volume_returning_to_wallet / total_outgoing_volume
```

**Interpretation**:
- `0.0`: No round-trips (volume flows out)
- `1.0`: All volume returns (closed-loop wash trading)
- `0.5`: Half the volume returns (partial wash trading)

### Graph Integration

Path flows are integrated into the wallet graph as edges with metadata:

```
graph.add_edge(
    source_wallet,
    destination_wallet,
    edge_type="payment_path",
    hop_count=2,
    source_amount=100.0,
    destination_amount=88.0,
    round_trip=True,
    timestamp="2024-01-01T12:00:00Z"
)
```

The GNN encoder can then use these edges to:
1. Identify multi-wallet coordination patterns
2. Measure network centrality for path-routing hubs
3. Detect communities of wallets participating in coordinated multi-hop schemes

## Feature: `path_payment_round_trip_frequency`

### Definition

**Feature Name**: `path_payment_round_trip_frequency`

**Type**: Float in range [0.0, 1.0]

**Description**: Fraction of a wallet's effective volume (after path reconstruction) that returns to the originating wallet within 24 hours.

### Computation

For a wallet W:

1. Extract all path payment flows where W is the source wallet
2. Sum the `source_amount` for all flows = `total_outgoing_volume`
3. Count flows where `destination_wallet == W` and `is_round_trip == True` = `round_trip_volume`
4. Return `round_trip_volume / total_outgoing_volume`

### Wash-Trading Signal

High values strongly indicate closed-loop wash trading:

| Value | Interpretation |
|---|---|
| **0.0** | No round-trips; wallet routes to different recipients |
| **0.1–0.3** | Some round-trips; mixed legitimate + suspicious activity |
| **0.7–1.0** | Dominant round-trip pattern; strong wash-trade indicator |

## Requirements & Constraints

### Functional Requirements

- **Path reconstruction** must handle both `path_payment_strict_send` and `path_payment_strict_receive` operation types
- **Intermediate hops** must be excluded from the wallet's direct trade count but included in its effective volume count
- **Reconstructed path flows** must be integrated into the wallet graph so the GNN can see true trading relationships
- **Round-trip detection** must use a configurable time window (default 24 hours)

### Security Requirements

- **Schema validation**: All path operations fetched from Horizon must be schema-validated before processing
- **Path length enforcement**: Paths with > 6 hops (Stellar's maximum) must be rejected as malformed
- **Account ID validation**: Account IDs must conform to Stellar's format (G + 55 base-32 characters)
- **Amount validation**: All amounts must be non-negative floats

### Performance Considerations

- Path reconstruction is O(1) per operation (flat asset path traversal)
- Round-trip frequency is O(n) where n = number of path flows for the wallet (typically < 100 per wallet)
- Graph integration is O(1) per edge addition
- No blocking network calls in the reconstruction path

## Testing

### Unit Tests: Path Reconstruction

```python
def test_3hop_round_trip_unit_test():
    """3-hop path payment from wallet A → B → C → A."""
    op = {
        "type": "path_payment_strict_send",
        "source_account": A,
        "destination_account": A,  # Round-trip
        "transaction_id": "tx-roundtrip",
        "amount": 100.0,
        "destination_amount": 88.0,
        "asset_path": [
            {"code": "USDC", "issuer": "..."},
            {"code": "XLM", "issuer": None},
            {"code": "USDT", "issuer": "..."}
        ],
    }
    flow = reconstruct_path_flow(op)
    
    assert flow["source_wallet"] == A
    assert flow["destination_wallet"] == A
    assert flow["is_round_trip"] is True
    assert flow["hop_count"] == 3
    assert flow["source_amount"] == 100.0
    assert flow["destination_amount"] == 88.0
```

### Unit Tests: Round-Trip Frequency

```python
def test_path_payment_round_trip_frequency_1_0():
    """Wallet with only path round-trips must have frequency 1.0."""
    wallet = A
    flows = [
        {
            "source_wallet": A,
            "destination_wallet": A,
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "is_round_trip": True,
            ...
        },
        {
            "source_wallet": A,
            "destination_wallet": A,
            "source_amount": 200.0,
            "destination_amount": 190.0,
            "is_round_trip": True,
            ...
        },
    ]
    freq = compute_path_payment_round_trip_frequency(wallet, flows)
    
    assert freq == 1.0
```

### Schema Validation Tests

```python
def test_strict_receive_variant():
    """Path payment strict_receive variant must be reconstructed correctly."""
    op = {
        "type": "path_payment_strict_receive",
        "source_account": A,
        "destination_account": B,
        "amount_sent": 105.0,  # Variable
        "amount": 100.0,  # Exact
        "asset_path": [
            {"code": "USDT", "issuer": "..."},
            {"code": "USDC", "issuer": "..."}
        ],
    }
    flow = reconstruct_path_flow(op)
    
    assert flow["source_amount"] == 105.0
    assert flow["destination_amount"] == 100.0
    assert flow["hop_count"] == 2
```

## Integration with Existing Features

### Wallet Graph Integration

Reconstructed path flows are added as a new edge type to the wallet funding + co-trade graph:

```python
# In detection/wallet_graph.py
def add_payment_path_edges(
    graph: nx.DiGraph,
    path_flows: list[ReconstructedPathFlow],
) -> None:
    """Add payment path edges to the wallet graph."""
    for flow in path_flows:
        graph.add_edge(
            flow["source_wallet"],
            flow["destination_wallet"],
            edge_type="payment_path",
            weight=flow["hop_count"],
            round_trip=flow["is_round_trip"],
            source_amount=flow["source_amount"],
            destination_amount=flow["destination_amount"],
        )
```

### Feature Engineering

The `path_payment_round_trip_frequency` feature is computed in `detection/feature_engineering.py` and included in `build_feature_vector()`:

```python
def compute_payment_path_features(
    wallet: str,
    path_flows: list[ReconstructedPathFlow] | None = None,
) -> dict:
    """Compute payment path analysis features."""
    if not path_flows:
        return {"path_payment_round_trip_frequency": 0.0}
    
    round_trip_freq = compute_path_payment_round_trip_frequency(wallet, path_flows)
    return {"path_payment_round_trip_frequency": float(round_trip_freq)}
```

### Streaming Integration

For real-time scoring, payment path operations are processed as they arrive:

```python
# In streaming/feature_buffer.py
def update_with_payment_path(self, path_payment_op: dict) -> None:
    """Add a payment path operation to the streaming buffer."""
    if not validate_path_schema(path_payment_op):
        return  # Ignore malformed operations
    
    flow = reconstruct_path_flow(path_payment_op)
    if flow is None:
        return  # Path exceeds Stellar's maximum length
    
    self.path_flows.append(flow)
    # Update round-trip features for affected wallets
```

## Future Enhancements

### 1. AMM Pool Detection

Distinguish intermediate AMM pools from wallet intermediaries:

```python
def is_amm_pool(account_id: str, amm_pools: set[str]) -> bool:
    """Check if an account is a known AMM pool."""
    return account_id in amm_pools

# Filter intermediate hops
def reconstruct_with_pool_awareness(
    path_payment_op: dict,
    amm_pools: set[str],
) -> ReconstructedPathFlow | None:
    """Reconstruct path, excluding AMM pools from intermediary tracking."""
    ...
```

### 2. Cross-Chain Bridge Detection

Identify cross-chain wrapped asset flows:

```python
BRIDGE_ISSUERS = {
    "USDC_WORMHOLE": "GA5Z...",
    "ETH_STELLAR": "GA7V...",
    ...
}

def detect_bridge_usage(asset_path: list[dict]) -> bool:
    """Check if path includes cross-chain bridged assets."""
    ...
```

### 3. Path Similarity Clustering

Group similar multi-hop patterns to identify coordinated wash-trading rings:

```python
def compute_path_fingerprint(flow: ReconstructedPathFlow) -> tuple:
    """Compute a fingerprint of the path topology."""
    return (
        flow["hop_count"],
        tuple(sorted([a["code"] for a in flow["asset_path"]])),
        flow["source_amount"] // 1000,  # Quantize amounts
    )

def cluster_similar_paths(flows: list[ReconstructedPathFlow]) -> dict[str, list]:
    """Cluster flows by fingerprint to find coordinated patterns."""
    ...
```

## References

- **Stellar Path Payments**: https://developers.stellar.org/docs/learn/path-payments
- **Wash Trading Detection**: Elliptic dataset (Weber et al., 2019) — anti-money laundering with GNNs
- **Multi-hop Obfuscation**: Inspection-L (Lo et al., 2023) — flow-level DEX fraud detection

## Contributor Guidance

**Area of Specialty**: Stellar SDK, blockchain analytics, Python, graph algorithms.

**How to Contribute**:

1. Comment on the issue describing your experience with:
   - Stellar path payments or equivalent multi-hop routing (e.g., Ethereum DEX aggregators)
   - Graph algorithms for detecting cycles or coordinated patterns
   - Real-time fraud detection systems

2. Propose enhancements such as:
   - How to handle path payments where intermediate wallets are AMM pools (not user wallets)
   - Cross-chain bridge detection for wrapped assets
   - Path similarity clustering for coordinated ring detection
   - Performance optimizations for real-time path ingestion

3. Ensure your PR includes:
   - The 3-hop round-trip unit test
   - The strict_receive variant test
   - Comprehensive schema validation tests
   - Integration tests with the wallet graph and feature matrix

**Code Review Checklist**:

- [ ] Path reconstruction handles both `path_payment_strict_send` and `path_payment_strict_receive`
- [ ] Paths exceeding 6 hops are rejected (Stellar maximum)
- [ ] Account IDs are validated before processing
- [ ] Round-trip detection is accurate for same-source/destination flows
- [ ] Feature values are in range [0.0, 1.0] for frequency metrics
- [ ] All tests pass, including edge cases (empty data, single flow, 100% round-trips)
- [ ] Integration with `build_feature_vector()` is seamless
- [ ] Documentation is updated with examples and interpretation guide

