# Cross-Chain Identity Resolution & Bridge Mechanisms

Covers confidence-scored, evidence-linked identity edges (Issue #879) and
mechanism-aware bridge detection (Issue #880). All code lives in
`detection/cross_chain/`; the labelled benchmarks live in
`benchmarks/cross_chain.py`.

## Confidence-scored identity edges (Issue #879)

### Evidence model

Every row in `cross_chain_edges` is one piece of **evidence** for an address
pair. `link_type` is the evidence type, `confidence` is the detector's own
strength in `[0, 1]`, and `metadata_json` holds the detector payload (tx ids,
memo, amounts, Pearson `r`, bridge mechanism, ...) for forensic traceability.

| Source | `link_type` | Strength | Reliability |
|--------|-------------|----------|-------------|
| `BridgeDetector.detect_bridge_links` | `bridge` / `wormhole_bridge` | `1.0` (memo-encoded destination) | 1.0 |
| Shared deposit address | `shared_deposit` | caller-supplied | 0.9 |
| Generic behavioral match | `behavioral` | caller-supplied | 0.8 |
| `BehavioralMatcher.match_timing_correlation` | `timing_correlation` | Pearson `r` | 0.7 |
| `BehavioralMatcher.match_amount_fingerprints` | `amount_fingerprint` | `1 - relative amount diff` | 0.55 |

Unknown evidence types use a reliability of 0.8. The priors are defined in
`EVIDENCE_RELIABILITY` in `detection/cross_chain/confidence.py`.

Each item contributes `p_i = strength_i × reliability_i`. A pair's items are
combined with a noisy-OR:

```
confidence = 1 - Π (1 - p_i)
```

Independent corroborating signals raise the confidence. A single weak or
commonly coincidental signal stays low: a lone amount match scores at most
0.55, while an amount match plus an `r = 0.9` timing correlation scores about 0.84.

### Ingesting and retrieving evidence

```python
from detection.cross_chain import IdentityGraph, BridgeDetector, BehavioralMatcher
from detection.cross_chain.resolver import get_link_evidence

graph = IdentityGraph()
graph.add_links(BridgeDetector(anchors).detect_bridge_links(txs), "bridge")
graph.add_links(BehavioralMatcher.match_timing_correlation(s_txs, e_txs), "timing_correlation")

graph.get_link_confidence("GABC...", "0xdef...")   # combined score
get_link_evidence("GABC...", "0xdef...")
# {"confidence": 0.93, "evidence": [{"evidence_type": "bridge", "strength": 1.0,
#   "reliability": 1.0, "probability": 1.0, "metadata": {"tx_id": ..., "memo": ...}}, ...]}
```

`get_connected_component(address, min_confidence=...)` only traverses pairs at or
above the threshold, and every returned node carries the `confidence` of the
link it was reached through.

### Threshold: `CROSS_CHAIN_MIN_CONFIDENCE` (default `0.65`)

`resolve_risk_scores()` is the entry point that `propagate_risk_scores()` and
`propagation_attribution()` use. It excludes links below `config.CROSS_CHAIN_MIN_CONFIDENCE`,
so those links carry no risk across chains. `resolve()` defaults to
`min_confidence=0.0` so investigators can still see every candidate link.

Results on the labelled identity-pair benchmark (200 true pairs and 400
coincidental pairs, seed 879; run `python -m benchmarks.cross_chain`):

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.50 | 0.333 | 1.000 | 0.500 |
| 0.55 | 0.429 | 0.825 | 0.564 |
| 0.60 | 0.612 | 0.740 | 0.670 |
| **0.65** | **0.733** | **0.645** | **0.686** |
| 0.70 | 0.715 | 0.590 | 0.647 |
| 0.80 | 0.715 | 0.590 | 0.647 |
| 0.90 | 1.000 | 0.360 | 0.529 |

**Rationale:** 0.65 maximises F1. It is also the lowest threshold that
rejects a lone amount fingerprint (max 0.55) and a lone weak timing
correlation (`r < 0.93`), the two most common sources of coincidental links.
A confirmed bridge memo (1.0) or a strong behavioral match (strength ≥ 0.82, i.e. p ≥ 0.65)
still passes on its own. Because risk propagation *raises* a wallet's score
from its linked counterparts, a false link does more damage than a missed
one, so the default leans towards precision. Set it to `0.9` for
bridge-only propagation, or lower it to around `0.6` for investigations
where recall matters more.

## Mechanism-aware bridge detection (Issue #880)

`detection/cross_chain/bridge_mechanisms.py` first classifies a bridge
transaction by mechanism, then applies that mechanism's heuristics:

| Mechanism | Classification signal | Features | Detection rule |
|-----------|----------------------|----------|----------------|
| `lock_and_mint` | share of `lock`/`burn`/`mint`/`unlock` events | `locked_amount`, `minted_amount`, `mint_burn_ratio`, `cross_chain` | lock/burn ↔ mint/unlock on different chains, 1:1 within 0.1% (confidence decays to 0 at 0.3%) |
| `liquidity_pool` | share of `pool_deposit`/`pool_withdraw`/`swap*` events, plus `pool_balances` | `amount_in`, `amount_out`, `fee_ratio`, `source_pool_delta`, `destination_pool_delta`, `pool_delta_error` | deposit and payout on disjoint chains, fee in `[0, 1%]`, pool balance deltas consistent with event amounts |

`BridgeDetector.classify_transactions(txs)` returns, for each transaction,
the `mechanism`, `classification_score`, `features`, `confidence` and
`is_bridge`. `detect_bridge_links` adds `mechanism`, `mechanism_confidence`
and `mechanism_features` to any memo link whose transaction carries
`events`, so the mechanism is stored as edge evidence too.

### Adding a mechanism

```python
from detection.cross_chain.bridge_mechanisms import (
    BridgeMechanismHandler, register_bridge_mechanism,
)

@register_bridge_mechanism
class MessagePassingHandler(BridgeMechanismHandler):
    name = "message_passing"
    def classification_score(self, tx): ...
    def extract_features(self, tx): ...
    def detect(self, features): ...
```

The handler with the highest classification score claims the transaction.
Scores below 0.5 fall back to `unknown`, which is never flagged.

### Benchmark

The labelled set has 600 transactions. For each mechanism there are 150
genuine bridges and 150 look-alikes:
- **Lock-and-mint look-alikes:** same-chain wrapping, and unrelated mints that
  coincide with a lock (0.2–0.9% apart).
- **Liquidity-pool look-alikes:** same-chain DEX swaps, and spoofed pool events
  whose balances disagree with the event amounts.

The baseline is the previous unified approach, which matches the first and
last event amounts within 1% and ignores the mechanism.

| | Lock-and-mint P / R | Liquidity-pool P / R |
|---|---|---|
| Unified baseline | 0.500 / 1.000 | 0.500 / 1.000 |
| Mechanism-specific | **1.000 / 1.000** | **1.000 / 1.000** |

Mechanism classification accuracy is **1.000** (600 / 600).

Both benchmark sets are synthetic. They encode the structural signatures
described above, so they show that the heuristics separate those
signatures; they are not a production error rate. Re-run the benchmark
against labelled on-chain data before changing thresholds.
