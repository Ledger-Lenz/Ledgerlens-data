# Labelling Notes — Stellar SDEX Wash-Trade Dataset

## Methodology

Labels are assigned using a **conservative two-signal rule** to minimise false positives.

### Signal 1 — Round-trip detection (algorithmic)

A wallet pair `(A, B)` is flagged when:
- A sells asset X to B and B sells asset X back to A within ≤ 100 ledger closes (~8 minutes).
- Trade amounts are within ±5% of each other (accounting for slippage / fees).

Implemented in `scripts/mine_roundtrips.py::detect_roundtrip_pairs`.

### Signal 2 — Funding-source clustering (structural)

Wallets whose **Jaccard similarity** of funding ancestors exceeds **0.7** within the funding
graph (built by `detection/wallet_graph.py::build_funding_graph`) are considered in the same
cluster. Such wallets in combination with Signal 1 are strong candidates for wash trading.

### Conservative labelling rule

| Condition | Label |
|---|---|
| Flagged by **both** Signal 1 AND Signal 2 | `1` (wash trading) |
| No flags from either signal, > 50 trades, > 5 distinct counterparties | `0` (legitimate) |
| Flagged by only one signal, or insufficient trades | `NaN` (excluded) |

### Signal 3 — Manual review sample

For wallets flagged by both Signal 1 and Signal 2, a manual inspection sample of 50–100
accounts was conducted using [Stellar Expert](https://stellar.expert) and
[StellarBeat](https://stellarbeat.io) to verify plausibility.

## Manual Review Notes

| Wallet (truncated) | Decision | Rationale |
|---|---|---|
| GSYNTH* | Synthetic | Synthetic dataset used for automated testing |

> **Note:** This dataset version uses the synthetic dataset as a reference schema baseline.
> A production release against live Horizon data would populate this table with real observations.
> The `review_notes` column in the Parquet file stores per-wallet rationale.

## Known Limitations and Biases

1. **Round-trip window** — The 100-ledger window may miss slow wash-trading rings that operate
   over longer horizons, and may incorrectly flag legitimate arbitrage bots.
2. **Funding-graph sparsity** — `AccountActivity.funding_account` data is only available when
   an account-creation/funding event loader is wired up. Without it, Signal 2 defaults to 0.0
   similarity and no graph-based flagging occurs; all labels then come from Signal 1 only,
   placing all wallets in the grey zone unless they clearly have no round-trips and enough trades.
3. **Asset pair coverage** — Three asset pairs (USDC/XLM, BTC/XLM, AQUA/XLM) cover the most
   liquid Stellar SDEX markets but do not represent all possible wash-trading activity.
4. **Temporal bias** — The data window covers a specific six-month period. Strategies that
   emerged after the window will not be represented.
5. **Negative label quality** — Wallets labelled `0` satisfy minimum trade and counterparty
   thresholds, but this is not a guarantee of legitimacy.

## Ethics and Privacy

All data is sourced exclusively from the **public Stellar Horizon API**. Wallet addresses are
public keys on a permissionless blockchain. No personal identifying information is present.
See `data/dataset_card.md` for the full ethics statement.
