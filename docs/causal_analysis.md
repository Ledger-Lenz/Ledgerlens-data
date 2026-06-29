# Causal Analysis of Wallet Features

`analysis/causal_discovery.py` runs the **PC algorithm** on the LedgerLens feature matrix to produce a Directed Acyclic Graph (DAG) identifying which features causally influence the risk score and which are downstream effects of shared confounders.

## Why Causal Discovery?

SHAP values measure correlation-based feature importance — not causation. A market-wide event (e.g., an XLM price spike) may simultaneously inflate many features (counterparty concentration, volume spikes, Benford deviation) without representing genuine wash trading. Causal discovery identifies such confounders so they can be controlled for or down-weighted.

## The PC Algorithm

The PC algorithm (Spirtes, Glymour, Scheines, 2000) recovers a Completed Partially Directed Acyclic Graph (CPDAG) from observational data by:

1. **Skeleton search** — Tests every pair of variables for conditional independence, progressively conditioning on larger sets. Pairs that are not conditionally independent for any conditioning set are connected by an undirected edge.
2. **V-structure orientation** — Identifies colliders (X → Z ← Y where X ⊥ Y but X ⊥ Y | Z is not true) and orients those edges.
3. **Propagation** — Applies Meek rules to orient remaining edges without introducing new v-structures or cycles.

The output is a CPDAG where some edges may remain undirected (equivalence class).

## Assumptions

| Assumption | Description | Violation risk in wash-trade data |
|---|---|---|
| **Causal Markov condition** | Each variable is independent of its non-descendants given its parents in the true DAG. | Low — standard for tabular feature data. |
| **Faithfulness** | Every conditional independence in the data corresponds to a d-separation in the true DAG. | Medium — could be violated if two causal paths cancel each other out (e.g., a wash trader's volume spike is simultaneously masked by a bot pausing activity). |
| **Causal sufficiency** | No unmeasured common causes exist. | Possible — latent coordinating infrastructure (shared bots, off-chain coordination) is not captured as a feature. |

## Independence Tests

| Feature type | Test used | Notes |
|---|---|---|
| Continuous (majority) | Fisher's Z | Assumes Gaussian residuals; appropriate for normalised financial features. |
| Discrete (>90% columns) | G² (likelihood ratio) | Appropriate when most features are categorical or low-cardinality integers. |

## Output

The discovered DAG is persisted as:

- `analysis/feature_dag.dot` — GraphViz DOT format for visualisation.
- `analysis/feature_dag.json` — JSON adjacency list `{node: [children]}`.

## CLI Usage

```bash
python -m scripts.run_causal_discovery \
    --data data/synthetic_dataset.parquet \
    --significance-level 0.05 \
    --max-cond-set-size 3 \
    --timeout 600
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Path to feature Parquet file (must be inside `data/`) |
| `--significance-level` | `0.05` | α for independence tests |
| `--max-cond-set-size` | `3` | Max conditioning set size |
| `--timeout` | `600` | Wall-clock limit in seconds; partial DAG returned on timeout |

## Interpreting the Output DAG

An edge `A → B` means the PC algorithm found evidence that A is a direct cause of B given the observed features. Use this to:

1. **Identify confounders** — nodes with many outgoing edges to both features and the label are likely confounders (e.g., market volatility).
2. **Select features for model input** — prefer direct causes of the label over features that are merely correlated via a shared confounder.
3. **Explain score changes** — an auditor asking "why did the score jump?" can trace the causal path rather than relying purely on SHAP correlation.

Undirected edges in the CPDAG indicate membership in a Markov equivalence class — the algorithm cannot distinguish the direction from observational data alone.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CAUSAL_DISCOVERY_TIMEOUT_SECONDS` | `600` | Hard timeout; partial DAG returned |
| `CAUSAL_SIGNIFICANCE_LEVEL` | `0.05` | Default α |
| `CAUSAL_MAX_COND_SET_SIZE` | `3` | Default max conditioning set depth |
