# Causal Inference for LedgerLens

This document describes the causal attribution layer added to LedgerLens on top of the existing feature pipeline and ensemble scorer.

## Why Causal Attribution

SHAP explains which features contributed to a score. Causal attribution asks a more operational question: which trades, counterparties, or funding paths would need to change for the wallet to fall below the risk threshold?

That distinction matters in investigations. A wallet can be high-risk because a feature is large, but the analyst still needs to know which observable trades or upstream wallets are driving that feature.

## Structural Causal Model

LedgerLens builds a lightweight SCM from the existing feature vector:

- Nodes are features.
- Edges represent simple structural dependencies between features computed from the same trade set.
- Interventions propagate through the graph so downstream features are recomputed rather than blindly overwritten.

The SCM is intentionally small and deterministic. It is not a symbolic causal discovery engine; it is a forensic explanation layer built around known feature relationships.

## Counterfactual Scoring

`CounterfactualAttributor.counterfactual_score()` removes selected trades, rebuilds the wallet features, and rescales the wallet with the same trained ensemble used in production.

This is different from feature substitution. Removing trades changes the trade-derived features, the Benford metrics, and the graph-derived signals together.

## Greedy Exoneration Search

`minimal_exonerating_set()` uses greedy backward elimination:

1. Score the wallet with the current trade set.
2. Remove the trade that lowers the score the most.
3. Repeat until the score falls below the threshold or the search limit is reached.

If no subset of up to 20 trades can move the wallet below threshold, the result is `None`. That indicates the signal is structural or graph-driven rather than explained by a small trade subset.

## Root Cause Wallets

`root_cause_wallet()` evaluates each counterparty wallet and measures the score reduction if its shared trades are removed. Ties prefer counterparties with stronger funding-source similarity and larger shared trade sets.

## Interventions

`interventional_score()` applies a `do(feature = value)` style intervention to the SCM and propagates the effect to downstream features. This is useful for questions like:

- What happens if the Benford anomaly is neutralized?
- Does the round-trip signal remain high after upstream changes?
- Which downstream indicators move together with the manipulated feature?

## Counterfactual vs SHAP

SHAP is correlational. It tells you which features are most associated with the model output.

The causal layer is operational. It tells you which trades and wallets change the score when removed or intervened on.

Use SHAP for attribution. Use causal scoring for investigation and evidence triage.

## Investigative Use Cases

- Identify the smallest trade subset that keeps a wallet below threshold.
- Rank counterparties by how much they contribute to the score.
- Trace the funding chain behind a flagged wallet.
- Test whether an apparent wash-trading signal propagates into downstream trade-pattern features.

---

## Causal Transfer Learning (Issue #255)

Causal models trained on one asset pair (e.g. USDC/XLM) do not always
generalise to structurally different pairs (e.g. a low-liquidity token pair)
because the causal graph structure may differ.  **Causal transfer learning**
identifies the invariant causal mechanisms shared across pairs and trains
pair-specific adjustments on top.

### Invariant Causal Prediction (ICP)

`CausalTransfer` (`detection/causal_transfer.py`) implements ICP
(Peters et al., 2016).

**Algorithm**:

1. Each unique asset pair is treated as a separate *environment*.
2. For every feature subset S ⊆ features (up to `max_subset_size=8`):
   - Fit a linear regression of the label on S within each environment.
   - Pool within-environment residuals.
   - Run a one-way ANOVA F-test at α=0.01 across environments.
   - If p > 0.01 (fail to reject equal-mean null), S is "potentially invariant".
3. The invariant feature set = intersection of all accepted subsets.
4. If no features survive the test (empty intersection), fall back to the global model.

### Shared mechanism + pair-specific adjustments

Once the invariant set is identified:

- A **global model** is trained on all environments using only the invariant features.
- A **pair-specific logistic regression** is trained on each environment's data
  as a local adjustment layer.
- At inference, the pair-specific model is used when the pair is known; the
  global model handles unseen pairs.

### Environment definition for Stellar DEX

Environments are defined by **asset pair**.  Alternative definitions
(time period, market regime) can be substituted by changing the `pair_col`
argument to `CausalTransfer.fit`.

### Security: anonymised environment labels

Raw pair IDs (e.g. `USDC:GA5Z.../XLM:native`) are hashed with SHA-256
(first 8 hex characters) before use as environment labels.  The raw pair
string is never stored in the fitted model.

### Usage

```python
from detection.causal_transfer import CausalTransfer

ct = CausalTransfer(feature_cols=["benford_chi_square_24h", "round_trip_frequency", ...])
result = ct.fit(train_df, pair_col="pair_id", label_col="label")

# Evaluate on a held-out pair
auc = ct.evaluate(test_df, pair_col="pair_id", label_col="label")

# Predict probabilities (uses pair-specific model if available)
probs = ct.predict_proba(new_df, pair_col="pair_id")
```

### Fallback behaviour

When `result.fallback_to_global is True`, all predictions use a single
logistic regression trained on the full feature set.  This happens when the
ANOVA test finds no stable features across environments.

### Generalisation benchmark

The benchmark compares three models on a held-out pair:

| Model | Description |
|---|---|
| Transferred causal model | `CausalTransfer` trained on all other pairs |
| Pair-specific model (10× data) | Trained on 10× more labelled data for the held-out pair |
| Global model | Single model trained on all data without causal transfer |

The target is: transferred model AUC ≥ pair-specific model AUC − 0.05.

### References

- Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal Inference Using Invariant Prediction: Identification and Confidence Intervals. *Journal of the Royal Statistical Society: Series B*, 78(5), 947–1012.
