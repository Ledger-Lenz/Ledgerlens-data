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
