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

## E-Value Sensitivity Analysis

### What Is an E-Value?

The causal attribution system attributes a wallet's risk score to specific causal
factors (e.g. *"high counterparty concentration caused a 30-point score increase"*).
However, the causal model may be confounded by unobserved variables — for example,
a market-maker responding to a liquidity event might share observable features with
a wash trader, but the underlying cause differs.

An **E-value** (VanderWeele & Ding, 2017) quantifies the minimum strength of
association that an unobserved confounder would need to have with *both* the exposure
and the outcome to fully explain away the observed causal effect.

- **High E-value** → the attribution is robust; a very strong confounder would be
  required to invalidate it.
- **Low E-value** → the attribution is fragile; even a modest unobserved variable
  could account for the observed effect.

### Formula

For a risk ratio RR ≥ 1:

```
E = RR + sqrt(RR × (RR − 1))
```

Special case: RR = 1.0 (no effect) → E-value = 1.0.

For RR < 1 the ratio is inverted first.

**Verification**: RR = 2.0 → E = 2.0 + √(2.0 × 1.0) = 2.0 + 1.414 ≈ **3.41**

### Confidence Threshold

Attributions with **E-value < 2.0** are flagged as *"low confidence — possible
confounding"* in the forensic report.  This threshold means a confounder with
only a 2× association with both exposure and outcome could invalidate the finding.

### For Compliance Officers

Think of the E-value as a *minimum bar for an alternative explanation*.  An
E-value of 3.41 means any alternative explanation would need to be at least
3.41× more common in wash traders than in legitimate traders, *and* 3.41× more
likely to produce the observed risk signal.  The higher the E-value, the harder
it is for any hidden factor to explain away the flagged behaviour.

E-values are **advisory context for investigators** and must not be used to
suppress alerts or change gating logic.  A low E-value means the finding
warrants more scrutiny, not dismissal.

### In the Forensic Report

Each causal attribution in the `CausalForensicReport` now includes a
`sensitivity_results` field:

```json
"sensitivity_results": [
  {
    "label": "overall score vs counterfactual",
    "risk_ratio": 2.0,
    "evalue": 3.41,
    "low_confidence": false,
    "interpretation": "Robust attribution (E-value=3.41): an unobserved confounder
      would need a 3.41× association with both exposure and outcome to explain
      away this effect."
  }
]
```

### Reference

VanderWeele, T.J. & Ding, P. (2017). Sensitivity Analysis in Observational
Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4), 268–274.
