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

## Instrumental Variable (IV) Estimation for Market Maker Activity

### The Problem: Confounding Between Market Makers and Wash Traders

Market maker wallets trade frequently and with concentrated counterparties by design. These observable features are statistically similar to wash trade patterns, so a purely observational model may assign inflated risk scores to legitimate market makers. Standard regression or SHAP cannot separate the causal effect of genuine market-making intensity from wash trade activity because both share the same confounder: high-frequency trading behaviour driven by unobserved strategy.

### What is 2-Stage Least Squares (2SLS)?

2SLS is an econometric technique for estimating causal effects when the variable of interest (the *endogenous* variable) is correlated with unobserved confounders. It works by finding an *instrument* — an external variable that:

1. **Relevance**: Strongly predicts the endogenous variable (market-making intensity).
2. **Exclusion restriction**: Has no direct effect on the outcome (risk score) other than through the endogenous variable.

**Stage 1** — Regress `counterparty_concentration_ratio` on the instrument (and any controls). This extracts the part of concentration that is driven purely by the instrument, removing the confounded variation.

**Stage 2** — Regress `risk_score` on the fitted values from Stage 1. The coefficient on the fitted values is the 2SLS causal estimate.

### The Instrument: Stellar DEX Liquidity Incentive Programme Flag

The instrument used in LedgerLens is whether a wallet has participated in a Stellar DEX liquidity incentive programme, derived from the `data_entries` field of account effects. The instrument is valid if:

- **Relevance**: Wallets in the programme trade more actively as market makers, which raises their `counterparty_concentration_ratio`. A strong first-stage F-statistic (> 10) confirms this.
- **Exclusion restriction**: Programme participation is unlikely to directly cause a wallet to engage in wash trading. Participation is an administrative flag granted by a protocol-level incentive; it does not change the underlying intent of trades. This assumption is untestable but economically plausible.

**Exclusion restriction risks**: A violation would occur if programme participants systematically engaged in wash trading as a side-effect of the incentive (e.g., to inflate volume metrics to qualify for rewards). Analysts should be aware of this risk and treat IV estimates as supportive evidence rather than proof.

### How to Interpret the Results

The IV estimate and its 95% confidence interval appear in the forensic report alongside the observational (OLS) estimate:

```
Method: 2SLS   coef=1.23   95% CI=[0.87, 1.59]   F=42.1   reliable=True
Method: OLS    coef=3.41   95% CI=[3.10, 3.72]
```

- **A 2SLS coefficient lower than OLS** is expected when market-making activity is upwardly confounding the risk score. The 2SLS estimate strips out the confounded variation.
- **First-stage F-statistic**: Must be > 10 for the instrument to be considered strong. F < 10 triggers a `reliable=False` flag and a `UserWarning`. Do not use weak-instrument IV estimates as evidence.
- **`reliable=False`**: Shown when the instrument is weak or when no programme participants exist in the analysis window (OLS fallback is used instead, with a confounding caveat).
- **OLS fallback**: Used automatically when no wallets in the batch hold the instrument. The report notes this and warns that estimates may be confounded.

**Important**: IV estimates are statistical approximations. The confidence interval quantifies sampling uncertainty but not violation of the exclusion restriction. These results must not be presented as definitive causal facts in legal or regulatory proceedings without review by a qualified econometrician.

### Reading the Forensic Report

The `iv_estimate` field in the forensic report contains:

| Field | Meaning |
|---|---|
| `method` | `"2SLS"` or `"OLS_fallback"` |
| `coef` | Point estimate of the causal effect on risk score |
| `ci_lower`, `ci_upper` | 95% HC1-robust confidence interval |
| `first_stage_f` | First-stage F-statistic (`null` for OLS fallback) |
| `reliable` | `false` if instrument is weak or OLS fallback was used |
| `warning` | Human-readable explanation when `reliable=false` |
| `disclaimer` | Mandatory legal/statistical caveat |

### Using `IVEstimator` Directly

```python
from detection.iv_estimator import IVEstimator

result = IVEstimator().estimate(
    endog=df["counterparty_concentration_ratio"],
    instrument=df["liquidity_programme_flag"],
    outcome=df["risk_score"],
    controls=df[["trade_count"]],  # optional
)
print(result)
# 2SLS coef=1.2345 95%CI=[0.8700, 1.5990]
```

If `np.std(instrument) < 1e-10` (no programme participants in the batch), the estimator falls back to OLS and emits a `UserWarning` with a confounding note.
