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

---

## Prior Knowledge Constraints (Issue #192)

### Why Prior Constraints?

Purely data-driven causal graphs can contain spurious edges or miss known
causal relationships.  The PC algorithm is powerful but may infer, for
example, that trading volume *causes* account age — a physically impossible
relationship, since account age is determined at creation and cannot be
changed retroactively.

`CausalPriorConstraints` prevents such violations by encoding domain-expert
knowledge as (cause, effect, required|forbidden) triples that are enforced
*after* the data-driven discovery step.

### Constraint Types

| Kind | Semantics | Consequence if violated |
|---|---|---|
| **required** (hard) | This directed edge MUST appear in the learned DAG | Missing or reversed edges are inserted / corrected. |
| **forbidden** (soft) | This directed edge must NOT appear in the learned DAG | Violating edges are removed and a `UserWarning` is emitted. |

The "soft" designation for forbidden edges reflects that data evidence alone
is not sufficient to override a domain-validated constraint — the constraint
always wins — but the conflict is surfaced as a warning for operator review.

### Constraint Format (YAML)

Constraints are stored in `data/causal_priors.yaml` as a list of triples:

```yaml
constraints:
  # Hard constraint: account age causally precedes trading activity
  - cause: account_age_days
    effect: round_trip_frequency
    kind: required

  # Soft constraint: trading cannot make an account older
  - cause: round_trip_frequency
    effect: account_age_days
    kind: forbidden
```

| Field | Type | Description |
|---|---|---|
| `cause` | string | Name of the causal variable (must exist in the feature set). |
| `effect` | string | Name of the effect variable (must exist in the feature set). |
| `kind` | `"required"` or `"forbidden"` | Constraint type (case-insensitive). |

The YAML is parsed with `yaml.safe_load` — `!!python/...` object tags are
rejected, preventing injection of arbitrary Python objects.

### Security: Schema Validation

Every call to `CausalPriorConstraints.load()` performs:

1. Top-level key check: `constraints` must be a list of mappings.
2. Per-constraint validation: `cause`, `effect`, `kind` are required string
   fields; `kind` must be `"required"` or `"forbidden"`.
3. Variable validation (via `priors.validate(feature_columns)`): all named
   variables must exist in the feature set; unknown variables raise a
   `ValueError` with a clear message listing every unknown variable.

### Domain-Validated Constraints (Stellar DEX)

The shipped `data/causal_priors.yaml` encodes the following domain facts:

| Constraint | Kind | Rationale |
|---|---|---|
| `account_age_days → round_trip_frequency` | required | Account creation predates all trading. |
| `account_age_days → self_matching_rate` | required | Same temporal reasoning. |
| `funding_source_similarity → round_trip_frequency` | required | Shared funder → coordinated wash-ring round-trips. |
| `counterparty_concentration_ratio → label` | required | High concentration is a direct wash-trade signal. |
| `benford_mad_24h → label` | required | Benford non-conformity is a direct wash-trade signal. |
| `round_trip_frequency → account_age_days` | forbidden | Time cannot run backward. |
| `volume_per_counterparty_ratio → account_age_days` | forbidden | Same: age is immutable. |
| `label → benford_mad_24h` | forbidden | Label is derived from features, not the reverse. |
| `label → counterparty_concentration_ratio` | forbidden | Same: label cannot cause its own inputs. |

### Usage

```python
from detection.causal_discovery import CausalPriorConstraints, WashTradeCausalDiscovery

# Load and validate priors
priors = CausalPriorConstraints.load("data/causal_priors.yaml")
priors.validate(df.columns)   # raises ValueError if unknown variables

# Run causal discovery with prior enforcement
discoverer = WashTradeCausalDiscovery()
dag = discoverer.fit(df, alpha=0.05, priors=priors)
```

Priors are optional: calling `discoverer.fit(df)` without `priors` is
fully backward-compatible.

### Adding New Constraints

1. Identify the domain fact (e.g. "A precedes B in the wash-trade pipeline").
2. Add the triple to `data/causal_priors.yaml`.
3. Run `python -m pytest tests/test_causal_prior_constraints.py -v` to verify.
4. If the constraint references a new feature column, also update the
   `test_yaml_features_exist_in_synthetic_dataset` test.

### References

- Spirtes, P., Glymour, C. & Scheines, R. (2000) *Causation, Prediction, and
  Search*, MIT Press.
- Pearl, J. (2009) *Causality*, 2nd ed., Cambridge University Press.
- VanderWeele, T.J. & Ding, P. (2017) "Sensitivity Analysis in Observational
  Research: Introducing the E-Value", *Annals of Internal Medicine*, 167(4).
