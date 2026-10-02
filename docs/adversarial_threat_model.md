# Adversarial Robustness Threat Model

This document formalizes the threat model assumed by the adversarial-robustness
modules in this repository. It describes attacker capabilities, the knowledge an
attacker is assumed to possess, and what is explicitly **out of scope** for each
defense. Each defense module's docstring cross-references the relevant section
below.

## 1. Scope and conventions

- **Assets.** Model parameters/updates, training data, inference inputs, and the
  integrity of the aggregation and certification pipeline.
- **Trust boundary.** The aggregator/verifier is trusted; participating clients
  and any network transport are untrusted.
- **Threat identifiers.** `T1`–`T5` are referenced from module docstrings.

## 2. Threat catalog

| ID  | Threat | Attacker model |
|-----|--------|----------------|
| T1  | Evasion / adversarial input perturbation | Black-box or white-box, inference-time |
| T2  | Poisoning via malicious model updates | Byzantine clients, up to `f` of `n` |
| T3  | Sybil / collusion among clients | Multiple identities controlled by one adversary |
| T4  | Backdoor / targeted misclassification | White-box, training-time |
| T5  | Certification forgery / replay | Network or client-level, message-level |

## 3. Per-module threat model

### 3.1 `certified_robustness.py`

- **Mitigates:** T1 (evasion), T4 (backdoor certification bounds).
- **Attacker capabilities:** may perturb inputs within an Lp ball of radius `eps`;
  may attempt to forge or replay certification messages (T5).
- **Assumed knowledge:** white-box access to the model and the certification
  procedure; black-box access to the deployed endpoint.
- **Query budget:** unbounded for the certification analysis; the certificate is
  valid only for the declared `eps` and norm.
- **Graph-structure access:** none assumed.
- **Out of scope:** poisoning (T2/T3), transport-level attacks, and any guarantee
  outside the certified `eps`/norm.

### 3.2 `adversarial/*`

- **Mitigates:** T2 (poisoning), T3 (Sybil/collusion) via robust aggregation.
- **Attacker capabilities:** up to `f` Byzantine clients may submit arbitrary
  updates; colluding clients may coordinate (T3).
- **Assumed knowledge:** black-box with respect to honest clients' data; the
  attacker knows the aggregation rule but not honest gradients.
- **Query budget:** one update per round per controlled identity.
- **Graph-structure access:** none assumed; defenses do not rely on the
  communication graph being honest.
- **Out of scope:** evasion at inference time (T1), and guarantees when the
  fraction of Byzantine clients exceeds the module's stated bound.

## 4. Cross-reference summary

| Module | Threats mitigated | Out of scope |
|--------|-------------------|--------------|
| `certified_robustness.py` | T1, T4, T5 | T2, T3 |
| `adversarial/*` | T2, T3 | T1, T4 |

## 5. Identified gaps (follow-up issues)

- **Gap G1:** No defense currently binds a certificate to a specific model
  version, so a certificate issued for model `v1` may be replayed against model
  `v2` (T5). Filed as a separate tracked issue.
- **Gap G2:** Robust aggregation bounds are stated for a fixed `f`; behavior
  under adaptive, round-varying Byzantine fractions is undocumented.
