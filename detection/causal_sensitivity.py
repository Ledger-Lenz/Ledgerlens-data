"""E-value sensitivity analysis for causal attributions.

Implements VanderWeele & Ding (2017) E-value computation applied to the causal
attribution outputs from ``detection/causal_attribution.py``.

An E-value quantifies the minimum strength of association that an unobserved
confounder would need to have — simultaneously with both the exposure and the
outcome — to fully explain away the observed causal effect.  High E-values
indicate robust findings; low E-values indicate that even a weak confounder
could invalidate the attribution.

Reference
---------
VanderWeele, T.J. & Ding, P. (2017). Sensitivity Analysis in Observational
Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4),
268–274. https://doi.org/10.7326/M16-2607
"""

from __future__ import annotations

import math
from dataclasses import dataclass


EVALUE_LOW_CONFIDENCE_THRESHOLD = 2.0


def compute_evalue(risk_ratio: float) -> float:
    """Compute the E-value for a risk ratio (RR).

    Formula (VanderWeele & Ding, 2017):
        E = RR + sqrt(RR * (RR - 1))   for RR >= 1
        E = 1 / RR + sqrt((1/RR) * (1/RR - 1))  for RR < 1 (inverted)

    Special case: RR == 1.0 → E-value is 1.0 (no effect, no confounding needed).

    Parameters
    ----------
    risk_ratio:
        Estimated causal risk ratio (must be positive).

    Returns
    -------
    float — E-value >= 1.0.
    """
    if risk_ratio <= 0:
        raise ValueError(f"risk_ratio must be positive, got {risk_ratio}")

    if math.isclose(risk_ratio, 1.0, rel_tol=1e-9):
        return 1.0

    rr = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return rr + math.sqrt(rr * (rr - 1.0))


@dataclass
class SensitivityResult:
    """E-value result for a single causal attribution."""

    attribution_label: str
    risk_ratio: float
    evalue: float
    low_confidence: bool
    interpretation: str


def analyse_attribution(attribution_label: str, risk_ratio: float) -> SensitivityResult:
    """Compute E-value sensitivity for one causal attribution.

    Parameters
    ----------
    attribution_label:
        Human-readable description of the causal factor (e.g.
        ``'high counterparty concentration → +30 score points'``).
    risk_ratio:
        Estimated causal risk ratio for this attribution.

    Returns
    -------
    :class:`SensitivityResult` with E-value and confidence flag.
    """
    ev = compute_evalue(risk_ratio)
    low_conf = ev < EVALUE_LOW_CONFIDENCE_THRESHOLD
    if low_conf:
        interpretation = (
            f"Low confidence (E-value={ev:.2f} < {EVALUE_LOW_CONFIDENCE_THRESHOLD}): "
            "a relatively weak unobserved confounder could explain this attribution. "
            "Treat as advisory context only."
        )
    else:
        interpretation = (
            f"Robust attribution (E-value={ev:.2f}): an unobserved confounder would need "
            f"a {ev:.2f}× association with both exposure and outcome to explain away "
            "this effect."
        )
    return SensitivityResult(
        attribution_label=attribution_label,
        risk_ratio=risk_ratio,
        evalue=round(ev, 4),
        low_confidence=low_conf,
        interpretation=interpretation,
    )


def analyse_attributions(attributions: list[dict]) -> list[SensitivityResult]:
    """Batch E-value analysis over a list of attribution dicts.

    Each dict must have ``'label'`` (str) and ``'risk_ratio'`` (float) keys.
    Dicts missing ``'risk_ratio'`` are skipped.
    """
    results = []
    for attr in attributions:
        rr = attr.get("risk_ratio")
        label = attr.get("label", "unknown")
        if rr is None:
            continue
        results.append(analyse_attribution(label, float(rr)))
    return results
