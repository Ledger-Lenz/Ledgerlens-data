"""Risk-score contracts for cross-package communication.

The ``RiskScore`` TypedDict is the single most important cross-package
data structure in the platform — produced by the detection layer and
consumed by streaming, integrations, alerts, and reporting.

The ``Scorer`` protocol defines the interface that every scoring
component (production, shadow-deployment, ensemble) must satisfy.
"""

from __future__ import annotations

import typing
from typing import Protocol, TypedDict, runtime_checkable


class RiskScore(TypedDict, total=False):
    """The canonical risk-score shape flowing from detection to consumers.

    All fields are optional via ``total=False`` so that producers can
    include only the fields they populate and consumers can safely
    access fields via ``.get()``.

    Required in practice
    --------------------
    - ``wallet`` — wallet public key
    - ``asset_pair`` — e.g. ``"USDC_native"``
    - ``score`` — integer risk score 0–100
    - ``benford_flag`` — whether Benford analysis flagged
    - ``ml_flag`` — whether ML model flagged
    - ``confidence`` — confidence level 0–100
    - ``timestamp`` — unix seconds
    """

    wallet: str
    asset_pair: str
    score: int
    benford_flag: bool
    ml_flag: bool
    confidence: int
    timestamp: int

    # Extended fields
    propagated_risk: float
    ring_id: int | None
    score_lower: float
    score_upper: float
    coverage_guarantee: float
    replay_model_version: str
    model_name: str
    feature_contributions: dict[str, float]

    # ------------------------------------------------------------------
    # Planned for Issue #856 (uncertainty-aware score fusion)
    # ------------------------------------------------------------------
    # New optional fields. They are additive, and because the TypedDict is
    # total=False, nothing that reads RiskScore today breaks:
    #
    #   fused_score_lower: float   # lower bound of fused interval, 0-100
    #   fused_score_upper: float   # upper bound of fused interval, 0-100
    #   fusion_strategy: str       # "fixed_weight" | "inverse_variance" | "stacked"
    #
    # Migration note for consuming repos (ledgerlens-core, ledgerlens-api,
    # ledgerlens-contract, ledgerlens-dashboard):
    #   - No action needed to keep working: `score`, `score_lower` and
    #     `score_upper` keep their current meaning. `score_lower`/`score_upper`
    #     stay the union of per-model conformal intervals.
    #   - To use the new interval, read `fused_score_lower` /
    #     `fused_score_upper` with `.get()` and fall back to
    #     `score_lower` / `score_upper` when absent (older producers, or the
    #     strategy is "fixed_weight" with no calibrators loaded).
    #   - `fusion_strategy` tells consumers how `score` was produced. When it
    #     is "inverse_variance" or "stacked", `score` can differ from the
    #     previous BFT trimmed-mean value for the same wallet. Dashboards
    #     comparing scores over time should group by this field.
    #   - ledgerlens-contract (on-chain) needs no change: only the integer
    #     `score` is written on-chain.
    #   - ledgerlens-core's shared type should add the three keys as
    #     optional to mirror this TypedDict.


@runtime_checkable
class Scorer(Protocol):
    """Interface for a component that produces risk scores.

    Implementations include ``RiskScorer``, ``ShadowDeploymentScorer``,
    and ensemble wrappers.

    Usage::

        def score(self, feature_row: pd.Series, **kwargs: Any) -> RiskScore:
            ...
    """

    def score(self, feature_row: typing.Any, **kwargs: typing.Any) -> RiskScore:
        """Score a single feature row and return a ``RiskScore`` dict."""
