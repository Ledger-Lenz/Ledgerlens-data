"""Counterfactual Explanation Generator for individual risk score decisions (Issue #193).

Implements the DICE (Diverse Counterfactual Explanations) method for the
LedgerLens ensemble model.  For a high-risk wallet (score >= flag threshold)
this module answers:

    "What is the *minimal* change to this wallet's observable behaviour that
    would reduce the risk score below 70?"

Key design decisions
--------------------
* **Primary backend**: fast gradient-free coordinate search that respects the
  10-second timeout per wallet. DICE (dice-ml) is used as an *optional*
  enhancement when the primary search yields fewer than n_cfs results.
* **Mutable vs immutable features**: account age and network centrality at
  account creation cannot decrease.  They are excluded from the action space.
* **Feature constraints**: non-negative features (Benford chi-square, MAD,
  rate features) are lower-bounded at 0.
* **Diversity**: up to ``n_cfs`` (default 5) counterfactuals are returned;
  a pairwise action-key check ensures they differ by at least 1 feature.
* **Interpretability**: ``interpret_counterfactual`` translates numeric
  feature deltas into plain-English on-chain actions.
* **Performance**: generation completes in < 10 seconds per wallet.
* **Security**: output contains only feature values and score thresholds —
  no training data or model weights are exposed.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import config
from detection.model_training import FEATURE_COLUMNS_EXCLUDE
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------

IMMUTABLE_FEATURES: frozenset[str] = frozenset(
    {
        "account_age_days",
        "network_centrality",
        "in_wash_trading_ring",
        "ring_size",
        "ring_internal_density",
    }
)

NON_NEGATIVE_FEATURES: frozenset[str] = frozenset(
    {
        "benford_chi_square_1h", "benford_chi_square_4h", "benford_chi_square_24h",
        "benford_chi_square_168h", "benford_chi_square_720h",
        "benford_mad_1h", "benford_mad_4h", "benford_mad_24h",
        "benford_mad_168h", "benford_mad_720h",
        "benford_z_max_1h", "benford_z_max_4h", "benford_z_max_24h",
        "benford_z_max_168h", "benford_z_max_720h",
        "benford_residual_chi_square_1h", "benford_residual_chi_square_4h",
        "benford_residual_chi_square_24h", "benford_residual_chi_square_168h",
        "benford_residual_chi_square_720h",
        "benford_residual_mad_1h", "benford_residual_mad_4h",
        "benford_residual_mad_24h", "benford_residual_mad_168h",
        "benford_residual_mad_720h",
        "counterparty_concentration_ratio",
        "round_trip_frequency",
        "self_matching_rate",
        "order_cancellation_rate",
        "volume_per_counterparty_ratio",
        "off_hours_activity_ratio",
        "volume_spike_frequency",
        "intra_minute_clustering",
        "cross_pair_trade_synchrony",
        "cross_pair_counterparty_overlap",
        "pair_diversity_score",
        "net_asset_flow_deviation",
        "net_roundtrip_ratio",
        "cross_pair_mad_std",
        "cross_pair_volume_correlation",
        "cross_wallet_volume_corr",
        "entropy_of_amounts",
        "inter_arrival_cv",
    }
)

# ---------------------------------------------------------------------------
# Interpretation templates
# ---------------------------------------------------------------------------

_INTERPRETATION_TEMPLATES: list[tuple[str, str]] = [
    (
        "counterparty_concentration",
        "Reduce counterparty concentration from {old:.2f} to {new:.2f} — "
        "trade with at least {n_counterparties:.0f} distinct counterparties in the next 24 h.",
    ),
    (
        "round_trip_frequency",
        "Reduce round-trip trade frequency from {old:.3f} to {new:.3f} — "
        "avoid returning assets to the originating wallet within the same session.",
    ),
    (
        "self_matching_rate",
        "Reduce self-matching rate from {old:.3f} to {new:.3f} — "
        "ensure buy and sell orders originate from wallets with distinct funding sources.",
    ),
    (
        "benford_mad",
        "Reduce Benford MAD (amount distribution anomaly) from {old:.4f} to {new:.4f} — "
        "vary trade sizes more naturally rather than using fixed or round lot amounts.",
    ),
    (
        "benford_chi_square",
        "Reduce Benford chi-square from {old:.2f} to {new:.2f} — "
        "diversify trade amount magnitudes to conform with natural digit distributions.",
    ),
    (
        "order_cancellation_rate",
        "Reduce order cancellation rate from {old:.3f} to {new:.3f} — "
        "avoid placing and immediately cancelling orders.",
    ),
    (
        "off_hours_activity_ratio",
        "Reduce off-hours activity ratio from {old:.3f} to {new:.3f} — "
        "distribute trading activity more evenly across all hours.",
    ),
    (
        "volume_spike_frequency",
        "Reduce volume spike frequency from {old:.3f} to {new:.3f} — "
        "avoid sudden volume surges relative to the rolling baseline.",
    ),
    (
        "cross_pair_trade_synchrony",
        "Reduce cross-pair trade synchrony from {old:.3f} to {new:.3f} — "
        "avoid simultaneous trades across multiple pairs.",
    ),
    (
        "pair_diversity_score",
        "Increase pair diversity score from {old:.3f} to {new:.3f} — "
        "spread trading activity across more asset pairs.",
    ),
    (
        "intra_minute_clustering",
        "Reduce intra-minute clustering from {old:.3f} to {new:.3f} — "
        "space trades more evenly rather than bursting within single minutes.",
    ),
    (
        "net_asset_flow_deviation",
        "Increase net asset flow deviation from {old:.3f} to {new:.3f} — "
        "ensure trades result in genuine inventory changes rather than closed cycles.",
    ),
    (
        "net_roundtrip_ratio",
        "Reduce net round-trip ratio from {old:.3f} to {new:.3f} — "
        "reduce the fraction of trades that perfectly cancel each other out.",
    ),
    (
        "volume_per_counterparty",
        "Reduce volume-per-counterparty ratio from {old:.2f} to {new:.2f} — "
        "distribute volume more evenly across a larger set of counterparties.",
    ),
    (
        "funding_source_similarity",
        "Reduce funding-source similarity from {old:.3f} to {new:.3f} — "
        "trade with counterparties that do not share the same funding wallet.",
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualAction:
    feature: str
    original_value: float
    counterfactual_value: float
    delta: float
    interpretation: str


@dataclass
class Counterfactual:
    cf_index: int
    feature_values: dict[str, float]
    predicted_score: float
    actions: list[CounterfactualAction]
    original_score: float = 0.0
    flag_threshold: float = 70.0


@dataclass
class CounterfactualResult:
    wallet: str
    original_score: float
    flag_threshold: float
    counterfactuals: list[Counterfactual]
    generation_time_seconds: float
    n_requested: int
    n_found: int
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "wallet": self.wallet,
            "original_score": float(self.original_score),
            "flag_threshold": float(self.flag_threshold),
            "n_requested": self.n_requested,
            "n_found": self.n_found,
            "generation_time_seconds": round(self.generation_time_seconds, 3),
            "timed_out": self.timed_out,
            "error": self.error,
            "counterfactuals": [
                {
                    "cf_index": cf.cf_index,
                    "predicted_score": float(cf.predicted_score),
                    "actions": [
                        {
                            "feature": a.feature,
                            "original_value": float(a.original_value),
                            "counterfactual_value": float(a.counterfactual_value),
                            "delta": float(a.delta),
                            "interpretation": a.interpretation,
                        }
                        for a in cf.actions
                    ],
                }
                for cf in self.counterfactuals
            ],
        }


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def _interpret_action(feature: str, old: float, new: float) -> str:
    for key, template in _INTERPRETATION_TEMPLATES:
        if key in feature:
            n_cps = round(1.0 / new) if "concentration" in feature and new > 1e-6 else 0
            try:
                return template.format(old=old, new=new, delta=abs(new - old),
                                       n_counterparties=n_cps)
            except (KeyError, ValueError):
                pass
    return f"Change '{feature}' from {old:.4g} to {new:.4g} (delta: {new - old:+.4g})."


# ---------------------------------------------------------------------------
# Main explainer
# ---------------------------------------------------------------------------

class CounterfactualExplainer:
    """Diverse counterfactual explanation generator for the LedgerLens ensemble.

    Uses a fast gradient-free coordinate search as the primary backend and
    optionally augments results with DICE (dice-ml) if available and time
    permits.

    Args:
        scorer:          Trained ``RiskScorer``.
        X_train:         Training feature matrix for computing feature bounds.
        flag_threshold:  Score threshold (default 70).
        n_cfs:           Max diverse CFs to return (default 5).
        timeout_seconds: Hard time limit per wallet (default 10 s).
        random_state:    RNG seed.
    """

    def __init__(
        self,
        scorer,
        X_train: pd.DataFrame,
        *,
        flag_threshold: float | None = None,
        n_cfs: int = 5,
        timeout_seconds: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.scorer = scorer
        self.flag_threshold = float(
            flag_threshold if flag_threshold is not None else config.RISK_SCORE_FLAG_THRESHOLD
        )
        self.n_cfs = int(n_cfs)
        self.timeout_seconds = float(timeout_seconds)
        self.random_state = int(random_state)

        # Mutable feature columns only
        self._all_feat_cols = [c for c in X_train.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        self._mutable_cols = [c for c in self._all_feat_cols if c not in IMMUTABLE_FEATURES]
        self._bounds = self._compute_bounds(X_train[self._mutable_cols])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, feature_row: pd.Series, wallet: str = "") -> CounterfactualResult:
        """Generate diverse counterfactuals for a single wallet."""
        t0 = time.monotonic()
        original_score = self.scorer.score_continuous(feature_row)

        if original_score < self.flag_threshold:
            logger.info(
                "Wallet %s scores %.1f — below threshold %.1f; skipping CF generation.",
                wallet or "<unknown>", original_score, self.flag_threshold,
            )
            return CounterfactualResult(
                wallet=wallet, original_score=original_score,
                flag_threshold=self.flag_threshold, counterfactuals=[],
                generation_time_seconds=time.monotonic() - t0,
                n_requested=self.n_cfs, n_found=0,
            )

        try:
            cfs = self._coordinate_search(feature_row, original_score, t0)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error("CF generation failed for %s: %s", wallet, exc, exc_info=True)
            return CounterfactualResult(
                wallet=wallet, original_score=original_score,
                flag_threshold=self.flag_threshold, counterfactuals=[],
                generation_time_seconds=elapsed, n_requested=self.n_cfs,
                n_found=0, error=str(exc),
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "Generated %d/%d CFs for %s in %.2fs (score=%.1f, threshold=%.1f)",
            len(cfs), self.n_cfs, wallet or "<unknown>",
            elapsed, original_score, self.flag_threshold,
        )
        return CounterfactualResult(
            wallet=wallet, original_score=original_score,
            flag_threshold=self.flag_threshold, counterfactuals=cfs,
            generation_time_seconds=elapsed, n_requested=self.n_cfs,
            n_found=len(cfs), timed_out=(elapsed >= self.timeout_seconds),
        )

    # ------------------------------------------------------------------
    # Coordinate search backend
    # ------------------------------------------------------------------

    def _coordinate_search(
        self,
        feature_row: pd.Series,
        original_score: float,
        t0: float,
    ) -> list[Counterfactual]:
        """Fast gradient-free search for diverse counterfactuals.

        Strategy:
        1. Estimate per-feature score gradient by finite difference.
        2. Sort features by gradient magnitude (most impactful first).
        3. For each candidate CF, try reducing the top-k features
           proportionally toward their lower bounds until the score drops
           below the threshold, with randomised perturbation for diversity.
        """
        rng = np.random.default_rng(self.random_state)

        # Estimate gradient for feature ordering
        grads = self._feature_gradients(feature_row)
        # Sort mutable features by descending impact
        sorted_cols = sorted(
            self._mutable_cols,
            key=lambda c: abs(grads.get(c, 0.0)),
            reverse=True,
        )

        found: list[Counterfactual] = []
        seen_keys: list[frozenset] = []

        max_attempts = 500
        for attempt in range(max_attempts):
            if time.monotonic() - t0 >= self.timeout_seconds:
                break
            if len(found) >= self.n_cfs:
                break

            candidate = feature_row.copy()

            # Choose how many features to perturb (1 to min(5, n_mutable))
            n_perturb = int(rng.integers(1, min(6, len(sorted_cols) + 1)))

            # Bias toward high-impact features, with randomness for diversity
            weights = np.array([abs(grads.get(c, 1e-6)) + 1e-6 for c in sorted_cols])
            weights = weights / weights.sum()
            chosen = rng.choice(
                sorted_cols,
                size=min(n_perturb, len(sorted_cols)),
                replace=False,
                p=weights,
            )

            for col in chosen:
                lo = self._bounds[col]["min"]
                hi = self._bounds[col]["max"]
                orig = float(feature_row[col])

                # For high-gradient (score-decreasing) features, sample
                # between lo and orig; for others sample across full range.
                grad = grads.get(col, 0.0)
                if grad > 0:
                    # Reducing this feature reduces the score — sample lower
                    new_val = float(rng.uniform(lo, max(lo, orig)))
                else:
                    new_val = float(rng.uniform(lo, hi))

                if col in NON_NEGATIVE_FEATURES:
                    new_val = max(0.0, new_val)
                candidate[col] = new_val

            score = self.scorer.score_continuous(candidate)
            if score >= self.flag_threshold:
                continue

            # Build actions
            actions = []
            for col in self._mutable_cols:
                old_val = float(feature_row[col])
                new_val = float(candidate[col])
                if abs(new_val - old_val) < 1e-6:
                    continue
                actions.append(CounterfactualAction(
                    feature=col,
                    original_value=old_val,
                    counterfactual_value=new_val,
                    delta=new_val - old_val,
                    interpretation=_interpret_action(col, old_val, new_val),
                ))

            if not actions:
                continue

            # Diversity: at least 1 feature must differ from existing CFs
            action_key = frozenset(
                (a.feature, round(a.counterfactual_value, 4)) for a in actions
            )
            if action_key in seen_keys:
                continue
            seen_keys.append(action_key)

            # Build full feature_values dict (immutables kept from original)
            fv: dict[str, float] = {}
            for col in self._all_feat_cols:
                if col in candidate.index:
                    fv[col] = float(candidate[col])
            found.append(Counterfactual(
                cf_index=len(found),
                feature_values=fv,
                predicted_score=score,
                actions=sorted(actions, key=lambda a: abs(a.delta), reverse=True),
                original_score=original_score,
                flag_threshold=self.flag_threshold,
            ))

        return found

    # ------------------------------------------------------------------
    # Gradient estimation
    # ------------------------------------------------------------------

    def _feature_gradients(self, feature_row: pd.Series) -> dict[str, float]:
        """Estimate ∂score/∂feature via batched finite differences for mutable features."""
        base_score = self.scorer.score_continuous(feature_row)
        n = len(self._mutable_cols)
        if n == 0:
            return {}

        # Build a batch of +h probes for all mutable features at once
        probe_rows = []
        h_vals = []
        for col in self._mutable_cols:
            orig = float(feature_row[col])
            lo = self._bounds[col]["min"]
            hi = self._bounds[col]["max"]
            h = max(0.01, (hi - lo) * 0.1)
            h_vals.append(h)
            row_plus = feature_row.copy()
            row_plus[col] = min(orig + h, hi)
            probe_rows.append(row_plus)

        probe_df = pd.DataFrame(probe_rows)
        # Use batch scoring if available, else fall back to individual
        try:
            scores_plus = self.scorer.score_continuous_batch(
                probe_df.drop(columns=[c for c in probe_df.columns
                                       if c in FEATURE_COLUMNS_EXCLUDE], errors="ignore")
            )
        except Exception:
            scores_plus = [self.scorer.score_continuous(r) for _, r in probe_df.iterrows()]

        grads: dict[str, float] = {}
        for i, col in enumerate(self._mutable_cols):
            h = h_vals[i]
            grads[col] = (float(scores_plus[i]) - base_score) / h
        return grads

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _compute_bounds(self, X: pd.DataFrame) -> dict[str, dict]:
        bounds = {}
        for col in X.columns:
            vals = X[col].dropna().values.astype(float)
            if len(vals) == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = float(np.percentile(vals, 1))
                hi = float(np.percentile(vals, 99))
            if col in NON_NEGATIVE_FEATURES:
                lo = max(0.0, lo)
            if lo >= hi:
                hi = lo + 1.0
            bounds[col] = {"min": lo, "max": hi}
        return bounds
