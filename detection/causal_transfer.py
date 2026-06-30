"""Causal transfer learning via Invariant Causal Prediction (Issue #255).

Peters et al. (2016) — "Causal Inference Using Invariant Prediction: Identification
and Confidence Intervals", JRSS-B.

ICP identifies the subset of features whose relationship to the label is
*stable* (invariant) across multiple environments (asset pairs).  These
invariant features form the shared causal mechanism; pair-specific adjustments
are trained on top.

Algorithm
---------
For each subset S ⊆ features:
  1. Fit a linear regression of label on S within each environment.
  2. Pool the within-environment residuals.
  3. Run an F-test at α=0.01 across environments; if residual distributions do
     not differ, S is accepted as potentially invariant.
The invariant set is the *intersection* of all accepted subsets that survive
the test.  If no features survive, fall back to the global model.

Environments
------------
Each unique asset pair in the training data is treated as one environment.
Pair IDs are anonymised (hashed) before use so raw pair strings are never
stored in the fitted model.

Security
--------
The environment→pair-ID mapping is anonymised before ICP.  Raw pair IDs are
not stored in the returned model object.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from utils.logging import get_logger

logger = get_logger(__name__)

ICP_ALPHA = 0.01  # significance level for the F-test invariance check


def _anon_env(pair_id: str) -> str:
    """Anonymise a pair ID to an opaque environment label."""
    return hashlib.sha256(pair_id.encode()).hexdigest()[:8]


def _residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS residuals of y ~ X."""
    if X.shape[1] == 0:
        return y - y.mean()
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    return y - reg.predict(X)


def _invariance_f_test(
    residuals_by_env: dict[str, np.ndarray],
    alpha: float = ICP_ALPHA,
) -> bool:
    """Return True if residuals are invariant across environments (F-test).

    Uses one-way ANOVA (equivalent to an F-test for equal means across groups).
    A p-value > alpha means we *fail to reject* the null hypothesis that the
    means are equal → the subset is invariant.
    """
    groups = [r for r in residuals_by_env.values() if len(r) >= 2]
    if len(groups) < 2:
        return True  # can't test with fewer than 2 environments
    f_stat, p_value = stats.f_oneway(*groups)
    return float(p_value) > alpha


@dataclass
class CausalTransferResult:
    """Result of a causal transfer fit."""
    invariant_features: list[str]
    fallback_to_global: bool
    pair_models: dict[str, Any] = field(default_factory=dict)
    global_model: Any = None


class CausalTransfer:
    """Invariant Causal Prediction + pair-specific linear adjustments.

    Usage::

        ct = CausalTransfer(feature_cols=["f1", "f2", "f3"])
        result = ct.fit(train_df, pair_col="pair_id", label_col="label")
        auc = ct.evaluate(test_df, pair_col="pair_id", label_col="label")

    Args:
        feature_cols: Feature column names to consider.
        alpha: F-test significance level (default ICP_ALPHA=0.01).
        max_subset_size: Cap on subset enumeration to avoid combinatorial explosion.
            Subsets larger than this are not tested (default 8).
    """

    def __init__(
        self,
        feature_cols: list[str],
        alpha: float = ICP_ALPHA,
        max_subset_size: int = 8,
    ) -> None:
        self.feature_cols = feature_cols
        self.alpha = alpha
        self.max_subset_size = max_subset_size
        self._result: CausalTransferResult | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        pair_col: str = "pair_id",
        label_col: str = "label",
    ) -> CausalTransferResult:
        """Fit ICP on *df* and train pair-specific adjustments.

        Returns a :class:`CausalTransferResult`.
        """
        # Anonymise environment labels
        df = df.copy()
        df["_env"] = df[pair_col].apply(_anon_env)
        envs = df["_env"].unique().tolist()

        feat_cols = [c for c in self.feature_cols if c in df.columns]
        if not feat_cols:
            raise ValueError("No valid feature columns found in DataFrame")

        y_all = df[label_col].values.astype(float)
        X_all = df[feat_cols].fillna(0.0).values.astype(float)

        # ICP: test all subsets up to max_subset_size
        invariant_subsets: list[frozenset[str]] = []
        n_feats = len(feat_cols)
        cap = min(n_feats, self.max_subset_size)

        for r in range(1, cap + 1):
            for subset in itertools.combinations(range(n_feats), r):
                subset_cols = [feat_cols[i] for i in subset]
                residuals_by_env: dict[str, np.ndarray] = {}
                for env in envs:
                    mask = df["_env"] == env
                    if mask.sum() < 2:
                        continue
                    X_env = df.loc[mask, subset_cols].fillna(0.0).values.astype(float)
                    y_env = df.loc[mask, label_col].values.astype(float)
                    residuals_by_env[env] = _residuals(X_env, y_env)

                if _invariance_f_test(residuals_by_env, alpha=self.alpha):
                    invariant_subsets.append(frozenset(subset_cols))

        # Invariant set = intersection of all accepted subsets
        if invariant_subsets:
            invariant_set = set(invariant_subsets[0])
            for s in invariant_subsets[1:]:
                invariant_set &= s
            invariant_features = sorted(invariant_set)
        else:
            invariant_features = []

        fallback = len(invariant_features) == 0

        if fallback:
            logger.warning(
                "ICP: no invariant feature set found across %d environments — "
                "falling back to global model",
                len(envs),
            )
            from sklearn.linear_model import LogisticRegression
            global_model = LogisticRegression(max_iter=500, random_state=42)
            global_model.fit(X_all, y_all.astype(int))
            result = CausalTransferResult(
                invariant_features=[],
                fallback_to_global=True,
                global_model=global_model,
            )
        else:
            logger.info(
                "ICP: invariant features = %s across %d environments",
                invariant_features,
                len(envs),
            )
            from sklearn.linear_model import LogisticRegression

            # Train pair-specific adjustments on top of invariant features
            pair_models: dict[str, Any] = {}
            for env in envs:
                mask = df["_env"] == env
                X_env = df.loc[mask, invariant_features].fillna(0.0).values.astype(float)
                y_env = df.loc[mask, label_col].values.astype(int)
                if len(np.unique(y_env)) < 2:
                    continue
                m = LogisticRegression(max_iter=500, random_state=42)
                m.fit(X_env, y_env)
                pair_models[env] = m

            # Global fallback (for unseen pairs)
            global_model = LogisticRegression(max_iter=500, random_state=42)
            X_inv = df[invariant_features].fillna(0.0).values.astype(float)
            global_model.fit(X_inv, y_all.astype(int))

            result = CausalTransferResult(
                invariant_features=invariant_features,
                fallback_to_global=False,
                pair_models=pair_models,
                global_model=global_model,
            )

        self._result = result
        return result

    # ------------------------------------------------------------------
    # Predict / Evaluate
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        df: pd.DataFrame,
        pair_col: str = "pair_id",
    ) -> np.ndarray:
        """Return class-1 probability for each row."""
        if self._result is None:
            raise RuntimeError("Call fit() before predict_proba()")

        result = self._result
        df = df.copy()
        df["_env"] = df[pair_col].apply(_anon_env)

        probs = np.zeros(len(df), dtype=float)

        if result.fallback_to_global:
            feat_cols = [c for c in self.feature_cols if c in df.columns]
            X = df[feat_cols].fillna(0.0).values.astype(float)
            probs = result.global_model.predict_proba(X)[:, 1]
        else:
            for i, (_, row_env) in enumerate(df["_env"].items()):
                model = result.pair_models.get(row_env, result.global_model)
                x = df.iloc[i][result.invariant_features].fillna(0.0).values.astype(float).reshape(1, -1)
                probs[i] = model.predict_proba(x)[0, 1]

        return probs

    def evaluate(
        self,
        test_df: pd.DataFrame,
        pair_col: str = "pair_id",
        label_col: str = "label",
    ) -> float:
        """Return AUC-ROC on *test_df*."""
        y_true = test_df[label_col].values.astype(int)
        y_pred = self.predict_proba(test_df, pair_col=pair_col)
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_pred))
