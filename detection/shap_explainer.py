"""SHAP-based interpretability for risk scores.

Wraps each trained ensemble model with a SHAP explainer so that every
risk score can be accompanied by a per-feature attribution, surfaced via
the API for auditors and end-users.
"""

import numpy as np
import pandas as pd
import shap

from config import config
from detection.differential_privacy import (

    feature_sensitivity,
    gaussian_sigma,
    load_shap_sensitivity,
    renyi_noise_multiplier,
)
from detection.model_training import FEATURE_COLUMNS_EXCLUDE
from detection.feature_engineering import feature_dict_url as _feature_dict_url



class ShapExplainer:
    """Produces SHAP value explanations for one or more trained models.

    `TreeExplainer` construction is not free, so explainers are cached per
    model id (`id(model)`) and reused across calls.
    """

    def __init__(self, model=None):
        self._explainers: dict[int, shap.TreeExplainer] = {}
        self.model = model
        if model is not None:
            self.explainer = self._get_explainer(model)

    def _get_explainer(self, model) -> shap.TreeExplainer:
        key = id(model)
        if key not in self._explainers:
            self._explainers[key] = shap.TreeExplainer(model)
        return self._explainers[key]

    def _shap_values_for(self, model, X: pd.DataFrame):
        explainer = self._get_explainer(model)
        shap_values = explainer.shap_values(X)
        # Binary classifiers may return a list [class_0, class_1], or a single
        # ndarray shaped (n_samples, n_features, n_classes).
        if isinstance(shap_values, list):
            return shap_values[1][0]
        if shap_values.ndim == 3:
            return shap_values[0, :, 1]
        return shap_values[0]

    def explain(self, feature_row: pd.Series, top_n: int = 5, model=None) -> list[dict]:
        """Return the top `top_n` features driving this wallet's score
        according to a single model.

        Each entry: {"feature": str, "contribution": float, "value": float}
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided to explain()")

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T.astype(float)

        values = self._shap_values_for(model, X)

        contributions = sorted(
            zip(feature_cols, values, X.iloc[0].values, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        return [
            {"feature": name, "contribution": float(value), "value": float(raw),
             "dict_url": _feature_dict_url(name)}
            for name, value, raw in contributions
        ]

    def shap_dict(self, X: pd.DataFrame, model=None) -> dict[str, float]:
        """Return exact ``{feature: contribution}`` for the single row in `X`."""
        model = model or self.model
        if model is None:
            raise ValueError("No model provided to shap_dict()")

        feature_cols = [c for c in X.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        X_features = X[feature_cols].astype(float)
        values = self._shap_values_for(model, X_features)
        return {col: float(v) for col, v in zip(feature_cols, values, strict=True)}

    def explain_private(
        self,
        model,
        X: pd.DataFrame,
        wallet: str,
        epsilon: float | None = None,
        delta: float | None = None,
        *,
        private: bool = True,
        sensitivities: dict | None = None,
        query_store=None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Return SHAP values with calibrated Gaussian noise for (epsilon, delta)-DP.

        The Gaussian mechanism adds ``N(0, sigma_i^2)`` to each feature's SHAP
        value, where ``sigma_i`` is derived from that feature's sensitivity
        (max SHAP change from adding/removing one trade, see
        `scripts/estimate_shap_sensitivity.py`). This makes individual trade
        contributions indistinguishable, defeating model-inversion attacks.

        Audit mode (`private=False`) returns the exact SHAP values unchanged and
        consumes no privacy budget — intended for the authenticated, logged
        internal audit API only.

        When a `query_store` is supplied, the per-wallet query count is
        incremented and Rényi composition scales `sigma` once the wallet exceeds
        `config.DP_RENYI_QUERY_THRESHOLD` queries.
        """
        exact = self.shap_dict(X, model)
        if not private:
            return exact

        epsilon = config.DP_EPSILON if epsilon is None else epsilon
        delta = config.DP_DELTA if delta is None else delta
        if sensitivities is None:
            sensitivities = load_shap_sensitivity()

        query_count = 0
        if query_store is not None:
            query_count = query_store.increment_shap_query(wallet)
        noise_scale = renyi_noise_multiplier(query_count)

        rng = np.random.default_rng(seed)
        private_values: dict[str, float] = {}
        for feature, value in exact.items():
            sensitivity = feature_sensitivity(sensitivities, feature)
            sigma = gaussian_sigma(sensitivity, epsilon, delta) * noise_scale
            private_values[feature] = float(value + rng.normal(0.0, sigma))
        return private_values

    def compute_interaction_values(
        self, model, X: pd.DataFrame, top_n: int = 5
    ) -> list[dict]:
        """Return the top `top_n` pairwise feature interactions by absolute mean
        interaction value across all rows in `X`.

        Uses ``shap.TreeExplainer.shap_interaction_values``, which implements the
        Shapley interaction index (Lundberg et al., 2018).

        **API compatibility notes:**
        - XGBoost: ``TreeExplainer.shap_interaction_values`` returns an ndarray
          shaped ``(n_samples, n_features, n_features)`` for regressors and binary
          classifiers.  For multi-class XGBoost it returns a list of such arrays;
          we take index ``[1]`` (positive class).
        - sklearn RandomForest (binary): returns ``(n_samples, n_features, n_features,
          n_classes)``; we select ``[:, :, :, 1]`` for the positive class.
        - LightGBM: ``TreeExplainer`` supports interaction values via the same
          ``shap_interaction_values`` call when ``feature_perturbation="tree_path_dependent"``
          (the default).  The returned shape is ``(n_samples, n_features, n_features)``,
          identical to XGBoost.  LightGBM does **not** support
          ``feature_perturbation="interventional"`` for interaction values — if you
          override the explainer's ``feature_perturbation``, you will receive a
          ``NotImplementedError``.

        Raises ``RuntimeError`` if ``config.SHAP_INTERACTIONS_ENABLED`` is False.

        Returns a list of dicts::

            [{"feature_a": str, "feature_b": str, "interaction": float}, ...]
        """
        if not config.SHAP_INTERACTIONS_ENABLED:
            raise RuntimeError(
                "SHAP interaction values are disabled. "
                "Set SHAP_INTERACTIONS_ENABLED=true to enable them."
            )

        feature_cols = [c for c in X.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        X_features = X[feature_cols].astype(float)

        explainer = self._get_explainer(model)
        interaction_values = explainer.shap_interaction_values(X_features)

        # Normalise the output shape to (n_samples, n_features, n_features):
        #
        # XGBoost binary/regressor:  ndarray (n, d, d)  — no change needed
        # sklearn RF binary:         ndarray (n, d, d, n_classes)  — take [:, :, :, 1]
        # Multi-class list (XGBoost): list of (n, d, d) arrays — take index [1]
        # LightGBM binary:           ndarray (n, d, d)  — no change needed
        #   (LightGBM does NOT support feature_perturbation="interventional")
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]
        elif interaction_values.ndim == 4:
            # sklearn RF: (n, d, d, n_classes) — select positive class
            interaction_values = interaction_values[:, :, :, 1]

        # interaction_values: (n_samples, n_features, n_features)
        # Mean absolute value across samples; zero out diagonal (main effects)
        mean_abs = np.abs(interaction_values).mean(axis=0)
        np.fill_diagonal(mean_abs, 0.0)

        n = len(feature_cols)
        # Collect upper-triangle pairs only (symmetric matrix)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((feature_cols[i], feature_cols[j], float(mean_abs[i, j])))

        pairs.sort(key=lambda p: p[2], reverse=True)
        return [
            {"feature_a": fa, "feature_b": fb, "interaction": v}
            for fa, fb, v in pairs[:top_n]
        ]

    def explain_ensemble(self, feature_row: pd.Series, models: dict, top_n: int = 5) -> list[dict]:
        """Aggregate per-model SHAP contributions across an ensemble into a
        single ranked list.

        `models` maps model name -> fitted estimator (e.g. the `MODEL_REGISTRY`
        models loaded by `RiskScorer`). Contributions for each feature are
        averaged across models, then sorted by absolute magnitude.

        Each entry: {"feature": str, "contribution": float, "value": float}
        """
        if not models:
            raise ValueError("No models provided to explain_ensemble()")

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T.astype(float)
        raw_values = X.iloc[0].values

        totals = [0.0] * len(feature_cols)
        for model in models.values():
            values = self._shap_values_for(model, X)
            for i, value in enumerate(values):
                totals[i] += float(value)

        averaged = [total / len(models) for total in totals]

        contributions = sorted(
            zip(feature_cols, averaged, raw_values, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        return [
            {"feature": name, "contribution": float(value), "value": float(raw),
             "dict_url": _feature_dict_url(name)}
            for name, value, raw in contributions
        ]


def format_top_interactions(interactions: list[dict]) -> list[str]:
    """Format a list of interaction dicts (from ``ShapExplainer.compute_interaction_values``)
    into human-readable strings.

    Each output string has the form::

        "feature_a x feature_b contributes X.XXXX points to the score"

    Exactly ``len(interactions)`` strings are returned; callers should pass the
    top-N list they want formatted.
    """
    return [
        f"{item['feature_a']} x {item['feature_b']} contributes {item['interaction']:.4f} points to the score"
        for item in interactions
    ]
