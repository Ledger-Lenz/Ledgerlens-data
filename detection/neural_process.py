"""Neural Process meta-learning for cold-start asset pair scoring.

Implements a Conditional Neural Process (CNP) that is meta-trained across all
existing asset pairs. At inference time, when a pair has fewer than
``NP_COLD_START_THRESHOLD`` labelled trades, the NP encoder embeds a small
context set and the decoder produces a calibrated risk score for new queries.

Architecture
------------
- Encoder: MLP that maps each (features, label) context trade to a fixed-dim
  representation, then reduces the variable-size context via mean pooling.
- Decoder: MLP that combines the pooled context embedding with a query feature
  vector and outputs a wash-trade probability.

Both components are intentionally lightweight (~2-layer MLPs) so the model can
run without a GPU on the inference path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

NP_COLD_START_THRESHOLD = 50
_ENCODER_HIDDEN = 64
_DECODER_HIDDEN = 64
_LATENT_DIM = 32

# Minimum recommended context size for reliable predictions (Issue #867)
# Below this threshold, predictions should be treated as highly uncertain
NP_MIN_RELIABLE_CONTEXT = 10

# Temperature scaling parameter for uncertainty calibration (Issue #867)
# Set via calibrate_temperature() after training
_DEFAULT_TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
# Pure-numpy MLP helpers (no PyTorch dependency at import time)
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _LinearLayer:
    """Single affine layer with optional ReLU."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = math.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray, activate: bool = True) -> np.ndarray:
        out = x @ self.W + self.b
        return _relu(out) if activate else out


class NeuralProcess:
    """Conditional Neural Process for few-shot asset-pair risk scoring.

    Parameters
    ----------
    feature_dim:
        Number of input features per trade (must match the feature schema).
    seed:
        Random seed for weight initialisation.
    """

    def __init__(self, feature_dim: int = 32, seed: int = 42):
        self.feature_dim = feature_dim
        rng = np.random.default_rng(seed)

        # Encoder: (feature_dim + 1) → hidden → latent_dim
        encoder_in = feature_dim + 1  # +1 for label
        self._enc1 = _LinearLayer(encoder_in, _ENCODER_HIDDEN, rng)
        self._enc2 = _LinearLayer(_ENCODER_HIDDEN, _LATENT_DIM, rng)

        # Decoder: (latent_dim + feature_dim) → hidden → 1
        decoder_in = _LATENT_DIM + feature_dim
        self._dec1 = _LinearLayer(decoder_in, _DECODER_HIDDEN, rng)
        self._dec2 = _LinearLayer(_DECODER_HIDDEN, 1, rng)

        # Temperature scaling for uncertainty calibration (Issue #867)
        self.temperature = _DEFAULT_TEMPERATURE

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode_one(self, features: np.ndarray, label: float) -> np.ndarray:
        """Encode a single (features, label) pair → latent vector."""
        x = np.concatenate([features.astype(np.float32), [float(label)]])
        h = self._enc1.forward(x[None], activate=True)[0]
        return self._enc2.forward(h[None], activate=False)[0]

    def encode_context(
        self,
        context_features: np.ndarray,
        context_labels: Sequence[float],
    ) -> np.ndarray:
        """Mean-pool over a variable-size context set → aggregated representation.

        Parameters
        ----------
        context_features:
            Shape ``(n_context, feature_dim)``.  ``n_context`` can range
            from 1 to 50.
        context_labels:
            Wash-trade labels (0 or 1) for each context trade.

        Returns
        -------
        np.ndarray of shape ``(latent_dim,)``
        """
        if len(context_features) == 0:
            return np.zeros(_LATENT_DIM, dtype=np.float32)

        encodings = np.stack(
            [
                self._encode_one(feature, label)
                for feature, label in zip(context_features, context_labels, strict=True)
            ]
        )
        return encodings.mean(axis=0)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, representation: np.ndarray, query_features: np.ndarray) -> np.ndarray:
        """Produce wash-trade probabilities for a batch of query trades.

        Parameters
        ----------
        representation:
            Aggregated context embedding, shape ``(latent_dim,)``.
        query_features:
            Shape ``(n_queries, feature_dim)``.

        Returns
        -------
        np.ndarray of shape ``(n_queries,)`` with values in ``[0, 1]``.
        """
        rep = np.broadcast_to(representation, (len(query_features), _LATENT_DIM))
        x = np.concatenate([rep, query_features.astype(np.float32)], axis=1)
        h = self._dec1.forward(x, activate=True)
        logits = self._dec2.forward(h, activate=False)[:, 0]
        # Apply temperature scaling for calibrated uncertainty (Issue #867)
        calibrated_logits = logits / self.temperature
        return _sigmoid(calibrated_logits)

    # ------------------------------------------------------------------
    # High-level inference
    # ------------------------------------------------------------------

    def predict(
        self,
        context_features: np.ndarray,
        context_labels: Sequence[float],
        query_features: np.ndarray,
    ) -> np.ndarray:
        """End-to-end prediction: encode context then decode queries.

        Parameters
        ----------
        context_features:
            Shape ``(n_context, feature_dim)``.
        context_labels:
            Binary labels for context trades.
        query_features:
            Shape ``(n_queries, feature_dim)``.

        Returns
        -------
        np.ndarray of shape ``(n_queries,)`` — wash-trade probability per query.
        """
        rep = self.encode_context(context_features, context_labels)
        return self.decode(rep, query_features)

    def predict_score(
        self,
        context_features: np.ndarray,
        context_labels: Sequence[float],
        query_feature_row: np.ndarray,
    ) -> float:
        """Return a single risk score in ``[0, 100]`` for one query trade."""
        probs = self.predict(context_features, context_labels, query_feature_row[None])
        return float(probs[0]) * 100.0

    def predict_with_uncertainty(
        self,
        context_features: np.ndarray,
        context_labels: Sequence[float],
        query_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions with calibrated uncertainty estimates.

        Parameters
        ----------
        context_features:
            Shape ``(n_context, feature_dim)``.
        context_labels:
            Binary labels for context trades.
        query_features:
            Shape ``(n_queries, feature_dim)``.

        Returns
        -------
        predictions: np.ndarray of shape ``(n_queries,)`` — wash-trade probabilities
        uncertainties: np.ndarray of shape ``(n_queries,)`` — uncertainty scores in [0, 1]
        """
        probs = self.predict(context_features, context_labels, query_features)
        
        # Uncertainty increases when context size is small
        n_context = len(context_features)
        context_penalty = max(0.0, 1.0 - n_context / NP_MIN_RELIABLE_CONTEXT)
        
        # Epistemic uncertainty: predictions near 0.5 are more uncertain
        prediction_uncertainty = 1.0 - 2.0 * np.abs(probs - 0.5)
        
        # Combined uncertainty
        uncertainties = np.clip(prediction_uncertainty + context_penalty, 0.0, 1.0)
        
        return probs, uncertainties

    def calibrate_temperature(
        self,
        val_context_features: list[np.ndarray],
        val_context_labels: list[Sequence[float]],
        val_query_features: list[np.ndarray],
        val_query_labels: list[Sequence[float]],
    ) -> float:
        """Calibrate temperature scaling parameter using validation data.

        Parameters
        ----------
        val_context_features:
            List of context feature arrays for each validation episode.
        val_context_labels:
            List of context label sequences for each validation episode.
        val_query_features:
            List of query feature arrays for each validation episode.
        val_query_labels:
            List of query label sequences for each validation episode.

        Returns
        -------
        float
            Optimal temperature parameter (stored in self.temperature).
        """
        from scipy.optimize import minimize_scalar

        def negative_log_likelihood(temp: float) -> float:
            """Compute negative log-likelihood with given temperature."""
            if temp <= 0:
                return 1e10
            
            old_temp = self.temperature
            self.temperature = temp
            
            nll = 0.0
            for ctx_feat, ctx_lab, qry_feat, qry_lab in zip(
                val_context_features,
                val_context_labels,
                val_query_features,
                val_query_labels,
                strict=False,
            ):
                probs = self.predict(ctx_feat, ctx_lab, qry_feat)
                probs = np.clip(probs, 1e-9, 1 - 1e-9)
                for prob, label in zip(probs, qry_lab, strict=False):
                    nll -= label * np.log(prob) + (1 - label) * np.log(1 - prob)
            
            self.temperature = old_temp
            return nll

        result = minimize_scalar(negative_log_likelihood, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self.temperature


# ---------------------------------------------------------------------------
# Cold-start blending helpers
# ---------------------------------------------------------------------------


def cold_start_blend_weight(trade_count: int, threshold: int = NP_COLD_START_THRESHOLD) -> float:
    """Linear blend weight for the NP score.

    Returns 1.0 when ``trade_count == 0`` (pure NP) and 0.0 when
    ``trade_count >= threshold`` (pure ensemble).
    """
    if trade_count >= threshold:
        return 0.0
    return 1.0 - trade_count / threshold


def blend_scores(
    np_score: float,
    ensemble_score: float,
    trade_count: int,
    threshold: int = NP_COLD_START_THRESHOLD,
) -> float:
    """Linearly blend NP and ensemble scores based on available trade count."""
    w = cold_start_blend_weight(trade_count, threshold)
    return w * np_score + (1.0 - w) * ensemble_score
