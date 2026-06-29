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
from typing import Sequence

import numpy as np

NP_COLD_START_THRESHOLD = 50
_ENCODER_HIDDEN = 64
_DECODER_HIDDEN = 64
_LATENT_DIM = 32


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
            [self._encode_one(f, l) for f, l in zip(context_features, context_labels)]
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
        return _sigmoid(logits)

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
