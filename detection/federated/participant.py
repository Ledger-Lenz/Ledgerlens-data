"""Federated learning participant.

Each participant:
1. Receives the global model weights from the coordinator.
2. Trains locally for E epochs on its private data.
3. Computes the weight delta (local - global).
4. Applies a pairwise additive mask and posts the masked delta to the coordinator.

The participant is an async HTTP client; it interacts with the coordinator via
the REST API defined in coordinator.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from sklearn.linear_model import SGDClassifier

from .crypto import generate_masks, mask_delta
from .gradient_compression import ErrorFeedbackCompressor, TopKSparsifier, TopKPayload
from .secure_aggregation import SecureAggregationContext, encrypt_gradient

logger = logging.getLogger(__name__)


@dataclass
class FederatedParticipant:
    participant_id: str
    coordinator_url: str
    # sklearn estimator with coef_/intercept_ (SGDClassifier by default)
    model: Any = field(default_factory=lambda: SGDClassifier(loss="log_loss", max_iter=1))
    # Gradient compressor — defaults to Top-1% sparsification with error feedback
    compressor: ErrorFeedbackCompressor = field(
        default_factory=lambda: ErrorFeedbackCompressor(TopKSparsifier(k_ratio=0.01))
    )
    # Optional CKKS secure aggregation context; when set, gradients are
    # encrypted before submission using homomorphic encryption (#188)
    secure_agg_context: SecureAggregationContext | None = None

    # ------------------------------------------------------------------ #
    # Weight helpers (flatten coef_ + intercept_ to a 1-D numpy vector)  #
    # ------------------------------------------------------------------ #

    def _get_weights(self) -> np.ndarray:
        return np.concatenate([self.model.coef_.ravel(), self.model.intercept_.ravel()])

    def _set_weights(self, w: np.ndarray) -> None:
        n_coef = self.model.coef_.size
        self.model.coef_ = w[:n_coef].reshape(self.model.coef_.shape)
        self.model.intercept_ = w[n_coef:].reshape(self.model.intercept_.shape)

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def local_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
    ) -> None:
        """Warm-start the model on local data for `epochs` passes."""
        for _ in range(epochs):
            self.model.partial_fit(X, y, classes=[0, 1])

    def compute_delta(self, global_weights: np.ndarray) -> np.ndarray:
        """Weight delta = local weights - global weights."""
        delta: np.ndarray = self._get_weights() - global_weights
        return delta

    # ------------------------------------------------------------------ #
    # Protocol                                                            #
    # ------------------------------------------------------------------ #

    async def run_round(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
    ) -> None:
        """Execute one federated round against the coordinator."""
        async with httpx.AsyncClient(base_url=self.coordinator_url) as client:
            # 1. Fetch current global weights + peer list
            resp = await client.get("/global_weights")
            resp.raise_for_status()
            payload = resp.json()
            global_w = np.array(payload["weights"])
            peers: list[str] = payload["participants"]

            # Initialise model weights from global state on first round
            if not hasattr(self.model, "coef_"):
                self.model.fit(X[:1], y[:1])  # warm-start shape
            self._set_weights(global_w)

            # 2. Local training
            self.local_train(X, y, epochs=epochs)
            delta = self.compute_delta(global_w)

            # 2a. Compress gradient (with error-feedback memory correction)
            payload = self.compressor.compress(delta)
            if isinstance(payload, TopKPayload):
                compressed_delta = TopKSparsifier.decompress(payload)
            else:
                from .gradient_compression import PowerSGDCompressor
                compressed_delta = PowerSGDCompressor.decompress(payload)

            # 3. Mask compressed delta (pairwise additive masks cancel at coordinator)
            all_ids = sorted(set(peers) | {self.participant_id})
            masks = generate_masks(all_ids, compressed_delta.shape)
            my_mask = masks[self.participant_id]
            masked = mask_delta(compressed_delta, my_mask)

            # 4. Submit delta — encrypted when secure_agg_context is configured (#188)
            if self.secure_agg_context is not None:
                ciphertext = encrypt_gradient(masked, self.secure_agg_context)
                resp = await client.post(
                    "/submit_encrypted_delta",
                    content=ciphertext,
                    headers={"Content-Type": "application/octet-stream",
                             "X-Participant-Id": self.participant_id},
                )
            else:
                resp = await client.post(
                    "/submit_delta",
                    json={
                        "participant_id": self.participant_id,
                        "delta": masked.tolist(),
                    },
                )
            resp.raise_for_status()
            logger.info("Participant %s submitted delta for round.", self.participant_id)
