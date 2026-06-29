"""Temporal knowledge graph embedding for evolving wash trade ring detection.

Extends static knowledge graph embeddings (TransE, RotatE) with time-aware
relation embeddings (TComplEx model from PyKEEN) to reason about when
relationships hold, not just whether they hold.

Temporal KG triples: (wallet_A, traded_with, wallet_B, timestamp)
Timestamps are binned to 1-hour intervals.

Enables prediction of future links: e.g., if wallet C traded with common
counterparty X in hour t and wallet D also did so, predict they might trade
with each other in hour t+1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

# Temporal KGE configuration defaults
DEFAULT_KGE_EMBEDDING_DIM = 64
MAX_INFERENCE_TIME_MS = 5.0  # Inference must complete in <5ms

# Optional PyKEEN imports — graceful absence supported
try:
    from pykeen.models import DistMultiModel, RotatE, TComplEx, TransE
    from pykeen.triples import TriplesFactory
    from pykeen.training import SLCWATrainingLoop
    from pykeen.evaluation import RankBasedEvaluator

    _PYKEEN_AVAILABLE = True
except ImportError:
    _PYKEEN_AVAILABLE = False


class TemporalKGEError(Exception):
    """Raised when temporal KGE operations fail."""


class TemporalKGEncoder:
    """Temporal knowledge graph embedding encoder using TComplEx model.

    TComplEx (Tensor Completion with Complex Numbers) extends RotatE with
    temporal reasoning, modeling relationships as rotations in complex space
    with time-dependent parameters.

    Attributes:
        embedding_dim: Output embedding dimensionality.
        model_dir: Directory to save/load model artifacts.
        temporal_binning_hours: Hour interval for binning timestamps (default 1).
        inference_time_budget_ms: Maximum inference time per wallet pair.
    """

    def __init__(
        self,
        embedding_dim: int | None = None,
        model_dir: str | None = None,
        temporal_binning_hours: int = 1,
        inference_time_budget_ms: float = MAX_INFERENCE_TIME_MS,
        random_state: int = 42,
    ):
        """Initialize the temporal KGE encoder.

        Args:
            embedding_dim: Output embedding dimension (default config.KGE_EMBEDDING_DIM or 64).
            model_dir: Directory for model artifacts (default config.MODEL_DIR).
            temporal_binning_hours: Timestamp binning interval in hours (default 1).
            inference_time_budget_ms: Maximum time for single inference (default 5ms).
            random_state: Seed for reproducibility.
        """
        if not _PYKEEN_AVAILABLE:
            raise TemporalKGEError(
                "PyKEEN is required for temporal KGE. Install via: pip install pykeen"
            )

        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else DEFAULT_KGE_EMBEDDING_DIM
        )
        self.model_dir = model_dir or config.MODEL_DIR
        self.temporal_binning_hours = temporal_binning_hours
        self.inference_time_budget_ms = inference_time_budget_ms
        self.random_state = random_state

        self._model: TComplEx | None = None
        self._triples_factory: TriplesFactory | None = None
        self._entity_to_id: dict[str, int] = {}
        self._id_to_entity: dict[int, str] = {}
        self._relation_to_id: dict[str, int] = {}
        self._last_training_time: datetime | None = None

    def build_temporal_kg(self, trades_df: pd.DataFrame, reference_time: pd.Timestamp | None = None) -> dict:
        """Build temporal KG triples from trade data.

        Triples are (wallet_A, traded_with, wallet_B, timestamp_bin).
        Each hour forms a separate timestamp bin.

        Args:
            trades_df: DataFrame with columns: base_account, counter_account, ledger_close_time.
            reference_time: Current time for validating timestamps (default now).

        Returns:
            Dict with keys:
                - triples: List of (source, relation, target, time_bin) tuples
                - n_wallets: Number of unique entities
                - n_relations: Number of unique relations
                - n_timestamps: Number of unique time bins
                - timestamp_range: (min_bin, max_bin)

        Raises:
            ValueError: If trades_df is empty or timestamps are in the future.
        """
        if trades_df.empty:
            raise ValueError("trades_df cannot be empty")

        reference_time = reference_time or pd.Timestamp.now(tz="UTC")

        # Bin timestamps to temporal intervals
        timestamps = pd.to_datetime(trades_df["ledger_close_time"], utc=True)
        time_bins = timestamps.dt.floor(f"{self.temporal_binning_hours}h")

        # Validate no future timestamps
        if (time_bins > reference_time).any():
            raise ValueError(f"Timestamps must not be in the future (ref={reference_time})")

        triples: list[tuple[str, str, str, int]] = []
        entities: set[str] = set()
        timestamps_set: set[int] = set()

        for _, row in trades_df.iterrows():
            base = row["base_account"]
            counter = row["counter_account"]
            time_bin = time_bins.iloc[_] if isinstance(time_bins, pd.Series) else time_bins

            # Convert time_bin to integer (seconds since epoch, quantized to hour)
            time_bin_int = int(time_bin.timestamp() // (3600 * self.temporal_binning_hours))

            # Add undirected edge (both directions for symmetric trading)
            triples.append((base, "traded_with", counter, time_bin_int))
            triples.append((counter, "traded_with", base, time_bin_int))

            entities.add(base)
            entities.add(counter)
            timestamps_set.add(time_bin_int)

        # Build entity/relation mappings
        sorted_entities = sorted(entities)
        self._entity_to_id = {e: i for i, e in enumerate(sorted_entities)}
        self._id_to_entity = {i: e for e, i in self._entity_to_id.items()}
        self._relation_to_id = {"traded_with": 0}

        return {
            "triples": triples,
            "n_wallets": len(entities),
            "n_relations": 1,
            "n_timestamps": len(timestamps_set),
            "timestamp_range": (min(timestamps_set), max(timestamps_set)),
            "entity_id_map": self._entity_to_id,
        }

    def train(
        self,
        trades_df: pd.DataFrame,
        learning_rate: float = 0.1,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
    ) -> dict[str, Any]:
        """Train the temporal KGE model on trade data.

        Args:
            trades_df: Trade DataFrame with base_account, counter_account, ledger_close_time.
            learning_rate: Optimizer learning rate.
            num_epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_split: Fraction of data for validation.

        Returns:
            Training report dict with keys: n_epochs, final_loss, training_time_s, model_path.
        """
        import time

        if not _PYKEEN_AVAILABLE:
            raise TemporalKGEError("PyKEEN is required for training temporal KGE")

        start_time = time.time()

        # Build temporal KG
        kg_info = self.build_temporal_kg(trades_df)
        triples = kg_info["triples"]

        # Convert to integer tensor format (head_id, relation_id, tail_id, time_id)
        tensor_triples = []
        for head, rel, tail, time_id in triples:
            head_id = self._entity_to_id.get(head, 0)
            tail_id = self._entity_to_id.get(tail, 0)
            rel_id = self._relation_to_id.get(rel, 0)
            time_id_normalized = time_id  # Use as-is for temporal index
            tensor_triples.append((head_id, rel_id, tail_id, time_id_normalized))

        # Create PyKEEN triples factory
        try:
            # TComplEx requires (head, relation, tail, time) format
            import torch

            heads, relations, tails, times = zip(*tensor_triples)
            triples_tensor = torch.LongTensor([heads, relations, tails, times]).T

            # Initialize TComplEx model
            self._model = TComplEx(
                embedding_dim=self.embedding_dim,
                entity_initializer="uniform",
                relation_initializer="uniform",
                time_initializer="uniform",
            )

            # Configure training loop
            training_loop = SLCWATrainingLoop(
                model=self._model,
                optimizer_cls="Adam",
                optimizer_kwargs={"lr": learning_rate},
            )

            # Train for specified epochs
            losses = []
            for epoch in range(num_epochs):
                loss = training_loop.train(
                    triples_tensor,
                    batch_size=batch_size,
                    use_tqdm=False,
                )
                losses.append(float(loss))
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{num_epochs}: loss={loss:.4f}")

            final_loss = float(losses[-1]) if losses else float("nan")

        except Exception as e:
            raise TemporalKGEError(f"Training failed: {e}") from e

        training_time = time.time() - start_time
        self._last_training_time = datetime.now(timezone.utc)

        # Save model artifacts
        self._save_model()

        return {
            "n_epochs": num_epochs,
            "final_loss": final_loss,
            "training_time_s": training_time,
            "model_path": self._model_path(),
            "n_wallets": kg_info["n_wallets"],
            "n_relations": kg_info["n_relations"],
        }

    def predict_collaboration_score(
        self,
        wallet_a: str,
        wallet_b: str,
        target_time_bin: int | None = None,
    ) -> float:
        """Predict likelihood of future collaboration between two wallets.

        Uses the trained TComplEx model to compute a link prediction score
        for the given wallet pair at a future time.

        Args:
            wallet_a: First wallet (Stellar account ID).
            wallet_b: Second wallet (Stellar account ID).
            target_time_bin: Target time bin (default next hour).

        Returns:
            Float in [0, 1] representing predicted collaboration likelihood.
        """
        import time

        inference_start = time.perf_counter()

        if self._model is None:
            logger.warning("Model not trained; returning zero score")
            return 0.0

        if wallet_a not in self._entity_to_id or wallet_b not in self._entity_to_id:
            return 0.0  # Unknown wallets get zero score

        try:
            import torch

            head_id = self._entity_to_id[wallet_a]
            tail_id = self._entity_to_id[wallet_b]
            rel_id = 0  # "traded_with"

            # Default to next hour if not specified
            if target_time_bin is None:
                target_time_bin = int(datetime.now(timezone.utc).timestamp() // 3600)

            # Score the triple (wallet_a, traded_with, wallet_b, target_time_bin)
            with torch.no_grad():
                head_emb = self._model.entity_representations[0](torch.tensor([head_id]))
                rel_emb = self._model.relation_representations[0](torch.tensor([rel_id]))
                tail_emb = self._model.entity_representations[0](torch.tensor([tail_id]))
                time_emb = self._model.entity_representations[0](
                    torch.tensor([target_time_bin % 1000])  # Modulo to prevent overflow
                )

                # TComplEx scoring: compute rotation and magnitude
                score = self._tcmplex_score(head_emb, rel_emb, tail_emb, time_emb)

            # Normalize to [0, 1]
            normalized_score = float(1.0 / (1.0 + np.exp(-score.item())))

            inference_time = (time.perf_counter() - inference_start) * 1000
            if inference_time > self.inference_time_budget_ms:
                logger.warning(
                    f"Inference took {inference_time:.2f}ms, exceeds budget {self.inference_time_budget_ms}ms"
                )

            return normalized_score

        except Exception as e:
            logger.error(f"Prediction failed for {wallet_a}→{wallet_b}: {e}")
            return 0.0

    @staticmethod
    def _tcmplex_score(head_emb: Any, rel_emb: Any, tail_emb: Any, time_emb: Any) -> Any:
        """Compute TComplEx scoring function.

        S(h, r, t, τ) = Re(<h, r, t̄> + <r, τ, h̄>)

        where h, r, t are complex embeddings and τ is the time embedding.
        """
        import torch

        # Complex number operations
        score1 = torch.real(torch.sum(head_emb * rel_emb * torch.conj(tail_emb), dim=-1))
        score2 = torch.real(torch.sum(rel_emb * time_emb * torch.conj(head_emb), dim=-1))
        return score1 + score2

    def _save_model(self) -> None:
        """Save model artifacts with SHA-256 versioning."""
        if self._model is None:
            raise TemporalKGEError("No model to save; train first")

        os.makedirs(self.model_dir, exist_ok=True)

        try:
            import torch

            model_path = self._model_path()
            torch.save(self._model.state_dict(), model_path)

            # Compute artifact SHA-256
            artifact_sha = self._sha256_file(model_path)

            # Write metadata
            metadata = {
                "temporal_kge": {
                    "artifact_sha256": artifact_sha,
                    "embedding_dim": self.embedding_dim,
                    "model_type": "TComplEx",
                    "trained_at": self._last_training_time.isoformat() if self._last_training_time else None,
                    "n_entities": len(self._entity_to_id),
                    "entity_id_map": self._entity_to_id,
                }
            }

            metadata_path = os.path.join(self.model_dir, "metrics.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    existing = json.load(f)
                existing.update(metadata)
                metadata = existing

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved temporal KGE model to {model_path}")

        except Exception as e:
            raise TemporalKGEError(f"Failed to save model: {e}") from e

    def _load_model(self) -> None:
        """Load model artifacts with SHA-256 verification."""
        if not _PYKEEN_AVAILABLE:
            raise TemporalKGEError("PyKEEN required")

        model_path = self._model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            import torch

            # Verify SHA-256
            metadata_path = os.path.join(self.model_dir, "metrics.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)

                entry = metadata.get("temporal_kge", {})
                expected_sha = entry.get("artifact_sha256")
                if expected_sha:
                    actual_sha = self._sha256_file(model_path)
                    if actual_sha != expected_sha:
                        raise TemporalKGEError(
                            f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                        )

            # Load model state
            self._model = TComplEx(embedding_dim=self.embedding_dim)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()

            logger.info(f"Loaded temporal KGE model from {model_path}")

        except Exception as e:
            raise TemporalKGEError(f"Failed to load model: {e}") from e

    def _model_path(self) -> str:
        """Return path to model artifact."""
        return os.path.join(self.model_dir, "temporal_kge.pt")

    @staticmethod
    def _sha256_file(path: str) -> str:
        """Compute SHA-256 hash of file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


def build_temporal_kg_from_trades(
    trades_df: pd.DataFrame,
    temporal_binning_hours: int = 1,
) -> dict:
    """Build temporal knowledge graph triples from trade history.

    Args:
        trades_df: DataFrame with trades (base_account, counter_account, ledger_close_time).
        temporal_binning_hours: Hour interval for timestamp binning.

    Returns:
        Dict with triples, entity/relation maps, and metadata.
    """
    encoder = TemporalKGEncoder(temporal_binning_hours=temporal_binning_hours)
    return encoder.build_temporal_kg(trades_df)
