"""Transformer-based sequence model for temporal trade pattern detection (#182).

Implements a lightweight transformer encoder that operates on ordered sequences
of trade events per wallet.  The model captures sequential wash-trading patterns
(buy/sell ping-pong cycles, gradual volume escalation before a pump) that are
invisible to the aggregate feature vector consumed by the ensemble.

Architecture overview
---------------------
- **Input projection**: each trade event is embedded from 4 raw fields
  (log-amount, time-delta, pair-one-hot, direction) into ``embed_dim`` via a
  linear projection.
- **Positional encoding**: sinusoidal (no learned position embeddings — avoids
  aliasing on sequences shorter than the max seen at training time).
- **Transformer encoder**: 2–4 ``TransformerEncoderLayer`` blocks, each with
  multi-head self-attention + FFN.  Padding masks ensure that padded positions
  do not contribute to attention.
- **Pooling**: mean pooling over non-padding positions → ``embed_dim``
  sequence-level embedding.
- **Output**: sequence risk embedding (``embed_dim`` dims), concatenated with
  the existing 37-feature vector in ``model_inference.py``.

Input validation
----------------
``TradeSequenceTransformer.encode_trades`` validates:
  - sequence length ≤ ``config.SEQ_MODEL_MAX_LENGTH`` (guards memory exhaustion)
  - individual fields are finite floats

Persistence
-----------
Weights are saved to ``config.MODEL_DIR/sequence_transformer.pt`` and an
integrity entry is written to ``metrics.json``.  The load path validates the
SHA-256 of the saved file against the manifest (same pattern as
``detection/gnn_encoder.py``).

CPU latency target
------------------
Batch size 32, sequence length 64 completes in < 50 ms on a modern CPU.
``TradeSequenceTransformer.benchmark_latency()`` can be called to verify this.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional torch imports — graceful absence lets the rest of the codebase
# import this module even when torch is not installed.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ARTIFACT_FILENAME = "sequence_transformer.pt"
_METRICS_FILENAME = "metrics.json"

# Input dimensionality breakdown:
#   1  — normalised log-amount
#   1  — normalised time-delta (seconds since previous trade)
#   N  — asset pair one-hot  (N = config.SEQ_MODEL_NUM_PAIRS)
#   1  — direction (1 = buy, -1 = sell, 0 = unknown)
# Total = 3 + SEQ_MODEL_NUM_PAIRS
_INPUT_BASE_DIM = 3  # log_amount + time_delta + direction


class TradeEvent(NamedTuple):
    """Parsed trade event ready for embedding.

    Fields
    ------
    log_amount : float
        Natural log of the trade base_amount, normalised to [0, 1] at the
        batch level or globally.  NaN/Inf are rejected before reaching the model.
    time_delta_s : float
        Seconds since the previous trade event (0 for the first trade in a
        sequence).
    pair_index : int
        Integer index into the known asset-pair vocabulary
        (0 = unknown/OOV pair).
    direction : float
        +1.0 = base_account is the buyer, -1.0 = base_account is the seller,
        0.0 = unknown.
    """

    log_amount: float
    time_delta_s: float
    pair_index: int
    direction: float


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_pe(max_len: int, embed_dim: int) -> "torch.Tensor":
    """Build a (max_len, embed_dim) sinusoidal positional encoding matrix."""
    pe = torch.zeros(max_len, embed_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * (-math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, embed_dim)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TradeSequenceTransformer(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Lightweight transformer encoder over wallet trade sequences.

    Parameters
    ----------
    num_pairs : int
        Size of the asset-pair vocabulary (one-hot width).  Defaults to
        ``config.SEQ_MODEL_NUM_PAIRS``.
    embed_dim : int
        Dimensionality of the internal token embeddings and the output
        sequence embedding.  Defaults to ``config.SEQ_MODEL_EMBED_DIM``.
    num_heads : int
        Number of attention heads.  Must divide ``embed_dim``.
        Defaults to ``config.SEQ_MODEL_NUM_HEADS``.
    num_layers : int
        Number of transformer encoder layers (2–4 recommended).
        Defaults to ``config.SEQ_MODEL_NUM_LAYERS``.
    ffn_dim : int
        Feed-forward expansion dimension inside each transformer layer.
        Defaults to ``config.SEQ_MODEL_FFN_DIM``.
    dropout : float
        Dropout rate applied inside attention and FFN (disabled at eval time).
        Defaults to ``config.SEQ_MODEL_DROPOUT``.
    max_length : int
        Maximum allowed sequence length.  Inputs longer than this are
        rejected with a ``ValueError`` before reaching the model.
        Defaults to ``config.SEQ_MODEL_MAX_LENGTH``.
    """

    def __init__(
        self,
        num_pairs: int | None = None,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        ffn_dim: int | None = None,
        dropout: float | None = None,
        max_length: int | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TradeSequenceTransformer. "
                "Install it with: pip install torch"
            )
        super().__init__()

        self.num_pairs = num_pairs or config.SEQ_MODEL_NUM_PAIRS
        self.embed_dim = embed_dim or config.SEQ_MODEL_EMBED_DIM
        self.num_heads = num_heads or config.SEQ_MODEL_NUM_HEADS
        self.num_layers = num_layers or config.SEQ_MODEL_NUM_LAYERS
        self.ffn_dim = ffn_dim or config.SEQ_MODEL_FFN_DIM
        self.dropout_p = dropout if dropout is not None else config.SEQ_MODEL_DROPOUT
        self.max_length = max_length or config.SEQ_MODEL_MAX_LENGTH

        # Input dimensionality: base dims + one-hot pair vector
        input_dim = _INPUT_BASE_DIM + self.num_pairs

        # Input projection: raw event fields → embed_dim
        self.input_proj = nn.Linear(input_dim, self.embed_dim)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = _sinusoidal_pe(self.max_length, self.embed_dim)
        self.register_buffer("pos_enc", pe)  # (max_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout_p,
            batch_first=True,  # (batch, seq, embed_dim)
            norm_first=True,   # Pre-LN: better gradient flow for small models
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # Layer-norm on the pooled output
        self.output_norm = nn.LayerNorm(self.embed_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: "torch.Tensor",
        padding_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Encode a batch of trade sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_dim)``.  Padded with zeros.
        padding_mask : torch.Tensor | None
            Boolean mask of shape ``(batch, seq_len)``.  ``True`` positions are
            *ignored* by the transformer (i.e. ``True`` = this position is
            padding).  When ``None``, no masking is applied.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, embed_dim)`` — one embedding vector per sequence.
        """
        # Input projection: (batch, seq, input_dim) → (batch, seq, embed_dim)
        tokens = self.input_proj(x)

        # Add positional encoding (broadcast over batch)
        seq_len = x.size(1)
        tokens = tokens + self.pos_enc[:seq_len].unsqueeze(0)  # (batch, seq, embed_dim)

        # Transformer: attend over sequence positions
        # src_key_padding_mask shape: (batch, seq_len), True = padding
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )  # (batch, seq, embed_dim)

        # Mean-pool over non-padding positions
        if padding_mask is not None:
            # non_pad: (batch, seq, 1) — 1.0 for real positions
            non_pad = (~padding_mask).float().unsqueeze(-1)
            pooled = (encoded * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)  # (batch, embed_dim)

        return self.output_norm(pooled)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_sequence(events: list[TradeEvent], max_length: int) -> None:
        """Validate a sequence before encoding.

        Raises
        ------
        ValueError
            If the sequence is empty, too long, or contains non-finite values.
        """
        if not events:
            raise ValueError("Trade sequence must contain at least one event.")
        if len(events) > max_length:
            raise ValueError(
                f"Sequence length {len(events)} exceeds maximum allowed length "
                f"{max_length}.  Truncate or reject the input before calling encode."
            )
        for i, ev in enumerate(events):
            if not math.isfinite(ev.log_amount):
                raise ValueError(
                    f"Event {i}: log_amount={ev.log_amount!r} is not finite."
                )
            if not math.isfinite(ev.time_delta_s):
                raise ValueError(
                    f"Event {i}: time_delta_s={ev.time_delta_s!r} is not finite."
                )
            if not math.isfinite(ev.direction):
                raise ValueError(
                    f"Event {i}: direction={ev.direction!r} is not finite."
                )

    def events_to_tensor(
        self,
        events: list[TradeEvent],
    ) -> "torch.Tensor":
        """Convert a single sequence of TradeEvents to a (seq_len, input_dim) tensor.

        The pair index is one-hot encoded.  Time-deltas are normalised by
        ``log1p`` to compress the long tail of delays.
        """
        input_dim = _INPUT_BASE_DIM + self.num_pairs
        t = torch.zeros(len(events), input_dim)
        for i, ev in enumerate(events):
            t[i, 0] = ev.log_amount
            t[i, 1] = math.log1p(max(ev.time_delta_s, 0.0))  # log(1+delta)
            t[i, 2] = ev.direction
            # One-hot pair encoding (index 0 = OOV)
            pair_idx = min(ev.pair_index, self.num_pairs - 1)
            if pair_idx > 0:
                t[i, 3 + pair_idx - 1] = 1.0
        return t

    def encode_trades(
        self,
        event_sequences: list[list[TradeEvent]],
    ) -> "torch.Tensor":
        """Encode a batch of variable-length trade sequences.

        Sequences are padded to the length of the longest sequence in the
        batch, a padding mask is built, and the batch is passed through the
        transformer encoder.

        Parameters
        ----------
        event_sequences : list of list of TradeEvent
            Each inner list is one wallet's ordered trade sequence.
            Length must be in [1, max_length].

        Returns
        -------
        torch.Tensor
            Shape ``(batch, embed_dim)`` — one embedding per sequence.

        Raises
        ------
        ValueError
            On invalid inputs (empty sequences, sequences too long, NaN/Inf values).
        """
        if not event_sequences:
            raise ValueError("event_sequences must not be empty.")

        for seq in event_sequences:
            self.validate_sequence(seq, self.max_length)

        max_seq = max(len(s) for s in event_sequences)
        batch_size = len(event_sequences)
        input_dim = _INPUT_BASE_DIM + self.num_pairs

        # Allocate padded batch tensor and padding mask
        x = torch.zeros(batch_size, max_seq, input_dim)
        padding_mask = torch.ones(batch_size, max_seq, dtype=torch.bool)  # True = padding

        for b, seq in enumerate(event_sequences):
            tokens = self.events_to_tensor(seq)  # (seq_len, input_dim)
            x[b, : len(seq), :] = tokens
            padding_mask[b, : len(seq)] = False  # real positions

        return self(x, padding_mask)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str | None = None) -> str:
        """Save model weights and update metrics.json.

        Returns the path the weights were written to.
        """
        model_dir = model_dir or config.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        artifact_path = os.path.join(model_dir, _ARTIFACT_FILENAME)

        torch.save(self.state_dict(), artifact_path)
        sha256 = _sha256_file(artifact_path)

        # Update metrics.json entry
        metrics_path = os.path.join(model_dir, _METRICS_FILENAME)
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        metrics.setdefault("sequence_transformer", {})["artifact_sha256"] = sha256
        metrics["sequence_transformer"]["embed_dim"] = self.embed_dim
        metrics["sequence_transformer"]["num_layers"] = self.num_layers
        metrics["sequence_transformer"]["num_pairs"] = self.num_pairs
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            "TradeSequenceTransformer saved to %s (sha256=%s…)", artifact_path, sha256[:12]
        )
        return artifact_path

    @classmethod
    def load(
        cls,
        model_dir: str | None = None,
        verify_integrity: bool = True,
    ) -> "TradeSequenceTransformer":
        """Load weights from *model_dir* with optional SHA-256 integrity check.

        Falls back gracefully when the artifact is absent (returns an
        untrained model initialised to the current config defaults).
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load TradeSequenceTransformer.")

        model_dir = model_dir or config.MODEL_DIR
        artifact_path = os.path.join(model_dir, _ARTIFACT_FILENAME)

        if not os.path.exists(artifact_path):
            logger.info(
                "No sequence_transformer.pt found in %s — returning untrained model.",
                model_dir,
            )
            return cls()

        if verify_integrity:
            metrics_path = os.path.join(model_dir, _METRICS_FILENAME)
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                expected_sha = (
                    metrics.get("sequence_transformer", {}).get("artifact_sha256")
                )
                if expected_sha:
                    actual_sha = _sha256_file(artifact_path)
                    if actual_sha != expected_sha:
                        raise RuntimeError(
                            f"Sequence transformer integrity check failed: "
                            f"expected sha256={expected_sha}, got {actual_sha}."
                        )
                # Restore architecture params from metrics so model shape matches weights
                st_meta = metrics.get("sequence_transformer", {})
                model = cls(
                    num_pairs=st_meta.get("num_pairs"),
                    embed_dim=st_meta.get("embed_dim"),
                )
            else:
                model = cls()
        else:
            model = cls()

        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        logger.info("TradeSequenceTransformer loaded from %s.", artifact_path)
        return model

    # ------------------------------------------------------------------
    # CPU latency benchmark
    # ------------------------------------------------------------------

    def benchmark_latency(
        self,
        batch_size: int = 32,
        seq_len: int = 64,
        n_runs: int = 50,
    ) -> dict:
        """Measure per-batch CPU inference latency.

        Returns a dict with ``mean_ms``, ``median_ms``, ``p95_ms``.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for latency benchmarking.")

        self.eval()
        input_dim = _INPUT_BASE_DIM + self.num_pairs
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Warm-up
        for _ in range(5):
            with torch.no_grad():
                self(x, mask)

        latencies: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                self(x, mask)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies_np = np.array(latencies)
        result = {
            "mean_ms": float(np.mean(latencies_np)),
            "median_ms": float(np.median(latencies_np)),
            "p95_ms": float(np.percentile(latencies_np, 95)),
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        logger.info(
            "Latency benchmark — mean=%.1f ms, median=%.1f ms, p95=%.1f ms "
            "(batch=%d, seq=%d)",
            result["mean_ms"],
            result["median_ms"],
            result["p95_ms"],
            batch_size,
            seq_len,
        )
        return result


# ---------------------------------------------------------------------------
# Helper: SHA-256 of a file
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public helper: build a sequence embedding for one wallet's trades
# ---------------------------------------------------------------------------

def build_sequence_embedding(
    trades_df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    model: "TradeSequenceTransformer | None",
    pair_vocab: "dict[str, int] | None" = None,
) -> np.ndarray:
    """Convert a wallet's trades DataFrame to a sequence embedding vector.

    This is the bridge between ``detection/feature_engineering.py`` and the
    transformer model.  If *model* is ``None`` or torch is unavailable, an
    all-zeros vector of length ``config.SEQ_MODEL_EMBED_DIM`` is returned as a
    safe fallback so the feature schema stays consistent before the first
    training run.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Must contain columns: ``ledger_close_time``, ``amount``
        (base_amount), and optionally ``base_asset``, ``counter_asset``,
        ``base_account`` (used for direction detection).
    model : TradeSequenceTransformer | None
        Loaded and eval()-mode model.  When ``None``, returns zeros.
    pair_vocab : dict[str, int] | None
        Mapping ``pair_id → index`` for one-hot encoding.  If ``None``,
        all trades are encoded as OOV (index 0).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(embed_dim,)``.
    """
    embed_dim = config.SEQ_MODEL_EMBED_DIM

    if model is None or not _TORCH_AVAILABLE:
        return np.zeros(embed_dim, dtype=np.float32)

    if trades_df is None or len(trades_df) == 0:
        return np.zeros(embed_dim, dtype=np.float32)

    import pandas as pd

    # ---- Build TradeEvent list ----
    df = trades_df.sort_values("ledger_close_time").reset_index(drop=True)

    # Truncate to max_length silently (defensive truncation before validation)
    if len(df) > model.max_length:
        df = df.iloc[-model.max_length:]

    events: list[TradeEvent] = []
    prev_ts: float | None = None

    for _, row in df.iterrows():
        # Log-amount
        amt = float(row.get("amount", row.get("base_amount", 0.0)))
        if amt <= 0 or not math.isfinite(amt):
            amt = 1.0
        log_amt = math.log(amt)
        if not math.isfinite(log_amt):
            log_amt = 0.0

        # Time delta
        ts_val = row["ledger_close_time"]
        if hasattr(ts_val, "timestamp"):
            ts_float = ts_val.timestamp()
        else:
            ts_float = float(ts_val)

        delta = (ts_float - prev_ts) if prev_ts is not None else 0.0
        delta = max(delta, 0.0)
        if not math.isfinite(delta):
            delta = 0.0
        prev_ts = ts_float

        # Pair index
        pair_idx = 0
        if pair_vocab is not None:
            base_code = row.get("base_asset", "")
            ctr_code = row.get("counter_asset", "")
            pair_key = f"{base_code}/{ctr_code}"
            pair_idx = pair_vocab.get(pair_key, 0)

        # Direction: +1 = buy (base_account == wallet), -1 = sell
        direction = 0.0
        if "base_account" in row.index:
            wallet = row.get("wallet", "")
            direction = 1.0 if row["base_account"] == wallet else -1.0

        events.append(TradeEvent(
            log_amount=log_amt,
            time_delta_s=delta,
            pair_index=pair_idx,
            direction=direction,
        ))

    if not events:
        return np.zeros(embed_dim, dtype=np.float32)

    try:
        model.eval()
        with torch.no_grad():
            embedding = model.encode_trades([events])  # (1, embed_dim)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as exc:
        logger.warning("Sequence embedding failed: %s", exc)
        return np.zeros(embed_dim, dtype=np.float32)
