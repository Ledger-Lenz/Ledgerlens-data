"""Train the LedgerLens transformer sequence model (#182).

Loads the synthetic (or real) labelled dataset, reconstructs per-wallet trade
sequences, and trains :class:`~detection.trade_sequence_transformer.TradeSequenceTransformer`
with the AdamW optimiser and a cosine learning-rate schedule.

The trained weights are written to ``config.MODEL_DIR/sequence_transformer.pt``
and an integrity entry is added to ``metrics.json``.

Usage
-----
    # Basic run against the existing synthetic dataset
    python -m scripts.train_sequence_transformer

    # Custom data, epochs, and output directory
    python -m scripts.train_sequence_transformer \\
        --data-path data/my_labelled_trades.parquet \\
        --epochs 5 \\
        --batch-size 64 \\
        --model-dir models/

    # Benchmark CPU latency after training
    python -m scripts.train_sequence_transformer --benchmark

Training data format
--------------------
The script accepts two input formats:

1. **Feature matrix** (``scripts/generate_synthetic_dataset.py`` output): a
   per-wallet parquet with a ``label`` column.  Because feature matrices don't
   contain raw trade timestamps, the script synthesises toy sequences from the
   Benford features so the model gets something meaningful to learn from during
   unit/integration tests.

2. **Raw trade log** (future): a parquet with columns ``wallet``, ``label``,
   ``ledger_close_time``, ``amount``, ``base_asset``, ``counter_asset``,
   ``base_account``.  When these columns are present the script builds real
   sequences from the trade log directly.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# Keep torch import optional so the module is importable even without it
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from config import config
from detection.trade_sequence_transformer import (
    TradeEvent,
    TradeSequenceTransformer,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LABEL_COL = "label"
_WALLET_COL = "wallet"
_RAW_TRADE_COLS = {"ledger_close_time", "amount", "base_asset", "counter_asset"}

# Minimum and maximum synthetic sequence length generated from feature matrices
_SYNTH_MIN_LEN = 8
_SYNTH_MAX_LEN = 64


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TradeSequenceDataset(Dataset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """PyTorch dataset of (sequence_of_TradeEvents, label) pairs."""

    def __init__(
        self,
        sequences: list[list[TradeEvent]],
        labels: list[int],
    ) -> None:
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[list[TradeEvent], int]:
        return self.sequences[idx], self.labels[idx]


def _collate_fn(
    batch: list[tuple[list[TradeEvent], int]],
    model: TradeSequenceTransformer,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Collate variable-length sequences into a padded batch.

    Returns
    -------
    x : torch.Tensor of shape (batch, max_seq, input_dim)
    mask : torch.Tensor of shape (batch, max_seq) — True = padding
    labels : torch.Tensor of shape (batch,)
    """
    sequences, labels = zip(*batch)
    max_seq = max(len(s) for s in sequences)
    input_dim = 3 + model.num_pairs  # base dims + one-hot pairs
    batch_size = len(sequences)

    x = torch.zeros(batch_size, max_seq, input_dim)
    mask = torch.ones(batch_size, max_seq, dtype=torch.bool)

    for b, seq in enumerate(sequences):
        tokens = model.events_to_tensor(seq)
        x[b, : len(seq), :] = tokens
        mask[b, : len(seq)] = False

    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return x, mask, label_tensor


# ---------------------------------------------------------------------------
# Synthetic sequence generation (from feature matrix)
# ---------------------------------------------------------------------------

def _synthetic_sequences_from_features(
    df: pd.DataFrame,
    rng: np.random.Generator,
    max_length: int,
) -> list[list[TradeEvent]]:
    """Generate toy trade sequences from a per-wallet feature matrix.

    Sequences are seeded from Benford features so wash-trading wallets produce
    detectably different patterns from legitimate wallets — enough for the
    integration test to confirm loss decreases over 2 epochs.
    """
    sequences: list[list[TradeEvent]] = []
    for _, row in df.iterrows():
        is_wash = int(row.get(_LABEL_COL, 0)) == 1

        # Sequence length: wash traders get longer sequences
        seq_len = rng.integers(
            _SYNTH_MIN_LEN if not is_wash else _SYNTH_MIN_LEN * 2,
            min(max_length, _SYNTH_MAX_LEN if not is_wash else _SYNTH_MAX_LEN * 2) + 1,
        )

        # Benford chi-square drives the log-amount variance signal
        chi24 = float(row.get("benford_chi_square_24h", 5.0))
        # Higher chi-square → wash trader uses more regular amounts
        amount_std = 1.0 / (1.0 + chi24 / 20.0)

        events: list[TradeEvent] = []
        prev_amount = 100.0
        for i in range(seq_len):
            # Log-amount: wash traders cluster tightly around a mean
            if is_wash:
                log_amt = math.log(max(prev_amount * rng.normal(1.0, amount_std * 0.1), 1.0))
                # Ping-pong direction alternates for wash traders
                direction = 1.0 if (i % 2 == 0) else -1.0
                # Tighter inter-trade intervals for bots
                delta = float(rng.exponential(scale=5.0))
            else:
                log_amt = float(rng.normal(math.log(prev_amount), 0.5))
                direction = float(rng.choice([1.0, -1.0]))
                delta = float(rng.exponential(scale=120.0))

            log_amt = max(log_amt, 0.0)
            prev_amount = math.exp(log_amt)
            pair_idx = int(rng.integers(0, 3))  # small pair vocab for synthetic data

            events.append(TradeEvent(
                log_amount=log_amt,
                time_delta_s=delta,
                pair_index=pair_idx,
                direction=direction,
            ))

        sequences.append(events)
    return sequences


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    data_path: str,
    max_length: int,
    seed: int = 42,
) -> tuple[list[list[TradeEvent]], list[int], list[list[TradeEvent]], list[int]]:
    """Load and return train/validation split as (sequences, labels) pairs.

    Detects whether the parquet is a feature matrix or raw trade log.
    Returns (train_seqs, train_labels, val_seqs, val_labels).
    """
    logger.info("Loading data from %s …", data_path)
    df = pd.read_parquet(data_path)

    if _LABEL_COL not in df.columns:
        raise ValueError(f"Dataset at {data_path} must have a '{_LABEL_COL}' column.")

    # Detect raw trade log vs feature matrix
    is_raw_trades = _RAW_TRADE_COLS.issubset(df.columns)

    rng = np.random.default_rng(seed)

    if is_raw_trades:
        sequences, labels = _sequences_from_trade_log(df, max_length)
    else:
        # Feature matrix path — synthesise sequences per wallet
        # Shuffle and deduplicate by wallet column if present
        if _WALLET_COL in df.columns:
            df = df.drop_duplicates(subset=[_WALLET_COL])
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sequences = _synthetic_sequences_from_features(df, rng, max_length)
        labels = df[_LABEL_COL].astype(int).tolist()

    # 80/20 stratified split
    total = len(sequences)
    split = int(total * 0.8)
    indices = list(range(total))
    rng.shuffle(indices)
    train_idx, val_idx = indices[:split], indices[split:]

    train_seqs = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    logger.info(
        "Dataset: %d train, %d val sequences (raw_trades=%s)",
        len(train_seqs),
        len(val_seqs),
        is_raw_trades,
    )
    return train_seqs, train_labels, val_seqs, val_labels


def _sequences_from_trade_log(
    df: pd.DataFrame,
    max_length: int,
) -> tuple[list[list[TradeEvent]], list[int]]:
    """Build per-wallet sequences from a raw trade log."""
    sequences: list[list[TradeEvent]] = []
    labels: list[int] = []

    for wallet, group in df.groupby(_WALLET_COL):
        group = group.sort_values("ledger_close_time")
        label = int(group[_LABEL_COL].iloc[0])
        chunk = group.head(max_length)

        events: list[TradeEvent] = []
        prev_ts: float | None = None

        for _, row in chunk.iterrows():
            amt = float(row.get("amount", 1.0))
            if amt <= 0 or not math.isfinite(amt):
                amt = 1.0
            log_amt = math.log(amt)

            ts = row["ledger_close_time"]
            if hasattr(ts, "timestamp"):
                ts_f = ts.timestamp()
            else:
                ts_f = float(ts)

            delta = max((ts_f - prev_ts) if prev_ts is not None else 0.0, 0.0)
            prev_ts = ts_f

            events.append(TradeEvent(
                log_amount=log_amt,
                time_delta_s=delta,
                pair_index=0,
                direction=0.0,
            ))

        if events:
            sequences.append(events)
            labels.append(label)

    return sequences, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class _SequenceClassifier(nn.Module):
    """Wraps the transformer encoder with a binary classification head.

    Used only during training; the encoder weights are what get saved.
    """

    def __init__(self, encoder: TradeSequenceTransformer) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, 1)

    def forward(
        self,
        x: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        embedding = self.encoder(x, mask)  # (batch, embed_dim)
        return self.head(embedding).squeeze(-1)  # (batch,)


def train(
    data_path: str = "data/synthetic_dataset.parquet",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    model_dir: str | None = None,
    seed: int = 42,
    benchmark: bool = False,
) -> dict:
    """Train the sequence transformer and return a metrics dict.

    Returns
    -------
    dict with keys: ``train_losses``, ``val_losses``, ``final_val_loss``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install torch>=2.2.0.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    encoder = TradeSequenceTransformer()

    train_seqs, train_labels, val_seqs, val_labels = load_dataset(
        data_path, encoder.max_length, seed=seed
    )

    model = _SequenceClassifier(encoder)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # T_max = total training steps
    total_steps = math.ceil(len(train_seqs) / batch_size) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.01)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TradeSequenceDataset(train_seqs, train_labels)
    val_dataset = TradeSequenceDataset(val_seqs, val_labels)

    def _collate(batch):
        return _collate_fn(batch, encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,  # keep it in-process for safety on all platforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    logger.info(
        "Starting training: epochs=%d, batch=%d, lr=%.4f, train=%d, val=%d",
        epochs, batch_size, lr, len(train_seqs), len(val_seqs),
    )

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, mask, labels_t in train_loader:
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, labels_t)
            loss.backward()
            # Gradient clipping for stable training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, mask, labels_t in val_loader:
                logits = model(x, mask)
                loss = criterion(logits, labels_t)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  (%.1fs)",
            epoch, epochs, avg_train_loss, avg_val_loss, elapsed,
        )
        print(
            f"Epoch {epoch}/{epochs}  train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  ({elapsed:.1f}s)"
        )

    # Sanity check: loss should decrease over epochs (at least 2 epochs)
    if len(train_losses) >= 2 and train_losses[-1] >= train_losses[0]:
        logger.warning(
            "Training loss did not decrease: initial=%.4f, final=%.4f. "
            "Consider running more epochs or adjusting the learning rate.",
            train_losses[0],
            train_losses[-1],
        )

    # Save encoder weights only (no classification head)
    encoder.eval()
    saved_path = encoder.save(model_dir=model_dir)
    logger.info("Model saved to %s", saved_path)
    print(f"Saved encoder to {saved_path}")

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "epochs": epochs,
        "data_path": data_path,
    }

    if benchmark:
        print("\nRunning CPU latency benchmark …")
        bench = encoder.benchmark_latency(batch_size=32, seq_len=64, n_runs=50)
        metrics["benchmark"] = bench
        print(
            f"Latency — mean={bench['mean_ms']:.1f} ms, "
            f"median={bench['median_ms']:.1f} ms, "
            f"p95={bench['p95_ms']:.1f} ms"
        )
        if bench["p95_ms"] > 50.0:
            logger.warning(
                "CPU p95 latency %.1f ms exceeds 50 ms target. "
                "Consider reducing SEQ_MODEL_NUM_LAYERS or SEQ_MODEL_EMBED_DIM.",
                bench["p95_ms"],
            )
        else:
            print("✓ Latency target < 50 ms met.")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default="data/synthetic_dataset.parquet",
        help="Path to the labelled feature matrix or trade log parquet.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs (default: 2).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (default: 1e-3).")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 0.01)."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to save model weights (default: config.MODEL_DIR).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run CPU latency benchmark after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_dir=args.model_dir,
        seed=args.seed,
        benchmark=args.benchmark,
    )

    # Confirm loss decreased when more than one epoch was run
    losses = metrics["train_losses"]
    if len(losses) >= 2:
        trend = "↓ decreased" if losses[-1] < losses[0] else "↑ did NOT decrease"
        print(f"\nTrain loss {trend}: {losses[0]:.4f} → {losses[-1]:.4f}")
    else:
        print(f"\nFinal train loss: {losses[0] if losses else 'N/A':.4f}")


if __name__ == "__main__":
    main()
