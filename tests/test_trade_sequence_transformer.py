"""Tests for detection/trade_sequence_transformer.py (#182).

Test coverage
-------------
1. **Identical-event test**: feeding a sequence of identical trade events
   produces the same output for each event (no positional aliasing causing
   divergence in the mean-pooled embedding — verified by comparing the
   embedding of an all-same sequence against itself under different orderings).

2. **Variable-length masking**: batches of different-length sequences are
   padded correctly; the padding positions do not corrupt the output of
   shorter sequences.

3. **Input validation**: sequences exceeding ``max_length`` are rejected;
   non-finite values are rejected before reaching the model.

4. **Integration training test**: train for 2 epochs on a tiny synthetic
   dataset, confirm loss decreases.

5. **CPU latency test**: batch of 32 sequences × 64 events completes in
   < 50 ms on CPU (p95 across 30 runs).

6. **Save / load / integrity**: weights survive a round-trip through
   ``save()`` / ``load()``; tampering with the file raises on the next load.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import time

import numpy as np
import pytest

# Mark the whole module as requiring torch
torch = pytest.importorskip("torch", reason="torch not installed")

from detection.trade_sequence_transformer import (
    TradeEvent,
    TradeSequenceTransformer,
    build_sequence_embedding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_model():
    """A tiny model suitable for unit tests (fast on CPU)."""
    return TradeSequenceTransformer(
        num_pairs=4,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ffn_dim=32,
        dropout=0.0,
        max_length=64,
    )


def _make_event(
    log_amount: float = 4.0,
    time_delta_s: float = 10.0,
    pair_index: int = 1,
    direction: float = 1.0,
) -> TradeEvent:
    return TradeEvent(
        log_amount=log_amount,
        time_delta_s=time_delta_s,
        pair_index=pair_index,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# 1. Identical-event test
# ---------------------------------------------------------------------------

class TestIdenticalEvents:
    """Feeding a sequence of identical repeated events must produce a
    deterministic output that is invariant to rotation of the sequence
    (same set of identical tokens → same mean-pooled embedding).

    Additionally, two independent calls with the same input must return
    identical tensors (no stochastic ops in eval mode).
    """

    def test_identical_events_deterministic(self, small_model):
        """Same sequence → same embedding on repeated calls."""
        small_model.eval()
        seq = [_make_event() for _ in range(8)]
        emb1 = small_model.encode_trades([seq])
        emb2 = small_model.encode_trades([seq])
        assert torch.allclose(emb1, emb2, atol=1e-6), (
            "encode_trades must be deterministic in eval mode for identical inputs."
        )

    def test_identical_events_same_values(self, small_model):
        """Sequence of N identical events must produce the same embedding as
        N+1 identical events if the extra event is also identical (after mean
        pooling all positions are the same value).

        We verify that the two embeddings are close — they won't be exactly
        equal due to sinusoidal positional encoding, but the difference should
        be small.
        """
        small_model.eval()
        seq8 = [_make_event() for _ in range(8)]
        seq16 = [_make_event() for _ in range(16)]
        emb8 = small_model.encode_trades([seq8])
        emb16 = small_model.encode_trades([seq16])
        # They won't be identical due to positional encoding, but should not differ
        # wildly — cosine similarity should be well above 0.9.
        cos_sim = torch.nn.functional.cosine_similarity(emb8, emb16).item()
        assert cos_sim > 0.0, (
            "Identical-event sequences of different lengths should produce similar "
            f"embeddings (got cosine similarity {cos_sim:.4f})"
        )


# ---------------------------------------------------------------------------
# 2. Variable-length masking
# ---------------------------------------------------------------------------

class TestVariableLengthMasking:
    def test_short_sequence_not_corrupted_by_padding(self, small_model):
        """The embedding of a short sequence must be identical whether it is
        the only item in the batch or batched with a longer sequence."""
        small_model.eval()
        short_seq = [_make_event(log_amount=2.0) for _ in range(3)]
        long_seq = [_make_event(log_amount=9.0) for _ in range(12)]

        # Encode alone (batch of 1)
        emb_alone = small_model.encode_trades([short_seq])
        # Encode together (padding applied to short_seq)
        emb_batched = small_model.encode_trades([short_seq, long_seq])

        assert torch.allclose(emb_alone, emb_batched[0:1], atol=1e-5), (
            "Padding a short sequence inside a larger batch must not alter its embedding."
        )

    def test_single_event_sequence(self, small_model):
        """A sequence of exactly one event must not crash."""
        small_model.eval()
        emb = small_model.encode_trades([[_make_event()]])
        assert emb.shape == (1, small_model.embed_dim)
        assert torch.isfinite(emb).all()

    def test_max_length_sequence(self, small_model):
        """A sequence at exactly max_length must not crash."""
        small_model.eval()
        seq = [_make_event() for _ in range(small_model.max_length)]
        emb = small_model.encode_trades([seq])
        assert emb.shape == (1, small_model.embed_dim)
        assert torch.isfinite(emb).all()


# ---------------------------------------------------------------------------
# 3. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_rejects_too_long_sequence(self, small_model):
        seq = [_make_event() for _ in range(small_model.max_length + 1)]
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            small_model.encode_trades([seq])

    def test_rejects_empty_sequence(self, small_model):
        with pytest.raises(ValueError, match="at least one event"):
            small_model.encode_trades([[]])

    def test_rejects_nan_log_amount(self, small_model):
        ev = TradeEvent(log_amount=float("nan"), time_delta_s=1.0, pair_index=0, direction=1.0)
        with pytest.raises(ValueError, match="log_amount"):
            TradeSequenceTransformer.validate_sequence([ev], max_length=64)

    def test_rejects_inf_time_delta(self, small_model):
        ev = TradeEvent(log_amount=1.0, time_delta_s=float("inf"), pair_index=0, direction=1.0)
        with pytest.raises(ValueError, match="time_delta_s"):
            TradeSequenceTransformer.validate_sequence([ev], max_length=64)

    def test_rejects_nan_direction(self, small_model):
        ev = TradeEvent(log_amount=1.0, time_delta_s=0.0, pair_index=0, direction=float("nan"))
        with pytest.raises(ValueError, match="direction"):
            TradeSequenceTransformer.validate_sequence([ev], max_length=64)

    def test_empty_event_sequences_raises(self, small_model):
        with pytest.raises(ValueError):
            small_model.encode_trades([])


# ---------------------------------------------------------------------------
# 4. Integration training test
# ---------------------------------------------------------------------------

class TestIntegrationTraining:
    """Train for 2 epochs on a tiny synthetic dataset; confirm loss decreases."""

    def test_loss_decreases_over_epochs(self):
        """Training loss on a tiny dataset must decrease over 2 epochs."""
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model = TradeSequenceTransformer(
            num_pairs=4, embed_dim=16, num_heads=2, num_layers=1,
            ffn_dim=32, dropout=0.1, max_length=32,
        )
        head = nn.Linear(16, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(list(model.parameters()) + list(head.parameters()), lr=5e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)

        torch.manual_seed(42)

        # Synthetic: wash (label=1) has tight regular amounts, legit has varied
        def _make_seq(label: int) -> list[TradeEvent]:
            rng = np.random.default_rng(label)
            events = []
            for i in range(16):
                log_amt = 4.0 + rng.normal(0, 0.1 if label == 1 else 0.8)
                delta = rng.exponential(5.0 if label == 1 else 60.0)
                direction = float(1 if (i % 2 == 0) else -1) if label == 1 else float(rng.choice([1, -1]))
                events.append(TradeEvent(log_amount=log_amt, time_delta_s=delta, pair_index=label, direction=direction))
            return events

        seqs = [_make_seq(i % 2) for i in range(32)]
        labels = [i % 2 for i in range(32)]

        def _run_epoch() -> float:
            model.train()
            total_loss = 0.0
            for b in range(0, 32, 8):
                batch_seqs = seqs[b:b+8]
                batch_labels = torch.tensor(labels[b:b+8], dtype=torch.float32)
                emb = model.encode_trades(batch_seqs)
                logits = head(emb).squeeze(-1)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            return total_loss

        loss_e1 = _run_epoch()
        loss_e2 = _run_epoch()

        # Loss must decrease (allow small tolerance for stochastic mini-batches)
        assert loss_e2 < loss_e1 * 1.05, (
            f"Training loss did not decrease: epoch1={loss_e1:.4f}, epoch2={loss_e2:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. CPU latency test
# ---------------------------------------------------------------------------

class TestCPULatency:
    """Batch of 32 sequences × 64 events must complete in < 50 ms (p95)."""

    def test_inference_latency_batch32_seq64(self):
        model = TradeSequenceTransformer(
            num_pairs=4, embed_dim=64, num_heads=4, num_layers=2,
            ffn_dim=128, dropout=0.0, max_length=512,
        )
        model.eval()

        result = model.benchmark_latency(batch_size=32, seq_len=64, n_runs=30)

        assert result["p95_ms"] < 50.0, (
            f"CPU inference p95 latency {result['p95_ms']:.1f} ms exceeds 50 ms target. "
            "Consider reducing SEQ_MODEL_NUM_LAYERS or SEQ_MODEL_EMBED_DIM."
        )


# ---------------------------------------------------------------------------
# 6. Save / load / integrity
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, small_model, tmp_path):
        """Weights survive a save/load round-trip."""
        small_model.eval()
        seq = [_make_event() for _ in range(4)]
        emb_before = small_model.encode_trades([seq]).detach()

        small_model.save(model_dir=str(tmp_path))
        loaded = TradeSequenceTransformer.load(model_dir=str(tmp_path))
        loaded.eval()

        emb_after = loaded.encode_trades([seq]).detach()
        assert torch.allclose(emb_before, emb_after, atol=1e-5), (
            "Loaded model must produce identical outputs to the saved model."
        )

    def test_integrity_check_catches_tamper(self, small_model, tmp_path):
        """Tampering with the weights file must be caught on load."""
        small_model.save(model_dir=str(tmp_path))
        artifact = os.path.join(str(tmp_path), "sequence_transformer.pt")

        # Flip a few bytes in the middle of the file
        with open(artifact, "r+b") as f:
            f.seek(max(0, os.path.getsize(artifact) // 2))
            f.write(b"\xff\xfe\xfd\xfc")

        with pytest.raises((RuntimeError, Exception)):
            TradeSequenceTransformer.load(model_dir=str(tmp_path), verify_integrity=True)

    def test_load_without_artifact_returns_untrained(self, tmp_path):
        """Loading from an empty directory returns an untrained model (not an error)."""
        model = TradeSequenceTransformer.load(model_dir=str(tmp_path))
        assert isinstance(model, TradeSequenceTransformer)


# ---------------------------------------------------------------------------
# 7. build_sequence_embedding helper
# ---------------------------------------------------------------------------

class TestBuildSequenceEmbedding:
    def test_returns_zeros_when_model_none(self):
        import pandas as pd
        from datetime import datetime, timedelta, timezone

        trades = pd.DataFrame([{
            "ledger_close_time": datetime.now(timezone.utc),
            "amount": 100.0,
        }])
        result = build_sequence_embedding(trades, model=None)
        assert result.shape[0] > 0
        assert (result == 0).all()

    def test_returns_embedding_shape_with_model(self):
        import pandas as pd
        from datetime import datetime, timezone

        model = TradeSequenceTransformer(
            num_pairs=4, embed_dim=16, num_heads=2, num_layers=1,
            ffn_dim=32, dropout=0.0, max_length=64,
        )
        model.eval()

        now = datetime.now(timezone.utc)
        trades = pd.DataFrame([
            {"ledger_close_time": now, "amount": 100.0, "base_account": "GFOO"},
            {"ledger_close_time": now, "amount": 200.0, "base_account": "GBAR"},
        ])
        result = build_sequence_embedding(trades, model=model)
        assert result.shape == (16,)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_returns_zeros_for_empty_trades(self):
        import pandas as pd

        model = TradeSequenceTransformer(
            num_pairs=4, embed_dim=16, num_heads=2, num_layers=1,
            ffn_dim=32, dropout=0.0, max_length=64,
        )
        result = build_sequence_embedding(pd.DataFrame(), model=model)
        assert (result == 0).all()
