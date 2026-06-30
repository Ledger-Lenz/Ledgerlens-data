"""SimCLR contrastive pre-training with domain-aware negative mining.

Extends the original random-augmentation SimCLR loop with:
  - Hard negative mining via ``HardNegativeMiner`` (FAISS HNSW, O(k log n))
  - Ring-based positive pairs from labelled wash-trade data
  - A curriculum warm-up that ramps from easy (random) to hard negatives over
    ``CONTRASTIVE_CURRICULUM_EPOCHS`` epochs (default 5)

Backwards compatibility
-----------------------
The original ``pretrain()`` function signature is preserved.  Pass
``use_domain_sampling=False`` to reproduce the original random-augmentation
behaviour exactly.

Usage
-----
    # Domain-aware pre-training (recommended):
    python -m detection.contrastive.pretrain \
        --data data/synthetic_dataset.parquet \
        --epochs 20 \
        --domain-sampling

    # Original random-augmentation mode:
    python -m detection.contrastive.pretrain \
        --data data/synthetic_dataset.parquet \
        --epochs 20
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

import numpy as np
import pandas as _pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from detection.contrastive.augmentations import augment_trade_sequence
from detection.contrastive.encoder import TransactionEncoder
from detection.contrastive.negative_miner import (
    CONTRASTIVE_CURRICULUM_EPOCHS,
    EVENT_HMAC_SECRET,
    HardNegativeMiner,
    RingRegistry,
    _hash_wallet,
)
from detection.contrastive.simclr import NTXentLoss
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class UnlabeledWalletDataset(Dataset):
    """Dataset for unlabeled wallets (random-augmentation path, unchanged)."""

    def __init__(self, wallets_trades_list: list) -> None:
        self.wallets_trades_list = wallets_trades_list

    def __len__(self) -> int:
        return len(self.wallets_trades_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        df_trades = self.wallets_trades_list[idx]
        view1 = augment_trade_sequence(df_trades)  # noqa: F841 — kept for side effects
        view2 = augment_trade_sequence(df_trades)  # noqa: F841
        # Feature aggregation placeholder (unchanged from original)
        features_1 = np.random.randn(50).astype(np.float32)
        features_2 = np.random.randn(50).astype(np.float32)
        return features_1, features_2


class LabeledFeatureDataset(Dataset):
    """Dataset backed by a pre-computed (n, feature_dim) feature matrix.

    Used in domain-aware mode where the synthetic dataset has already been
    processed through ``build_feature_matrix``.

    Parameters
    ----------
    features:
        (n, dim) float32 array.
    labels:
        (n,) int array — 1 = wash trade, 0 = clean.
    wallet_ids:
        (n,) list of raw wallet G-addresses (hashed internally).
    rings:
        Optional list of raw-address ring groups for positive-pair construction.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        wallet_ids: list[str] | None = None,
        rings: list[list[str]] | None = None,
    ) -> None:
        self.features = torch.tensor(np.asarray(features, dtype=np.float32))
        self.labels = np.asarray(labels, dtype=np.int64)
        # Hash wallet IDs immediately — raw addresses not retained
        self.hashed_ids = (
            [_hash_wallet(w) for w in wallet_ids]
            if wallet_ids is not None
            else [str(i) for i in range(len(features))]
        )
        self.ring_registry = (
            RingRegistry.from_rings(rings) if rings else RingRegistry()
        )
        self.wash_mask = self.labels == 1
        self.clean_mask = self.labels == 0

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        return self.features[idx], int(self.labels[idx]), self.hashed_ids[idx]

    # Convenience accessors
    @property
    def wash_features(self) -> torch.Tensor:
        return self.features[self.wash_mask]

    @property
    def clean_features(self) -> torch.Tensor:
        return self.features[self.clean_mask]

    @property
    def wash_hashed_ids(self) -> list[str]:
        return [h for h, m in zip(self.hashed_ids, self.wash_mask) if m]


# ---------------------------------------------------------------------------
# Domain-aware collate & loss helpers
# ---------------------------------------------------------------------------


def _domain_aware_loss(
    encoder: TransactionEncoder,
    wash_batch: torch.Tensor,
    clean_pool: torch.Tensor,
    miner: HardNegativeMiner,
    hashed_wash_ids: list[str],
    epoch: int,
    ntxent: NTXentLoss,
    device: str,
    n_negatives: int = 4,
) -> torch.Tensor:
    """Compute contrastive loss with hard negatives and ring positives.

    Strategy
    --------
    For each wash-trade anchor we form positive pairs two ways:
      1. Standard SimCLR augmentation view (always included).
      2. Ring partner if present in the batch (injected as extra positives).

    Negatives come from the miner (curriculum mix of hard + random clean
    wallets).  We extend the NT-Xent batch so that ring pairs contribute
    additional gradient signal beyond the standard 2N formulation.
    """
    wash_batch = wash_batch.to(device)
    clean_pool = clean_pool.to(device)

    # Standard SimCLR views via feature noise (light augmentation in feature space)
    noise = torch.randn_like(wash_batch) * 0.01
    view1 = wash_batch
    view2 = wash_batch + noise

    _, z1 = encoder(view1)
    _, z2 = encoder(view2)

    # Base NT-Xent loss on the two views
    base_loss = ntxent(z1, z2)

    # Mine hard negatives — use detached embeddings for index queries
    with torch.no_grad():
        h_wash, _ = encoder(wash_batch)
        h_clean, _ = encoder(clean_pool)

    neg_indices = miner.mine_negatives(
        h_wash.cpu().numpy(), n_negatives=n_negatives, epoch=epoch
    )

    # Ring-positive auxiliary loss: pull ring partners together
    ring_pairs = miner.get_ring_positives(hashed_wash_ids)
    ring_loss = torch.tensor(0.0, device=device)
    if ring_pairs:
        for i, j in ring_pairs:
            zi = F.normalize(z1[i : i + 1], dim=1)
            zj = F.normalize(z1[j : j + 1], dim=1)
            # Cosine similarity — maximise (push loss)
            sim = (zi * zj).sum()
            ring_loss = ring_loss + (1.0 - sim)
        ring_loss = ring_loss / len(ring_pairs)

    # Hard-negative repulsion: push anchors away from hard negatives
    neg_vecs_list = [h_clean[neg_indices[:, k]] for k in range(n_negatives)]
    neg_loss = torch.tensor(0.0, device=device)
    if neg_vecs_list:
        h_anchor_norm = F.normalize(h_wash, dim=1)
        for neg_vecs in neg_vecs_list:
            neg_vecs_norm = F.normalize(neg_vecs.to(device), dim=1)
            sims = (h_anchor_norm * neg_vecs_norm).sum(dim=1)
            # Encourage large margin: penalise if similarity > 0
            neg_loss = neg_loss + F.relu(sims + 0.1).mean()
        neg_loss = neg_loss / n_negatives

    total = base_loss + 0.5 * ring_loss + 0.5 * neg_loss
    return total


# ---------------------------------------------------------------------------
# Public pre-training API
# ---------------------------------------------------------------------------


def pretrain(
    dataset: Any,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_domain_sampling: bool = False,
    curriculum_epochs: int = CONTRASTIVE_CURRICULUM_EPOCHS,
) -> TransactionEncoder:
    """Pre-train the TransactionEncoder with SimCLR.

    Parameters
    ----------
    dataset:
        A ``LabeledFeatureDataset`` when ``use_domain_sampling=True``;
        an ``UnlabeledWalletDataset`` otherwise.
    epochs:
        Total training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam learning rate.
    device:
        PyTorch device string.
    use_domain_sampling:
        If True, activates domain-aware hard negative mining and ring
        positives.  Requires *dataset* to be a ``LabeledFeatureDataset``.
    curriculum_epochs:
        Warm-up length for the hard-negative curriculum (only used when
        ``use_domain_sampling=True``).
    """
    if use_domain_sampling:
        return _pretrain_domain_aware(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            curriculum_epochs=curriculum_epochs,
        )

    # ---- Original random-augmentation path (unchanged) ----
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    input_dim = 50
    encoder = TransactionEncoder(input_dim=input_dim).to(device)
    criterion = NTXentLoss(temperature=0.5).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    logger.info("Starting random-augmentation pre-training for %d epochs on %s", epochs, device)
    encoder.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for view1, view2 in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            _, z1 = encoder(view1)
            _, z2 = encoder(view2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("Epoch [%d/%d] Loss: %.4f", epoch + 1, epochs, total_loss / max(len(dataloader), 1))

    return encoder


def _pretrain_domain_aware(
    dataset: LabeledFeatureDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    curriculum_epochs: int,
) -> TransactionEncoder:
    """Domain-aware SimCLR training loop."""
    if not isinstance(dataset, LabeledFeatureDataset):
        raise TypeError("use_domain_sampling=True requires a LabeledFeatureDataset")

    input_dim = dataset.features.shape[1]
    encoder = TransactionEncoder(input_dim=input_dim).to(device)
    criterion = NTXentLoss(temperature=0.5).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    miner = HardNegativeMiner(
        embedding_dim=256,  # matches TransactionEncoder hidden_dim
        curriculum_epochs=curriculum_epochs,
    )
    miner.set_ring_registry(dataset.ring_registry)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    logger.info(
        "Starting domain-aware pre-training for %d epochs (curriculum_epochs=%d) on %s",
        epochs, curriculum_epochs, device,
    )

    for epoch in range(epochs):
        encoder.train()

        # Refresh the clean-wallet ANN index once per epoch using detached embeddings
        clean_feats = dataset.clean_features.to(device)
        with torch.no_grad():
            h_clean_all, _ = encoder(clean_feats)
        miner.build_clean_index(h_clean_all.cpu().numpy())

        total_loss = 0.0
        n_batches = 0

        for features, labels, hashed_ids in dataloader:
            features = features.to(device)
            labels_np = labels.numpy()
            hashed_ids_list = list(hashed_ids)

            wash_mask = labels_np == 1
            clean_mask = labels_np == 0

            if wash_mask.sum() == 0:
                # No wash samples in this batch — fall back to standard NT-Xent
                optimizer.zero_grad()
                noise = torch.randn_like(features) * 0.01
                _, z1 = encoder(features)
                _, z2 = encoder(features + noise)
                loss = criterion(z1, z2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                continue

            wash_feats = features[wash_mask]
            # Use entire clean pool (refreshed above) for negative mining
            clean_pool = clean_feats

            wash_ids = [hashed_ids_list[i] for i, m in enumerate(wash_mask) if m]

            optimizer.zero_grad()
            loss = _domain_aware_loss(
                encoder=encoder,
                wash_batch=wash_feats,
                clean_pool=clean_pool,
                miner=miner,
                hashed_wash_ids=wash_ids,
                epoch=epoch,
                ntxent=criterion,
                device=device,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        hard_frac = miner.hard_fraction(epoch)
        logger.info(
            "Epoch [%d/%d] Loss: %.4f  hard_negative_fraction: %.2f",
            epoch + 1, epochs, avg, hard_frac,
        )

    return encoder


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark_auc(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
    wallet_ids: list[str] | None = None,
    rings: list[list[str]] | None = None,
) -> dict[str, float]:
    """Train both random and domain-aware encoders, return AUC comparison.

    Uses a linear probe (logistic regression on frozen encoder embeddings)
    to measure downstream classification quality.

    Returns
    -------
    dict with keys ``"random_auc"`` and ``"domain_auc"``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    ids_train = wallet_ids[: len(X_train)] if wallet_ids is not None else None

    def _extract_embeddings(enc: TransactionEncoder, X: np.ndarray) -> np.ndarray:
        enc.eval()
        with torch.no_grad():
            h, _ = enc(torch.tensor(X, dtype=torch.float32).to(device))
        return h.cpu().numpy()

    def _auc(enc: TransactionEncoder) -> float:
        h_train = _extract_embeddings(enc, X_train)
        h_test = _extract_embeddings(enc, X_test)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(h_train, y_train)
        proba = clf.predict_proba(h_test)[:, 1]
        return float(roc_auc_score(y_test, proba))

    # --- Random augmentation baseline ---
    dummy_wallets = [
        _pd.DataFrame({"amount": np.random.rand(10), "timestamp": np.arange(10)})
        for _ in range(len(X_train))
    ]
    rand_dataset = UnlabeledWalletDataset(dummy_wallets)
    # patch feature dim to match X_train
    original_getitem = rand_dataset.__getitem__

    def _patched_getitem(idx):
        return (
            X_train[idx].astype(np.float32),
            X_train[idx].astype(np.float32),
        )

    rand_dataset.__getitem__ = _patched_getitem  # type: ignore[method-assign]
    rand_dataset.__len__ = lambda: len(X_train)  # type: ignore[method-assign]

    rand_enc = pretrain(
        rand_dataset,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        use_domain_sampling=False,
    )
    random_auc = _auc(rand_enc)

    # --- Domain-aware ---
    labeled_ds = LabeledFeatureDataset(
        X_train, y_train, wallet_ids=ids_train, rings=rings
    )
    domain_enc = pretrain(
        labeled_ds,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        use_domain_sampling=True,
    )
    domain_auc = _auc(domain_enc)

    logger.info(
        "Benchmark — random SimCLR AUC: %.4f  domain-aware AUC: %.4f",
        random_auc, domain_auc,
    )
    return {"random_auc": random_auc, "domain_auc": domain_auc}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SimCLR contrastive pre-training")
    parser.add_argument("--data", default=None, help="Path to labelled .parquet dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--domain-sampling", action="store_true", default=False)
    parser.add_argument("--curriculum-epochs", type=int, default=CONTRASTIVE_CURRICULUM_EPOCHS)
    parser.add_argument("--output", default="./models/pretrained_encoder.pt")
    parser.add_argument("--benchmark", action="store_true", default=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data and os.path.exists(args.data):
        df = _pd.read_parquet(args.data)
        feature_cols = [c for c in df.columns if c not in {"wallet", "label", "profile"}]
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        wallet_ids = df["wallet"].tolist() if "wallet" in df.columns else None

        if args.benchmark:
            results = benchmark_auc(X, y, epochs=args.epochs, device=device, wallet_ids=wallet_ids)
            print(f"random_auc={results['random_auc']:.4f}  domain_auc={results['domain_auc']:.4f}")
        else:
            ds: Any
            if args.domain_sampling:
                ds = LabeledFeatureDataset(X, y, wallet_ids=wallet_ids)
            else:
                dummy = [_pd.DataFrame({"amount": np.random.rand(5), "timestamp": np.arange(5)}) for _ in range(len(X))]
                ds = UnlabeledWalletDataset(dummy)
                ds.__getitem__ = lambda idx: (X[idx], X[idx])  # type: ignore[method-assign]
                ds.__len__ = lambda: len(X)  # type: ignore[method-assign]

            enc = pretrain(
                ds, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, device=device,
                use_domain_sampling=args.domain_sampling,
                curriculum_epochs=args.curriculum_epochs,
            )
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            torch.save(enc.state_dict(), args.output)
            logger.info("Saved encoder to %s", args.output)
    else:
        # Dummy smoke-test run
        dummy_wallets = [
            _pd.DataFrame({"amount": np.random.rand(10), "timestamp": np.arange(10)})
            for _ in range(200)
        ]
        ds = UnlabeledWalletDataset(dummy_wallets)
        enc = pretrain(ds, epochs=2, batch_size=32, device=device)
        os.makedirs("./models", exist_ok=True)
        torch.save(enc.state_dict(), "./models/pretrained_encoder.pt")
        logger.info("Dummy pre-training completed.")
