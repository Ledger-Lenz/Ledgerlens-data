"""Domain-Adversarial Neural Network adapter: synthetic→real distribution shift (issue #215).

Wraps :class:`detection.dann_encoder.DANNEncoder` with a training loop that
uses a ``domain`` column to distinguish synthetic (domain=0) from real Stellar
DEX data (domain=1), producing domain-invariant embeddings so the label
classifier generalises beyond the simulator distribution.

CLI usage::

    python -m detection.dann_adapter \\
        --synthetic data/synthetic_dataset.parquet \\
        --real      data/real_unlabelled.parquet   \\
        --output    models/dann_adapter.pt

Checkpoint keys: ``model_state``, ``input_dim``, ``hidden_dim``,
``embedding_dim``, ``auc_roc``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from detection.dann_encoder import DANNEncoder, DANNTrainingReport, _auc_roc
from utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_COL = "is_wash_trade"
_DEFAULT_HIDDEN = 128
_DEFAULT_EMBED = 64
_DEFAULT_EPOCHS = 50
_DEFAULT_LR = 1e-3
_DEFAULT_BATCH = 64
_DEFAULT_LAMBDA_MAX = 1.0
_DEFAULT_SEED = 42

# Columns that are never features
_NON_FEATURE = {_LABEL_COL, "domain", "wallet_id", "pair_id", "timestamp"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])
    ]


def _load_and_tag(synthetic_path: str, real_path: str) -> pd.DataFrame:
    syn = pd.read_parquet(synthetic_path)
    syn["domain"] = 0.0
    real = pd.read_parquet(real_path)
    real["domain"] = 1.0
    if _LABEL_COL not in real.columns:
        real[_LABEL_COL] = -1.0  # unlabelled — excluded from label loss
    return pd.concat([syn, real], ignore_index=True)


def _lambda_schedule(epoch: int, total_epochs: int, lambda_max: float) -> float:
    """Smooth 0→lambda_max schedule following the GRL paper (Ganin et al., 2016)."""
    p = epoch / max(total_epochs - 1, 1)
    return float(lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


def train_dann_adapter(
    synthetic_path: str,
    real_path: str,
    *,
    hidden_dim: int = _DEFAULT_HIDDEN,
    embedding_dim: int = _DEFAULT_EMBED,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH,
    learning_rate: float = _DEFAULT_LR,
    lambda_max: float = _DEFAULT_LAMBDA_MAX,
    seed: int = _DEFAULT_SEED,
    device: str = "cpu",
) -> DANNTrainingReport:
    """Train a DANN adapter aligning synthetic→real feature distributions.

    Labelled rows (y >= 0) contribute to the label-classification loss.
    All rows contribute to the domain-adversarial loss.

    Returns:
        :class:`~detection.dann_encoder.DANNTrainingReport` with trained model
        and AUC-ROC evaluated on the held-out test split.
    """
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    _device = torch.device(device)

    df = _load_and_tag(synthetic_path, real_path)
    feat_cols = _feature_cols(df)
    if not feat_cols:
        raise ValueError("No numeric feature columns found in the loaded data.")

    df[feat_cols] = df[feat_cols].fillna(0.0)
    x_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df[_LABEL_COL].to_numpy(dtype=np.float32)
    d_all = df["domain"].to_numpy(dtype=np.float32)

    idx = np_rng.permutation(len(x_all))
    n_train = max(2, int(round(len(idx) * 0.8)))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def _loader(indices: np.ndarray) -> DataLoader:
        xt = torch.from_numpy(x_all[indices])
        yt = torch.from_numpy(y_all[indices])
        dt = torch.from_numpy(d_all[indices])
        return DataLoader(TensorDataset(xt, yt, dt), batch_size=batch_size, shuffle=True)

    train_loader = _loader(train_idx)
    test_loader = _loader(test_idx)

    model = DANNEncoder(x_all.shape[1], hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    model = model.to(_device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        lam = _lambda_schedule(epoch, epochs, lambda_max)
        epoch_d_loss = 0.0
        n_batches = 0

        for x_b, y_b, d_b in train_loader:
            x_b, y_b, d_b = x_b.to(_device), y_b.to(_device), d_b.to(_device)
            label_logits, domain_logits, _ = model(x_b, domain_lambda=lam)

            labelled = y_b >= 0
            if labelled.any():
                label_loss = F.binary_cross_entropy_with_logits(
                    label_logits.squeeze(-1)[labelled], y_b[labelled]
                )
            else:
                label_loss = torch.tensor(0.0, device=_device)

            domain_loss = F.binary_cross_entropy_with_logits(
                domain_logits.squeeze(-1), d_b
            )
            loss = label_loss + 0.5 * domain_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_d_loss += domain_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "DANN adapter epoch %d/%d  lambda=%.3f  domain_loss=%.4f",
                epoch + 1, epochs, lam, epoch_d_loss / max(n_batches, 1),
            )

    auc = _auc_roc(model, test_loader, _device)
    logger.info("DANN adapter done — test AUC-ROC: %.4f", auc)
    return DANNTrainingReport(
        model=model,
        auc_roc=auc,
        baseline_auc_roc=float("nan"),
        auc_roc_degradation=float("nan"),
        membership_inference_success_rate=float("nan"),
    )


def save_checkpoint(report: DANNTrainingReport, output_path: str) -> None:
    """Save model state and metadata to *output_path* (``*.pt``)."""
    m = report.model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": m.state_dict(),
            "input_dim": m.feature_extractor[0].in_features,
            "hidden_dim": m.feature_extractor[0].out_features,
            "embedding_dim": m.feature_extractor[2].out_features,
            "auc_roc": report.auc_roc,
        },
        output_path,
    )
    logger.info("DANN checkpoint saved → %s  (AUC-ROC=%.4f)", output_path, report.auc_roc)


def load_checkpoint(path: str, device: str = "cpu") -> DANNEncoder:
    """Load a previously saved checkpoint and return a model in eval mode."""
    ckpt = torch.load(path, map_location=device)
    model = DANNEncoder(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        embedding_dim=ckpt["embedding_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DANN domain adapter (synthetic → real Stellar DEX)."
    )
    parser.add_argument("--synthetic", required=True, help="Synthetic Parquet dataset")
    parser.add_argument("--real", required=True, help="Real Parquet dataset (labels optional)")
    parser.add_argument("--output", default="models/dann_adapter.pt")
    parser.add_argument("--hidden-dim", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--embedding-dim", type=int, default=_DEFAULT_EMBED)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--lambda-max", type=float, default=_DEFAULT_LAMBDA_MAX)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = train_dann_adapter(
        synthetic_path=args.synthetic,
        real_path=args.real,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_max=args.lambda_max,
        seed=args.seed,
        device=args.device,
    )
    save_checkpoint(report, args.output)
    print(json.dumps({"auc_roc": round(report.auc_roc, 4), "output": args.output}))


if __name__ == "__main__":
    _cli()
