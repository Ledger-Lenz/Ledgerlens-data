"""Deterministic hash-commitment helper for attested risk-score submissions.

V1 uses a reproducible SHA-256 commitment over public trade data, a committed
model version hash, the wallet identifier, and the submitted score. This keeps
the submitter from changing any of the public inputs without invalidating the
commitment.

V2 can swap the commitment helper for a zkVM receipt without changing the
callers that already depend on the receipt shape in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class CommitmentReceipt:
    """Public attestation payload for a score submission."""

    wallet: str
    trade_data_hash: str
    model_version_hash: str
    score: int
    commitment: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZKAttestor:
    """Build and verify deterministic commitments for attested score submissions."""

    def _normalize_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item") and callable(value.item):
            return value.item()
        return value

    def _canonical_records(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []

        ordered = trades.copy()
        ordered = ordered.reindex(sorted(ordered.columns), axis=1)
        records = []
        for row in ordered.to_dict(orient="records"):
            normalized = {key: self._normalize_value(value) for key, value in row.items()}
            records.append(normalized)

        records.sort(
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )
        return records

    def trade_data_hash(self, trades: pd.DataFrame) -> str:
        """Return a stable SHA-256 hash of the public trade set."""
        payload = json.dumps(
            self._canonical_records(trades),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build_commitment(
        self,
        wallet: str,
        trade_data_hash: str,
        model_version_hash: str,
        score: int,
    ) -> str:
        """Return the deterministic commitment for the attested public inputs."""
        payload = json.dumps(
            {
                "wallet": wallet,
                "trade_data_hash": trade_data_hash,
                "model_version_hash": model_version_hash,
                "score": int(score),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def generate_receipt(
        self,
        wallet: str,
        trades: pd.DataFrame,
        score: int,
        model_version_hash: str,
    ) -> CommitmentReceipt:
        """Create the V1 commitment receipt for a score submission."""
        trade_hash = self.trade_data_hash(trades)
        commitment = self.build_commitment(wallet, trade_hash, model_version_hash, score)
        return CommitmentReceipt(
            wallet=wallet,
            trade_data_hash=trade_hash,
            model_version_hash=model_version_hash,
            score=int(score),
            commitment=commitment,
        )

    def verify_receipt(
        self,
        receipt: CommitmentReceipt,
        trades: pd.DataFrame | None = None,
    ) -> bool:
        """Verify that a receipt matches the provided trade data and public inputs."""
        if trades is not None and self.trade_data_hash(trades) != receipt.trade_data_hash:
            return False
        expected = self.build_commitment(
            receipt.wallet,
            receipt.trade_data_hash,
            receipt.model_version_hash,
            receipt.score,
        )
        return expected == receipt.commitment

    def guest_program_interface(self, receipt: CommitmentReceipt) -> dict[str, Any]:
        """Describe the inputs a future zkVM guest would consume in V2."""
        return {
            "inputs": {
                "wallet": receipt.wallet,
                "trade_data_hash": receipt.trade_data_hash,
                "model_version_hash": receipt.model_version_hash,
                "score": receipt.score,
            },
            "public_outputs": {
                "commitment": receipt.commitment,
                "trade_data_hash": receipt.trade_data_hash,
                "model_version_hash": receipt.model_version_hash,
                "score": receipt.score,
            },
        }