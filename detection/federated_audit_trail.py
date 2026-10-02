"""Audit trail for federated training rounds linking model updates to participants.

Maintains forensic records of federated aggregation rounds, including participant
counts, aggregation strategy, resulting model versions, and other metadata needed
for post-incident review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FederatedRoundAuditRecord:
    """Audit record for a single federated training round."""

    round_id: str
    timestamp: str
    participant_count: int
    aggregation_strategy: str
    model_version: str
    model_hash: str
    update_norm: float
    privacy_budget_used: float
    status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FederatedRoundAuditTrail:
    """Audit trail for federated training rounds."""

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize federated round audit trail.

        Args:
            storage_path: Path to store audit trail records (optional).
        """
        self.storage_path = storage_path
        self.records: list[FederatedRoundAuditRecord] = []

    def create_round_record(
        self,
        round_id: str,
        participant_count: int,
        aggregation_strategy: str,
        model_version: str,
        model_bytes: bytes | None = None,
        update_norm: float = 0.0,
        privacy_budget_used: float = 0.0,
        status: str = "completed",
        notes: str = "",
    ) -> FederatedRoundAuditRecord:
        """Create an audit record for a federated round.

        Args:
            round_id: Unique identifier for this round.
            participant_count: Number of participants in this round.
            aggregation_strategy: Name of aggregation strategy used (e.g., 'FedAvg', 'FedProx').
            model_version: Version identifier or timestamp of resulting model.
            model_bytes: Model checkpoint bytes for computing hash.
            update_norm: L2 norm of aggregated model update.
            privacy_budget_used: Epsilon/privacy budget consumed by this round.
            status: Round status ('completed', 'failed', 'rolled_back', etc.).
            notes: Additional context or metadata.

        Returns:
            FederatedRoundAuditRecord for the round.
        """
        model_hash = self._compute_model_hash(model_bytes) if model_bytes else "unknown"

        record = FederatedRoundAuditRecord(
            round_id=round_id,
            timestamp=datetime.now(UTC).isoformat(),
            participant_count=participant_count,
            aggregation_strategy=aggregation_strategy,
            model_version=model_version,
            model_hash=model_hash,
            update_norm=update_norm,
            privacy_budget_used=privacy_budget_used,
            status=status,
            notes=notes,
        )

        self.records.append(record)

        logger.info(
            "Federated round audit record created: round_id=%s participants=%d "
            "strategy=%s model_hash=%s",
            round_id,
            participant_count,
            aggregation_strategy,
            model_hash,
        )

        return record

    def _compute_model_hash(self, model_bytes: bytes) -> str:
        """Compute SHA256 hash of model checkpoint.

        Args:
            model_bytes: Serialized model bytes.

        Returns:
            Hex-encoded SHA256 hash.
        """
        return hashlib.sha256(model_bytes).hexdigest()

    def link_to_governance(
        self,
        round_id: str,
        governance_action: str,
    ) -> None:
        """Link audit record to model governance system.

        Called when a round's output model is promoted/rolled back.

        Args:
            round_id: Round ID from audit record.
            governance_action: Governance action ('promote', 'rollback', etc.).
        """
        record = self._find_record(round_id)
        if record:
            record.notes += f" [governance: {governance_action}]"
            logger.info(
                "Linked federated round %s to governance action: %s",
                round_id,
                governance_action,
            )

    def get_round_context(self, round_id: str) -> dict[str, Any] | None:
        """Get full audit context for a round (for forensic review).

        Args:
            round_id: Round identifier.

        Returns:
            Dict with complete audit record and derived context.
        """
        record = self._find_record(round_id)
        if not record:
            return None

        return {
            "record": record.to_dict(),
            "derived_context": {
                "is_privacy_compliant": record.privacy_budget_used >= 0,
                "update_magnitude": "large" if record.update_norm > 1.0 else "normal",
                "participant_cohort_size": record.participant_count,
            },
        }

    def _find_record(self, round_id: str) -> FederatedRoundAuditRecord | None:
        """Find an audit record by round ID.

        Args:
            round_id: Round identifier.

        Returns:
            FederatedRoundAuditRecord or None if not found.
        """
        for record in self.records:
            if record.round_id == round_id:
                return record
        return None

    def export_audit_trail(self) -> dict[str, Any]:
        """Export complete audit trail as JSON-serializable dict.

        Returns:
            Dict with all audit records and metadata.
        """
        return {
            "exported_at": datetime.now(UTC).isoformat(),
            "total_rounds": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }

    def validate_trail_integrity(self) -> dict[str, Any]:
        """Validate audit trail for integrity issues.

        Checks for:
        - Duplicate round IDs
        - Chronological ordering
        - Missing critical fields

        Returns:
            Dict with validation results.
        """
        issues = []
        round_ids = set()

        prev_timestamp = None
        for record in self.records:
            if record.round_id in round_ids:
                issues.append(f"Duplicate round_id: {record.round_id}")
            round_ids.add(record.round_id)

            if not record.model_hash or record.model_hash == "unknown":
                issues.append(f"Missing model_hash for round {record.round_id}")

            if prev_timestamp and record.timestamp < prev_timestamp:
                issues.append(f"Chronological violation at round {record.round_id}")

            prev_timestamp = record.timestamp

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_records": len(self.records),
        }

    def get_summary_for_governance(self) -> dict[str, Any]:
        """Generate summary of federated activity for model governance.

        Used when integrating with model_governance.py version history.

        Returns:
            Summary suitable for governance system.
        """
        if not self.records:
            return {
                "total_rounds": 0,
                "summary": "No federated rounds recorded",
            }

        latest_record = self.records[-1]

        return {
            "total_rounds": len(self.records),
            "latest_round": latest_record.to_dict(),
            "latest_model_version": latest_record.model_version,
            "latest_model_hash": latest_record.model_hash,
            "total_privacy_consumed": sum(r.privacy_budget_used for r in self.records),
            "average_participants_per_round": sum(r.participant_count for r in self.records) / len(self.records),
            "aggregation_strategies_used": list(set(r.aggregation_strategy for r in self.records)),
        }

    def privacy_compliant_export(self) -> list[dict[str, Any]]:
        """Export audit trail with privacy-preserving aggregation.

        Removes participant identifiers and maintains only aggregated statistics
        that don't risk re-identification.

        Returns:
            List of privacy-safe audit records.
        """
        safe_records = []

        for record in self.records:
            safe_record = {
                "round_id_hash": hashlib.sha256(record.round_id.encode()).hexdigest()[:8],
                "timestamp": record.timestamp,
                "participant_count": record.participant_count,
                "aggregation_strategy": record.aggregation_strategy,
                "model_version": record.model_version,
                "model_hash": record.model_hash,
                "update_norm": record.update_norm,
                "privacy_budget_used": record.privacy_budget_used,
                "status": record.status,
            }
            safe_records.append(safe_record)

        return safe_records
