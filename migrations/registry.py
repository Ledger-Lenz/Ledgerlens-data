"""Authoritative registry of every known migration ID.

When the :class:`~migrations.runner.MigrationRunner` discovers migration
files in ``migrations/versions/``, it cross-references them against this
list.  If a registered ID is absent from the discovered files the runner
raises ``RuntimeError`` — this guard prevents accidental file deletion from
silently leaving a database in an inconsistent state.

**Maintenance rule**: every time you add a migration file under
``migrations/versions/`` you must also add its ID here in the same commit.
"""

from __future__ import annotations

# Ordered list of all migration IDs that have ever been created for this
# project.  IDs are the zero-padded 4-digit prefix of the migration filename.
REGISTRY: list[str] = [
    "0001",  # Add ring_id column to risk_scores (backport from scripts/migrate_add_ring_id.py)
    "0002",  # Add provenance_json column to risk_scores
    "0003",  # Add certified_robust column to risk_scores
    "0004",  # Add schema_version column to risk_scores for forward migration tracking
    "0005",  # Add content_hash/prev_merkle_root columns to audit_merkle_roots (Issue #670)
    "0006",  # Add finality column to risk_scores (Issue #670)
    "0007",  # Add promoted_by/rolled_back_by/parent_version_id columns to model_versions
    "0008",  # Add quarantined_at/quarantine_reason columns to model_versions (Issue #871)
]
