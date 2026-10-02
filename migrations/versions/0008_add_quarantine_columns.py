"""Migration 0008: Add quarantine columns to model_versions.

Issue #871 extends the backdoor detector to auto-quarantine a candidate
model flagged by activation-clustering + trigger-localization, pending
human review. ``detection.model_governance.promote_candidate`` refuses to
promote any candidate whose ``model_artifact_path`` has a
``status="quarantined"`` row. Existing rows default to NULL (never
quarantined).
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddQuarantineColumns(Migration):
    id = "0008"
    description = (
        "Add nullable quarantined_at (DATETIME) and quarantine_reason (TEXT) "
        "columns to model_versions"
    )

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "model_versions" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("model_versions")}
        if "quarantined_at" not in existing:
            conn.execute(text("ALTER TABLE model_versions ADD COLUMN quarantined_at DATETIME"))
        if "quarantine_reason" not in existing:
            conn.execute(text("ALTER TABLE model_versions ADD COLUMN quarantine_reason TEXT"))


migration = AddQuarantineColumns()
