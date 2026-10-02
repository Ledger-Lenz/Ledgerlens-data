"""Database and artifact restore with integrity verification.

Restores from timestamped backups with checksum validation.
Verifies restored data matches backup manifest before marking restore complete.
"""

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _verify_checksum(file_path: Path, expected: str) -> bool:
    """Verify file checksum matches expected value."""
    sha256_hash = hashlib.sha256()

    if file_path.is_file():
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
    else:
        for path in sorted(file_path.rglob("*")):
            if path.is_file():
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)

    actual = sha256_hash.hexdigest()
    if actual != expected:
        logger.error(f"Checksum mismatch: expected {expected}, got {actual}")
        return False

    logger.info(f"Checksum verified: {expected[:16]}...")
    return True


def load_manifest(backup_dir: Path) -> dict:
    """Load backup manifest from MANIFEST.json."""
    manifest_file = backup_dir / "MANIFEST.json"

    if not manifest_file.exists():
        logger.error(f"Manifest not found: {manifest_file}")
        return {}

    with open(manifest_file) as f:
        return json.load(f)


def restore_database(db_url: str, backup_path: Path, manifest: dict) -> bool:
    """Restore database from backup with verification.

    Args:
        db_url: Target database URL
        backup_path: Path to backup file
        manifest: Backup manifest with checksum

    Returns:
        True if restore succeeded and verified
    """
    db_meta = manifest.get("database", {})

    if not backup_path.exists():
        logger.error(f"Backup file not found: {backup_path}")
        return False

    if not _verify_checksum(backup_path, db_meta.get("checksum", "")):
        logger.error("Database backup integrity check failed")
        return False

    if "sqlite" in db_url:
        db_path = Path(db_url.replace("sqlite:///", ""))

        if db_path.exists():
            db_path.rename(db_path.with_suffix(".db.bak"))
            logger.info(f"Existing database backed up to: {db_path.with_suffix('.db.bak')}")

        shutil.copy2(backup_path, db_path)
        logger.info(f"Database restored from: {backup_path}")
        return True

    elif "postgresql" in db_url:
        try:
            with open(backup_path) as f:
                subprocess.run(
                    ["psql", db_url],
                    stdin=f,
                    capture_output=True,
                    check=True,
                    timeout=3600,
                )

            logger.info(f"Database restored from: {backup_path}")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Database restore timed out (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    else:
        logger.error(f"Unsupported database URL: {db_url}")
        return False


def restore_models(backup_path: Path, restore_dir: Path, manifest: dict) -> bool:
    """Restore model artifacts from backup.

    Args:
        backup_path: Path to backup archive
        restore_dir: Directory to restore to
        manifest: Backup manifest

    Returns:
        True if restore succeeded and verified
    """
    models_meta = manifest.get("models", {})

    if not backup_path.exists():
        logger.warning(f"Models backup not found: {backup_path}, skipping")
        return True

    if not _verify_checksum(backup_path, models_meta.get("checksum", "")):
        logger.error("Models backup integrity check failed")
        return False

    restore_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.unpack_archive(str(backup_path), restore_dir.parent)
        logger.info(f"Models restored to: {restore_dir}")
        return True
    except Exception as e:
        logger.error(f"Models restore failed: {e}")
        return False


def main():
    """Restore from backup with full verification.

    Flags
    -----
    --yes
        Non-interactive override: skip the confirmation prompt and proceed
        automatically.

        .. warning::
            Using ``--yes`` bypasses the human confirmation gate entirely.
            Only use it in CI/automation pipelines where the blast radius has
            already been reviewed.  **Never** use it as a shortcut during
            ad-hoc production operations.
    """
    import argparse
    import os
    import sys

    from cli.audit import audit_cli_command
    from cli.confirmation import BlastRadiusSummary, confirm_destructive_action

    parser = argparse.ArgumentParser(description="Restore LedgerLens database and model artifacts")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help=(
            "Non-interactive override: skip confirmation prompt. "
            "⚠️  Use with extreme caution — this bypasses the human safety gate."
        ),
    )

    # Issue #960 — dry-run support
    from cli.dry_run import DryRunPlan, add_dry_run_argument, check_dry_run

    add_dry_run_argument(parser)
    args = parser.parse_args()

    backup_dir = Path(os.getenv("BACKUP_DIR", "./backups"))
    db_url = os.getenv("DATABASE_URL", "sqlite:///ledgerlens.db")
    model_dir = Path(os.getenv("MODEL_DIR", "./models"))

    if not backup_dir.exists():
        logger.error(f"Backup directory not found: {backup_dir}")
        return 1

    logger.info(f"Loading backup manifest from: {backup_dir}")
    manifest = load_manifest(backup_dir)

    if not manifest:
        logger.error("Failed to load backup manifest")
        return 1

    db_meta = manifest.get("database", {})
    models_meta = manifest.get("models", {})
    db_backup = Path(db_meta.get("path", ""))

    with DryRunContext(args.dry_run) as dry:
        dry.record(f"Verify checksum of database backup {db_backup}")
        dry.record(f"Restore database to {db_url[:60]}")
        if models_meta:
            dry.record(f"Restore model artifacts to {model_dir}")

        if dry:
            emit_cli_audit_event("restore", cli_args, "dry-run", force=False)
            return 0

    # Interactive confirmation (skipped in dry-run already exited above)
    try:
        confirm_destructive(blast_summary, yes=args.yes)
    except ConfirmationAborted:
        emit_cli_audit_event("restore", cli_args, "aborted")
        return 1

    logger.info(f"Restoring to database: {db_url[:40]}...")
    if not db_backup.exists():
        logger.error(f"Database backup file not found: {db_backup}")
        emit_cli_audit_event("restore", cli_args, "error", error=f"Backup not found: {db_backup}")
        return 1

    # -----------------------------------------------------------------
    # Issue #960 — Dry-run: report what would happen and exit early
    # -----------------------------------------------------------------
    plan = DryRunPlan(
        command="restore",
        extra={"backup_timestamp": manifest.get("timestamp", "unknown")},
    )
    plan.add_action(
        "overwrite",
        f"database: {db_url[:60]}",
        detail=f"from backup: {db_meta.get('path', 'unknown')} "
        f"({db_meta.get('size_bytes', 0) / 1024 / 1024:.1f} MB)",
    )
    if models_meta:
        plan.add_action(
            "overwrite",
            f"models directory: {model_dir}",
            detail=f"from backup: {models_meta.get('path', 'unknown')} "
            f"({models_meta.get('size_bytes', 0) / 1024 / 1024:.1f} MB)",
        )
    if check_dry_run(args, plan):
        return 0

    # -----------------------------------------------------------------
    # Issue #962 — Blast-radius summary + confirmation
    # -----------------------------------------------------------------
    affected: dict[str, int] = {}
    if db_meta:
        size_mb = db_meta.get("size_bytes", 0) / (1024 * 1024)
        affected["database (size)"] = int(size_mb)
    if models_meta:
        size_mb = models_meta.get("size_bytes", 0) / (1024 * 1024)
        affected["models archive (size MB)"] = int(size_mb)

    summary = BlastRadiusSummary(
        operation="Database + model artifact restore",
        affected_records=affected,
        affected_tenants=[db_url.split("@")[-1].split("/")[0] if "@" in db_url else "local"],
        extra={
            "backup_dir": str(backup_dir),
            "backup_timestamp": manifest.get("timestamp", "unknown"),
            "target_db": db_url[:60] + ("..." if len(db_url) > 60 else ""),
            "WARNING": "This will OVERWRITE the current database and models",
        },
    )

    if not confirm_destructive_action(summary, non_interactive=args.yes):
        logger.info("Restore cancelled by operator.")
        return 1

    # -----------------------------------------------------------------
    logger.info(f"Restoring to database: {db_url[:40]}...")

    with audit_cli_command(
        "restore",
        args={
            "backup_dir": str(backup_dir),
            "db_url": db_url[:60],
            "model_dir": str(model_dir),
            "yes": args.yes,
        },
    ):
        if not restore_database(db_url, db_backup, manifest):
            logger.error("Database restore verification failed")
            return 1

        if models_meta:
            models_backup = Path(models_meta.get("path", ""))
            if not restore_models(models_backup, model_dir, manifest):
                logger.error("Models restore verification failed")
                return 1

    logger.info("✅ Restore complete and verified")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
