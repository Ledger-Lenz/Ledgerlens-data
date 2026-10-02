"""Database and artifact backup automation with integrity verification.

Backs up:
- Primary database (PostgreSQL/SQLite)
- Model artifacts directory
- Feature store (if separate)

Produces timestamped backups with checksums for verification.
"""

import hashlib
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _compute_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file or directory."""
    sha256_hash = hashlib.sha256()

    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
    else:
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def backup_database(db_url: str, backup_dir: Path) -> dict:
    """Backup the primary database.

    Args:
        db_url: Database connection URL
        backup_dir: Directory to store backup

    Returns:
        Metadata dict with checksum and timestamp
    """
    timestamp = datetime.now(UTC).isoformat()
    backup_dir.mkdir(parents=True, exist_ok=True)

    if "sqlite" in db_url:
        db_path = Path(db_url.replace("sqlite:///", ""))
        if not db_path.exists():
            logger.warning(f"Database file not found: {db_path}")
            return {}

        backup_file = backup_dir / f"database_{timestamp.replace(':', '-')}.db"
        shutil.copy2(db_path, backup_file)
        checksum = _compute_checksum(backup_file)

        metadata = {
            "type": "sqlite",
            "timestamp": timestamp,
            "path": str(backup_file),
            "checksum": checksum,
            "size_bytes": backup_file.stat().st_size,
        }
        logger.info(f"SQLite backup complete: {backup_file} (checksum: {checksum[:16]}...)")
        return metadata

    elif "postgresql" in db_url:
        backup_file = backup_dir / f"database_{timestamp.replace(':', '-')}.sql"

        try:
            with open(backup_file, "w") as f:
                subprocess.run(
                    ["pg_dump", db_url],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=3600,
                )

            checksum = _compute_checksum(backup_file)
            metadata = {
                "type": "postgresql",
                "timestamp": timestamp,
                "path": str(backup_file),
                "checksum": checksum,
                "size_bytes": backup_file.stat().st_size,
            }
            logger.info(f"PostgreSQL backup complete: {backup_file} (checksum: {checksum[:16]}...)")
            return metadata
        except subprocess.TimeoutExpired:
            logger.error("pg_dump timed out (>1 hour)")
            return {}
    else:
        logger.error(f"Unsupported database URL: {db_url}")
        return {}


def backup_models(model_dir: str, backup_dir: Path) -> dict:
    """Backup model artifacts directory.

    Args:
        model_dir: Path to models directory
        backup_dir: Directory to store backup

    Returns:
        Metadata dict with checksum
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.warning(f"Models directory not found: {model_path}")
        return {}

    timestamp = datetime.now(UTC).isoformat()
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_file = backup_dir / f"models_{timestamp.replace(':', '-')}.tar.gz"

    try:
        shutil.make_archive(
            str(backup_file).replace(".tar.gz", ""), "gztar", model_path.parent, model_path.name
        )

        checksum = _compute_checksum(backup_file)
        metadata = {
            "type": "models",
            "timestamp": timestamp,
            "path": str(backup_file),
            "checksum": checksum,
            "size_bytes": backup_file.stat().st_size,
        }
        logger.info(f"Models backup complete: {backup_file} (checksum: {checksum[:16]}...)")
        return metadata
    except Exception as e:
        logger.error(f"Failed to backup models: {e}")
        return {}


def create_backup_manifest(
    database_meta: dict, models_meta: dict, backup_dir: Path
) -> None:
    """Write backup manifest with all metadata and checksums."""
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "database": database_meta,
        "models": models_meta,
    }

    manifest_file = backup_dir / "MANIFEST.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Backup manifest written to: {manifest_file}")


def main():
    """Run full backup suite."""
    import argparse
    import os
    import sys

    from cli.audit_hook import emit_cli_audit_event
    from cli.dry_run import DryRunContext, add_dry_run_argument

    parser = argparse.ArgumentParser(description="LedgerLens database and artifact backup")
    add_dry_run_argument(parser)
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL", "sqlite:///ledgerlens.db")
    model_dir = os.getenv("MODEL_DIR", "./models")
    backup_dir = Path(os.getenv("BACKUP_DIR", "./backups"))

    timestamp = datetime.now(UTC).isoformat().replace(":", "-")
    planned_db_backup = backup_dir / f"database_{timestamp}.db"
    planned_models_backup = backup_dir / f"models_{timestamp}.tar.gz"
    planned_manifest = backup_dir / "MANIFEST.json"

    cli_args = {"dry_run": args.dry_run, "db_url": db_url[:40] + "...", "backup_dir": str(backup_dir)}

    with DryRunContext(args.dry_run) as dry:
        dry.record(f"Write database backup to {planned_db_backup}")
        dry.record(f"Write model archive to {planned_models_backup}")
        dry.record(f"Write manifest to {planned_manifest}")

        if dry:
            emit_cli_audit_event("backup", cli_args, "dry-run", force=False)
            return 0

    logger.info("Starting backup suite...")
    logger.info(f"  Database: {db_url[:40]}...")
    logger.info(f"  Models: {model_dir}")
    logger.info(f"  Backup destination: {backup_dir}")

    database_meta = backup_database(db_url, backup_dir)
    models_meta = backup_models(model_dir, backup_dir)

    if not database_meta:
        logger.error("Database backup failed")
        emit_cli_audit_event("backup", cli_args, "error", error="Database backup failed")
        return 1

    create_backup_manifest(database_meta, models_meta, backup_dir)
    emit_cli_audit_event("backup", cli_args, "success")
    logger.info("✅ Backup complete")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
