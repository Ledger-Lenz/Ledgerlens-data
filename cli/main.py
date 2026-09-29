"""LedgerLens operational CLI entrypoint.

Subcommands
-----------
healthcheck        Run diagnostic health checks on setup and environment.
validate-artifacts Validate local model and schema artifacts.
backup             Back up the database and model artifacts.
restore            Restore from a backup (interactive confirmation + blast-radius summary).
migrate            Apply database migrations (dry-run supported).
backfill           Backfill AMM trade history and compute cross-venue features.

All state-mutating subcommands support:
  --dry-run   Report what would change without applying any mutation (#960).
  --yes        Skip interactive confirmation for scripted/CI use (#962).
              ⚠️  Use only after reviewing the blast-radius with --dry-run first.

All subcommands emit audit events to CLI_AUDIT_LOG_PATH when LEDGERLENS_ENV=production (#961).
"""

import argparse
import json
import logging
import sys

from cli.commands.validate_artifacts import validate_artifacts
from cli.diagnostics import run_diagnostics
from cli.dry_run import add_dry_run_argument


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerlens-ops",
        description="Operational Harness for LedgerLens Data Pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── healthcheck ────────────────────────────────────────────────────────
    health_parser = subparsers.add_parser(
        "healthcheck", help="Run diagnostic health checks on setup and variables"
    )
    health_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    # ── validate-artifacts ────────────────────────────────────────────────
    val_parser = subparsers.add_parser(
        "validate-artifacts", help="Validate local model and schema artifacts"
    )
    val_parser.add_argument(
        "--dir", default="artifacts", help="Path to artifacts folder (default: artifacts)"
    )

    # ── backup ────────────────────────────────────────────────────────────
    backup_parser = subparsers.add_parser(
        "backup",
        help="Back up database and model artifacts (#960: --dry-run supported)",
    )
    add_dry_run_argument(backup_parser)

    # ── restore ───────────────────────────────────────────────────────────
    restore_parser = subparsers.add_parser(
        "restore",
        help=(
            "Restore database and model artifacts from backup "
            "(#962: requires confirmation; #960: --dry-run supported)"
        ),
    )
    add_dry_run_argument(restore_parser)
    restore_parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help=(
            "Skip interactive confirmation (non-interactive override). "
            "⚠️  USE WITH CAUTION — always run --dry-run first."
        ),
    )

    # ── migrate ───────────────────────────────────────────────────────────
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Apply pending database migrations (#960: --dry-run supported; #961: audited)",
    )
    migrate_parser.add_argument(
        "db_url",
        nargs="?",
        default=None,
        help="Database URL (defaults to RISK_SCORE_DB_URL from environment)",
    )
    migrate_parser.add_argument(
        "--target",
        metavar="ID",
        default=None,
        help="Stop after applying this migration ID (e.g. '0002')",
    )
    add_dry_run_argument(migrate_parser)
    migrate_parser.add_argument(
        "--status",
        action="store_true",
        help="Print current migration status without applying anything",
    )

    # ── backfill ──────────────────────────────────────────────────────────
    backfill_parser = subparsers.add_parser(
        "backfill",
        help=(
            "Backfill AMM trade history and cross-venue features "
            "(#960: --dry-run; #962: confirmation; #961: audited)"
        ),
    )
    backfill_parser.add_argument(
        "--pool-ids",
        nargs="+",
        default=None,
        help="AMM pool IDs (64-char hex). Defaults to WATCHED_AMM_POOLS from config.",
    )
    backfill_parser.add_argument("--since", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    backfill_parser.add_argument("--until", default="2024-06-30", help="End date (YYYY-MM-DD)")
    backfill_parser.add_argument(
        "--output", default="data/labelled_with_cross_venue.parquet", help="Output Parquet path"
    )
    backfill_parser.add_argument("--sdex-trades", default=None, help="Existing SDEX trades Parquet")
    backfill_parser.add_argument("--checkpoint-file", default=None, help="Resumable checkpoint path")
    backfill_parser.add_argument(
        "--fresh", action="store_true", help="Discard existing checkpoint and restart"
    )
    add_dry_run_argument(backfill_parser)
    backfill_parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help=(
            "Skip interactive confirmation (non-interactive override). "
            "⚠️  USE WITH CAUTION — always run --dry-run first."
        ),
    )

    return parser


def _format_health_summary(report: dict) -> str:
    lines = [
        f"Overall status: {report.get('overall_status', 'UNKNOWN')}",
        f"Checks run: {report.get('checks', {}).get('environment', {}).get('status', 'unknown')}",
    ]
    env = report.get("checks", {}).get("environment", {})
    if env:
        lines.append(f"Environment: {env.get('status', 'unknown')}")
    streaming = report.get("checks", {}).get("streaming", {})
    if streaming:
        lines.append(f"Streaming: {streaming.get('status', 'unknown')}")
    return "\n".join(lines)


def main(args=None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)
    setup_logging(opts.verbose)

    # ── healthcheck ────────────────────────────────────────────────────────
    if opts.command == "healthcheck":
        report = run_diagnostics()
        if getattr(opts, "json", False):
            print(json.dumps(report, indent=2))
        else:
            print(_format_health_summary(report))
        return 0 if report["overall_status"] == "PASS" else 2

    # ── validate-artifacts ────────────────────────────────────────────────
    elif opts.command == "validate-artifacts":
        res = validate_artifacts(opts.dir)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "PASS" else 1

    # ── backup ────────────────────────────────────────────────────────────
    elif opts.command == "backup":
        from scripts.backup import (
            backup_database,
            backup_models,
            create_backup_manifest,
        )
        import os
        from pathlib import Path
        from datetime import UTC, datetime
        from cli.audit_hook import emit_cli_audit_event
        from cli.dry_run import DryRunContext

        db_url = os.getenv("DATABASE_URL", "sqlite:///ledgerlens.db")
        model_dir = os.getenv("MODEL_DIR", "./models")
        backup_dir = Path(os.getenv("BACKUP_DIR", "./backups"))
        ts = datetime.now(UTC).isoformat().replace(":", "-")
        cli_args = {"dry_run": opts.dry_run, "backup_dir": str(backup_dir)}

        with DryRunContext(opts.dry_run) as dry:
            dry.record(f"Write database backup to {backup_dir}/database_{ts}.db")
            dry.record(f"Write model archive to {backup_dir}/models_{ts}.tar.gz")
            dry.record(f"Write manifest to {backup_dir}/MANIFEST.json")
            if dry:
                emit_cli_audit_event("backup", cli_args, "dry-run")
                return 0

        db_meta = backup_database(db_url, backup_dir)
        models_meta = backup_models(model_dir, backup_dir)
        if not db_meta:
            emit_cli_audit_event("backup", cli_args, "error", error="Database backup failed")
            return 1
        create_backup_manifest(db_meta, models_meta, backup_dir)
        emit_cli_audit_event("backup", cli_args, "success")
        return 0

    # ── restore ───────────────────────────────────────────────────────────
    elif opts.command == "restore":
        import os
        from pathlib import Path
        from cli.audit_hook import emit_cli_audit_event
        from cli.confirmation import ConfirmationAborted, confirm_destructive
        from cli.dry_run import DryRunContext
        from scripts.restore import load_manifest, restore_database, restore_models

        backup_dir = Path(os.getenv("BACKUP_DIR", "./backups"))
        db_url = os.getenv("DATABASE_URL", "sqlite:///ledgerlens.db")
        model_dir = Path(os.getenv("MODEL_DIR", "./models"))
        cli_args = {"dry_run": opts.dry_run, "yes": opts.yes}

        if not backup_dir.exists():
            print(f"Error: backup directory not found: {backup_dir}", file=sys.stderr)
            return 1
        manifest = load_manifest(backup_dir)
        if not manifest:
            print("Error: failed to load backup manifest", file=sys.stderr)
            return 1

        db_meta = manifest.get("database", {})
        models_meta = manifest.get("models", {})
        db_backup = Path(db_meta.get("path", ""))
        blast = {
            "operation": "restore database and model artifacts from backup",
            "target_database": db_url[:60],
            "backup_timestamp": manifest.get("timestamp", "unknown"),
            "database_backup_size": f"{db_meta.get('size_bytes', 0):,} bytes",
            "models_present": "yes" if models_meta else "no",
        }

        with DryRunContext(opts.dry_run) as dry:
            dry.record(f"Verify checksum of {db_backup}")
            dry.record(f"Restore database to {db_url[:60]}")
            if models_meta:
                dry.record(f"Restore model artifacts to {model_dir}")
            if dry:
                emit_cli_audit_event("restore", cli_args, "dry-run")
                return 0

        try:
            confirm_destructive(blast, yes=opts.yes)
        except ConfirmationAborted:
            emit_cli_audit_event("restore", cli_args, "aborted")
            return 1

        if not db_backup.exists():
            print(f"Error: database backup not found: {db_backup}", file=sys.stderr)
            return 1
        if not restore_database(db_url, db_backup, manifest):
            emit_cli_audit_event("restore", cli_args, "error", error="DB restore failed")
            return 1
        if models_meta:
            models_backup = Path(models_meta.get("path", ""))
            if not restore_models(models_backup, model_dir, manifest):
                emit_cli_audit_event("restore", cli_args, "error", error="Models restore failed")
                return 1
        emit_cli_audit_event("restore", cli_args, "success")
        return 0

    # ── migrate ───────────────────────────────────────────────────────────
    elif opts.command == "migrate":
        # Delegate to scripts.migrate.main() with the same argv semantics
        from scripts.migrate import main as migrate_main

        argv = []
        if opts.db_url:
            argv.append(opts.db_url)
        if opts.target:
            argv += ["--target", opts.target]
        if opts.dry_run:
            argv.append("--dry-run")
        if opts.status:
            argv.append("--status")
        return migrate_main(argv)

    # ── backfill ──────────────────────────────────────────────────────────
    elif opts.command == "backfill":
        import sys as _sys
        from unittest.mock import patch

        argv = []
        if opts.pool_ids:
            argv += ["--pool-ids"] + opts.pool_ids
        argv += ["--since", opts.since, "--until", opts.until, "--output", opts.output]
        if opts.sdex_trades:
            argv += ["--sdex-trades", opts.sdex_trades]
        if opts.checkpoint_file:
            argv += ["--checkpoint-file", opts.checkpoint_file]
        if opts.fresh:
            argv.append("--fresh")
        if opts.dry_run:
            argv.append("--dry-run")
        if opts.yes:
            argv.append("--yes")

        with patch("sys.argv", ["backfill_amm_trades"] + argv):
            from scripts.backfill_amm_trades import main as backfill_main

            backfill_main()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
