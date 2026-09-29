"""CLI entry point for the LedgerLens migration runner.

Usage::

    # Apply all pending migrations (uses RISK_SCORE_DB_URL from env / .env)
    python -m scripts.migrate

    # Migrate a specific database URL
    python -m scripts.migrate sqlite:///./my.db

    # Dry-run: show what would be applied without executing
    python -m scripts.migrate --dry-run

    # Apply up to (and including) migration 0002
    python -m scripts.migrate --target 0002

    # Show migration status without applying anything
    python -m scripts.migrate --status
"""

from __future__ import annotations

import argparse
import sys

from detection.persistence import get_engine
from migrations import MigrationRunner
from utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply LedgerLens database migrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "db_url",
        nargs="?",
        default=None,
        help="Database URL (defaults to RISK_SCORE_DB_URL from environment)",
    )
    parser.add_argument(
        "--target",
        metavar="ID",
        default=None,
        help="Stop after applying this migration ID (e.g. '0002')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which migrations would be applied without executing them",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the current migration status and exit without applying anything",
    )
    args = parser.parse_args(argv)

    from cli.audit_hook import emit_cli_audit_event

    cli_args = {
        "db_url": (args.db_url or "")[:40] + "..." if args.db_url else "from-env",
        "target": args.target,
        "dry_run": args.dry_run,
        "status": args.status,
    }

    engine = get_engine(args.db_url)
    runner = MigrationRunner(engine, target=args.target, dry_run=args.dry_run)

    if args.status:
        status = runner.status()
        if status.applied:
            print("Applied migrations:")
            for mid in status.applied:
                print(f"  [applied] {mid}")
        if status.pending:
            print("Pending migrations:")
            for mid in status.pending:
                print(f"  [pending] {mid}")
        if status.is_up_to_date:
            print("Database is up to date.")
        return 0 if status.is_up_to_date else 1

    try:
        status = runner.upgrade(args.target)
    except Exception as exc:
        emit_cli_audit_event("migrate", cli_args, "error", error=str(exc))
        raise

    if status.pending and not args.dry_run:
        # upgrade() returned but there are still pending migrations — something
        # went wrong; surface this as a non-zero exit.
        logger.error("Some migrations could not be applied: %s", status.pending)
        emit_cli_audit_event("migrate", cli_args, "error", error=f"Pending: {status.pending}")
        return 1

    outcome = "dry-run" if args.dry_run else "success"
    emit_cli_audit_event("migrate", cli_args, outcome)

    if args.dry_run:
        print("Dry-run complete. No migrations were applied.")
    else:
        remaining = len(status.pending)
        applied = len(status.applied)
        print(f"Migration complete. Applied: {applied}, Pending: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
