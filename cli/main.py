"""Operational CLI harness for LedgerLens Data Pipelines.

Issue #959 — CLI: add structured, machine-readable output mode for all
diagnostic commands.

All subcommands support ``--json`` to emit machine-readable JSON output that
conforms to a stable, versioned schema documented in the module that powers
each subcommand:

* ``healthcheck --json``     → schema in :mod:`cli.diagnostics`
* ``validate-artifacts --json`` → schema documented in the function below

Usage
-----
    # Human-readable
    python -m cli.main healthcheck
    python -m cli.main validate-artifacts --dir artifacts/

    # Machine-readable JSON (for CI scripts, piping into jq, etc.)
    python -m cli.main healthcheck --json
    python -m cli.main validate-artifacts --dir artifacts/ --json

Exit codes
----------
0  — all checks passed.
1  — ``validate-artifacts`` failed.
2  — ``healthcheck`` found at least one failing check.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from cli.audit import audit_cli_command
from cli.commands.validate_artifacts import validate_artifacts
from cli.diagnostics import run_diagnostics
from ingestion.historical_loader import backfill_status

# Bump when validate-artifacts JSON schema changes (field removal / type change).
VALIDATE_ARTIFACTS_SCHEMA_VERSION = "1.0"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerlens-ops",
        description=(
            "Operational Harness for LedgerLens Data Pipelines.\n\n"
            "Pass --json to any subcommand for machine-readable JSON output "
            "that conforms to a stable versioned schema."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # healthcheck — environment and streaming checks
    # ------------------------------------------------------------------
    health_parser = subparsers.add_parser(
        "healthcheck",
        help="Run diagnostic health checks on environment variables and streaming config",
        description=(
            "Run diagnostic health checks.\n\n"
            "JSON schema (--json):\n"
            "  schema_version    string   — e.g. '1.0'\n"
            "  overall_status    string   — 'PASS' or 'FAIL'\n"
            "  checks.environment.status  string  — 'PASS' or 'FAIL'\n"
            "  checks.environment.details dict    — VAR_NAME → sanitized value\n"
            "  checks.environment.missing list    — missing required variable names\n"
            "  checks.streaming.status    string  — 'PASS' or 'FAIL'\n"
            "  checks.streaming.backend   string  — backend name\n\n"
            "Full schema descriptor: python -c \"from cli.diagnostics import json_schema; "
            "import json; print(json.dumps(json_schema(), indent=2))\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    health_parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit machine-readable JSON conforming to the stable schema "
            "documented in cli/diagnostics.py (schema_version 1.0). "
            "Guaranteed fields: schema_version, overall_status, checks."
        ),
    )

    # ------------------------------------------------------------------
    # validate-artifacts — model artifact validation
    # ------------------------------------------------------------------
    val_parser = subparsers.add_parser(
        "validate-artifacts",
        help="Validate local model and schema artifacts",
        description=(
            "Validate model artifact files in a directory.\n\n"
            "JSON schema (--json):\n"
            "  schema_version  string   — e.g. '1.0'\n"
            "  status          string   — 'PASS' or 'FAIL'\n"
            "  artifacts_dir   string   — path that was checked\n"
            "  version         string|null  — model_version from metadata\n"
            "  schema_hash     string|null  — feature_schema_hash from metadata\n"
            "  error           string|null  — failure reason when status=FAIL\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    val_parser.add_argument(
        "--dir",
        default="artifacts",
        help="Path to artifacts folder (default: artifacts)",
    )
    val_parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit machine-readable JSON conforming to the stable schema "
            "documented in cli/main.py (schema_version 1.0). "
            "Guaranteed fields: schema_version, status, artifacts_dir, version, "
            "schema_hash, error."
        ),
    )
    val_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Report what artifacts would be validated without writing any outputs. "
            "Produces zero side effects."
        ),
    )

    backfill_parser = subparsers.add_parser(
        "backfill-status", help="Show progress of a resumable historical backfill"
    )
    backfill_parser.add_argument(
        "--checkpoint-file", required=True, help="Checkpoint file written by the backfill"
    )
    backfill_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

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


def _format_health_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Overall status: {report.get('overall_status', 'UNKNOWN')}",
    ]
    env = report.get("checks", {}).get("environment", {})
    if env:
        lines.append(f"Environment:    {env.get('status', 'unknown')}")
        for missing in env.get("missing", []):
            lines.append(f"  MISSING: {missing}")
        for k, v in env.get("details", {}).items():
            lines.append(f"  {k}: {v}")
    streaming = report.get("checks", {}).get("streaming", {})
    if streaming:
        lines.append(f"Streaming:      {streaming.get('status', 'unknown')} "
                     f"(backend: {streaming.get('backend', 'stdout')})")
    return "\n".join(lines)


def _format_backfill_status(status: dict) -> str:
    lines = [
        f"Pair: {status['pair']} (start_time={status['start_time'] or 'all'})",
        f"Chunks completed: {status['chunks_completed']} (last: {status['last_chunk']})",
        f"Trades loaded: {status['trades']} ({status['raw_records']} raw records scanned)",
        f"Last ledger close time: {status['last_ledger_close_time']}",
        f"Resume cursor: {status['cursor']}",
        f"Started: {status['started_at']}  Updated: {status['updated_at']}",
    ]
    for chunk, failure in sorted(status["failed_chunks"].items()):
        lines.append(f"FAILED {chunk} (attempts={failure.get('attempts')}): {failure.get('error')}")
    return "\n".join(lines)


def _format_artifacts_summary(res: dict[str, Any]) -> str:
    lines = [f"Status:  {res.get('status', 'UNKNOWN')}"]
    if res.get("artifacts_dir"):
        lines.append(f"Dir:     {res['artifacts_dir']}")
    if res.get("version"):
        lines.append(f"Version: {res['version']}")
    if res.get("schema_hash"):
        lines.append(f"Schema hash: {res['schema_hash']}")
    if res.get("error"):
        lines.append(f"Error:   {res['error']}")
    return "\n".join(lines)


def main(args: list[str] | None = None) -> int:
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
        # Enrich with schema_version for a stable JSON envelope
        res_with_schema: dict[str, Any] = {
            "schema_version": VALIDATE_ARTIFACTS_SCHEMA_VERSION,
            **res,
            # Ensure guaranteed fields are always present even on FAIL
            "version": res.get("version"),
            "schema_hash": res.get("schema_hash"),
            "error": res.get("error"),
        }
        if getattr(opts, "json", False):
            print(json.dumps(res_with_schema, indent=2))
        else:
            print(_format_artifacts_summary(res_with_schema))
        return 0 if res_with_schema["status"] == "PASS" else 1
        with audit_cli_command("validate-artifacts", args={"dir": opts.dir}):
            res = validate_artifacts(opts.dir)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "PASS" else 1

    elif opts.command == "backfill-status":
        try:
            status = backfill_status(opts.checkpoint_file)
        except (OSError, ValueError) as exc:
            print(f"Cannot read backfill checkpoint: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(status, indent=2) if opts.json else _format_backfill_status(status))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
