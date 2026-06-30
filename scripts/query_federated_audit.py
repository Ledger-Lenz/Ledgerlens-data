#!/usr/bin/env python3
"""Query the federated learning audit trail (issue #227).

Supports three query modes:
  --round-id    ROUND_ID     Look up a specific round by its deterministic hash.
  --participant FINGERPRINT  List all rounds a participant contributed to.
  --model-hash  HASH         Find rounds that produced a specific model version.
  --list                     List recent records (paginated with --limit/--offset).

Examples
--------
Query by round ID:
    python -m scripts.query_federated_audit --round-id abc123...

Query by participant certificate fingerprint:
    python -m scripts.query_federated_audit --participant deadbeef...

Query by aggregate model hash:
    python -m scripts.query_federated_audit --model-hash 1a2b3c...

List the 20 most recent records:
    python -m scripts.query_federated_audit --list --limit 20

Use a custom database URL:
    python -m scripts.query_federated_audit --db-url postgresql://... --list

Output format defaults to human-readable table; pass --json for machine-readable
JSON (one object per line, NDJSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _truncate(value: str, width: int = 16) -> str:
    """Return the first *width* characters of *value* followed by '…'."""
    s = str(value)
    return s[:width] + "…" if len(s) > width else s


def _print_table(records: list[dict[str, Any]]) -> None:
    """Render *records* as an ASCII table to stdout."""
    if not records:
        print("No records found.")
        return

    headers = [
        ("id", 6),
        ("round_id", 18),
        ("model_version", 13),
        ("participants", 12),
        ("outcome", 9),
        ("algorithm", 26),
        ("round_timestamp", 28),
    ]

    # Header row
    header_line = "  ".join(h.ljust(w) for h, w in headers)
    print(header_line)
    print("-" * len(header_line))

    for rec in records:
        row = [
            str(rec.get("id", "")).ljust(6),
            _truncate(rec.get("round_id", ""), 16).ljust(18),
            str(rec.get("model_version", "")).ljust(13),
            str(rec.get("participant_count", "")).ljust(12),
            str(rec.get("round_outcome", "")).ljust(9),
            str(rec.get("aggregation_algorithm", "")).ljust(26),
            str(rec.get("round_timestamp", "")).ljust(28),
        ]
        print("  ".join(row))


def _print_detail(records: list[dict[str, Any]]) -> None:
    """Print full detail for each record."""
    if not records:
        print("No records found.")
        return
    for rec in records:
        print(f"\n{'─' * 60}")
        print(f"  id               : {rec.get('id')}")
        print(f"  round_id         : {rec.get('round_id')}")
        print(f"  round_timestamp  : {rec.get('round_timestamp')}")
        print(f"  model_version    : {rec.get('model_version')}")
        print(f"  outcome          : {rec.get('round_outcome')}")
        print(f"  algorithm        : {rec.get('aggregation_algorithm')}")
        print(f"  participant_count: {rec.get('participant_count')}")
        print(f"  model_hash       : {rec.get('aggregate_model_hash')}")
        print(f"  prev_hash        : {rec.get('prev_hash')}")
        print(f"  recorded_at      : {rec.get('recorded_at')}")
        fingerprints = rec.get("participant_fingerprints", [])
        print(f"  fingerprints ({len(fingerprints)}):")
        for fp in fingerprints:
            print(f"    {fp}")
        norms = rec.get("gradient_norms", {})
        print(f"  gradient_norms ({len(norms)}):")
        for pid, norm in norms.items():
            print(f"    {pid}: {norm:.6f}")


def _print_ndjson(records: list[dict[str, Any]]) -> None:
    for rec in records:
        print(json.dumps(rec, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query the LedgerLens federated learning audit trail.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Query filters (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--round-id",
        metavar="ROUND_ID",
        help="Look up a round by its deterministic SHA-256 round ID.",
    )
    query_group.add_argument(
        "--participant",
        metavar="FINGERPRINT",
        help="List all rounds that include this participant fingerprint.",
    )
    query_group.add_argument(
        "--model-hash",
        metavar="HASH",
        help="Find rounds whose aggregate model hash matches HASH.",
    )
    query_group.add_argument(
        "--list",
        action="store_true",
        help="List recent audit records (paginated).",
    )

    # Pagination (used with --list)
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of records to return (default: 50).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="N",
        help="Pagination offset (default: 0).",
    )

    # Output format
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output records as NDJSON (one JSON object per line).",
    )
    output_group.add_argument(
        "--detail",
        action="store_true",
        help="Print full detail for each record.",
    )

    # Database
    parser.add_argument(
        "--db-url",
        default=None,
        metavar="URL",
        help="SQLAlchemy DB URL (default: RISK_SCORE_DB_URL env var / sqlite:///ledgerlens.db).",
    )

    args = parser.parse_args()

    # Initialise audit trail
    try:
        from detection.federated.coordinator import FederatedAuditTrail
        audit = FederatedAuditTrail(db_url=args.db_url)
    except Exception as exc:
        logger.error("Failed to initialise audit trail: %s", exc)
        return 1

    # Execute query
    records: list[dict[str, Any]] = []
    try:
        if args.round_id:
            records = audit.query_by_round_id(args.round_id)
        elif args.participant:
            records = audit.query_by_participant(args.participant)
        elif args.model_hash:
            records = audit.query_by_model_hash(args.model_hash)
        elif args.list:
            records = audit.list_all(limit=args.limit, offset=args.offset)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        return 1

    # Render output
    if args.output_json:
        _print_ndjson(records)
    elif args.detail:
        _print_detail(records)
    else:
        _print_table(records)

    print(f"\n{len(records)} record(s) returned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
