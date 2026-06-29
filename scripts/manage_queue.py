"""CLI tool for inspecting, editing, and exporting the active learning annotation queue.

Usage:
    # List pending wallets (default):
    python -m scripts.manage_queue list

    # List with filters:
    python -m scripts.manage_queue list --status pending --limit 10
    python -m scripts.manage_queue list --status annotated
    python -m scripts.manage_queue list --status skipped

    # Annotate a wallet:
    python -m scripts.manage_queue annotate GABCD... 1 --comment "obvious wash"
    python -m scripts.manage_queue annotate GABCD... 0 --comment "clean trade"

    # Skip a wallet:
    python -m scripts.manage_queue skip GABCD... --reason "insufficient data"

    # Export annotated records to CSV:
    python -m scripts.manage_queue export --output annotations.csv

    # Show queue statistics:
    python -m scripts.manage_queue stats
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from config import config
from detection.active_learning.annotation_queue import (
    _atomic_write,
    _compute_hmac,
    _load_queue,
    DEFAULT_QUEUE_PATH,
)
from detection.active_learning.queue_io import load_queue as load_queue_signed, save_queue
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_queue_path(args_queue: str | None = None) -> str:
    """Return the queue path from CLI arg or config default."""
    return args_queue or config.AL_QUEUE_PATH or DEFAULT_QUEUE_PATH


def _load_signed(queue_path: str) -> list[dict]:
    """Load queue using the signed (HMAC-verified) loader.

    Falls back to the unsigned loader if the signed loader fails or
    no HMAC secret is configured.
    """
    import json as _json

    secret = config.ANNOTATION_HMAC_SECRET
    path_obj = Path(queue_path)
    if not path_obj.exists():
        return []
    if secret:
        try:
            return load_queue_signed(path_obj, secret)
        except ValueError as exc:
            print(f"HMAC verification failed: {exc}", file=sys.stderr)
            print("Falling back to unsigned load.", file=sys.stderr)
        except Exception:
            pass
    # Unsigned fallback: read raw and handle both formats
    raw_data = _json.loads(path_obj.read_bytes())
    if isinstance(raw_data, dict):
        # Format from queue_io.save_queue: {"annotations": [...], "_hmac": "..."}
        return raw_data.get("annotations", [])
    if isinstance(raw_data, list):
        return raw_data
    return []


def _re_sign_and_save(queue_path: str, queue: list[dict]) -> None:
    """Re-sign the queue file with HMAC and write atomically.

    If ANNOTATION_HMAC_SECRET is configured, the queue is saved through
    the signed save_queue() path; otherwise it falls back to the plain
    atomic write.
    """
    secret = config.ANNOTATION_HMAC_SECRET
    path_obj = Path(queue_path)
    if secret:
        save_queue(path_obj, queue, secret)
    else:
        _atomic_write(queue_path, queue)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    """list -- show queue entries as a human-readable table."""
    queue = _load_signed(_get_queue_path(args.queue))
    status_filter = args.status
    limit = args.limit

    if status_filter:
        filtered = [item for item in queue if item.get("status") == status_filter]
    else:
        filtered = list(queue)

    if limit and limit > 0:
        filtered = filtered[:limit]

    if not filtered:
        print("No entries to display.")
        return

    # Build a table-friendly view
    rows: list[dict[str, Any]] = []
    for item in filtered:
        rows.append(
            {
                "wallet": item.get("wallet", ""),
                "status": item.get("status", ""),
                "label": item.get("label", ""),
                "annotator": item.get("annotator_id", ""),
                "strategy": item.get("query_strategy", ""),
                "selected_at": item.get("selected_at", ""),
                "annotated_at": item.get("annotated_at", ""),
            }
        )

    print(tabulate(rows, headers="keys", tablefmt="grid"))
    print(f"\nShowing {len(rows)} of {len(queue)} total entries.")


def cmd_annotate(args: argparse.Namespace) -> None:
    """annotate <wallet> <label> -- record an analyst verdict and re-sign."""
    wallet = args.wallet
    label = args.label
    comment = args.comment or ""

    # Validate label
    if label not in (0, 1):
        print("Error: label must be 0 (clean) or 1 (wash trading).", file=sys.stderr)
        sys.exit(1)

    # Determine annotator identity
    annotator_id = args.annotator_id or os.getenv("USER", "cli-user")

    queue_path = _get_queue_path(args.queue)
    queue_data = _load_signed(queue_path)

    annotated_at = datetime.now(UTC).isoformat()
    mac = _compute_hmac(wallet, label, annotator_id, annotated_at)

    found = False
    for item in queue_data:
        if item.get("wallet") == wallet:
            item.update(
                {
                    "label": label,
                    "annotator_id": annotator_id,
                    "notes": comment,
                    "annotated_at": annotated_at,
                    "status": "annotated",
                    "annotation_hmac": mac,
                }
            )
            found = True
            break

    if not found:
        queue_data.append(
            {
                "wallet": wallet,
                "asset_pair": "",
                "score": None,
                "query_strategy": "manual",
                "selected_at": annotated_at,
                "status": "annotated",
                "label": label,
                "annotator_id": annotator_id,
                "notes": comment,
                "annotated_at": annotated_at,
                "annotation_hmac": mac,
            }
        )

    _re_sign_and_save(queue_path, queue_data)
    print(f"Annotated {wallet} -> label={label} (annotator={annotator_id})")


def cmd_skip(args: argparse.Namespace) -> None:
    """skip <wallet> -- mark wallet as skipped and re-sign."""
    wallet = args.wallet
    reason = args.reason or ""

    queue_path = _get_queue_path(args.queue)
    queue_data = _load_signed(queue_path)

    found = False
    for item in queue_data:
        if item.get("wallet") == wallet:
            item["status"] = "skipped"
            if reason:
                item["skip_reason"] = reason
            found = True
            break

    if not found:
        print(f"Error: wallet {wallet} not found in the queue.", file=sys.stderr)
        sys.exit(1)

    _re_sign_and_save(queue_path, queue_data)
    msg = f"Skipped {wallet}"
    if reason:
        msg += f" (reason: {reason})"
    print(msg)


def cmd_export(args: argparse.Namespace) -> None:
    """export -- write annotated records to CSV."""
    output = args.output
    if not output:
        print("Error: --output is required for export.", file=sys.stderr)
        sys.exit(1)

    queue = _load_signed(_get_queue_path(args.queue))

    verified = []
    secret = config.ANNOTATION_HMAC_SECRET
    for item in queue:
        if item.get("status") != "annotated":
            continue
        if secret:
            expected = _compute_hmac(
                item.get("wallet", ""),
                item.get("label", -1),
                item.get("annotator_id", ""),
                item.get("annotated_at", ""),
            )
            import hmac as hmac_mod

            if not hmac_mod.compare_digest(expected, item.get("annotation_hmac", "")):
                logger.warning(
                    "Invalid HMAC for annotation wallet=%s -- excluded from export",
                    item.get("wallet"),
                )
                continue
        verified.append(item)

    fieldnames = ["wallet", "label", "annotator_id", "annotated_at", "notes"]
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in verified:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Exported {len(verified)} annotated records to {output}")


def cmd_stats(args: argparse.Namespace) -> None:
    """stats -- show queue statistics grouped by status, date, and annotator."""
    queue = _load_signed(_get_queue_path(args.queue))

    if not queue:
        print("Queue is empty.")
        return

    total = len(queue)

    status_counts: dict[str, int] = {}
    for item in queue:
        s = item.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print("=== Status Breakdown ===")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    annotator_counts: dict[str, int] = {}
    for item in queue:
        if item.get("status") == "annotated":
            a = item.get("annotator_id", "unknown")
            annotator_counts[a] = annotator_counts.get(a, 0) + 1

    if annotator_counts:
        print("\n=== By Annotator ===")
        for annotator, count in sorted(annotator_counts.items(), key=lambda x: -x[1]):
            print(f"  {annotator}: {count}")

    timestamps = [
        item.get("selected_at", "")
        for item in queue
        if item.get("selected_at")
    ]
    if timestamps:
        timestamps.sort()
        print("\n=== Date Range ===")
        print(f"  First selected: {timestamps[0]}")
        print(f"  Last selected:  {timestamps[-1]}")

    annotated_timestamps = [
        item.get("annotated_at", "")
        for item in queue
        if item.get("status") == "annotated" and item.get("annotated_at")
    ]
    if annotated_timestamps:
        annotated_timestamps.sort()
        print(f"  First annotated: {annotated_timestamps[0]}")
        print(f"  Last annotated:  {annotated_timestamps[-1]}")

    print(f"\nTotal entries: {total}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage the LedgerLens active learning annotation queue."
    )
    parser.add_argument(
        "--queue",
        default=None,
        help=f"Path to annotation queue JSON (default: {config.AL_QUEUE_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List queue entries")
    list_parser.add_argument(
        "--status",
        default=None,
        choices=["pending", "annotated", "skipped"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of entries to show (0 = no limit)",
    )

    # annotate
    annotate_parser = subparsers.add_parser("annotate", help="Annotate a wallet")
    annotate_parser.add_argument("wallet", help="Wallet ID to annotate")
    annotate_parser.add_argument(
        "label", type=int, choices=[0, 1], help="Label: 0 (clean) or 1 (wash trading)"
    )
    annotate_parser.add_argument("--comment", default="", help="Optional annotation comment")
    annotate_parser.add_argument(
        "--annotator-id",
        default="",
        help="Annotator identity (defaults to $USER)",
    )

    # skip
    skip_parser = subparsers.add_parser("skip", help="Skip a wallet")
    skip_parser.add_argument("wallet", help="Wallet ID to skip")
    skip_parser.add_argument("--reason", default="", help="Optional skip reason")

    # export
    export_parser = subparsers.add_parser("export", help="Export annotated records to CSV")
    export_parser.add_argument(
        "--output", required=True, help="Output CSV file path"
    )

    # stats
    subparsers.add_parser("stats", help="Show queue statistics")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    command_handlers = {
        "list": cmd_list,
        "annotate": cmd_annotate,
        "skip": cmd_skip,
        "export": cmd_export,
        "stats": cmd_stats,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()