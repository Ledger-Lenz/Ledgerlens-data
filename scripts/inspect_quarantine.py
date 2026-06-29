"""CLI tool to review and manage quarantined samples.

Usage:
    python scripts/inspect_quarantine.py list
    python scripts/inspect_quarantine.py dismiss --wallet GA...
    python scripts/inspect_quarantine.py summary

This tool provides operators with visibility into samples flagged by
Activation Clustering (AC) backdoor detection, allowing them to review
the reasons for quarantine and override flags if needed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from detection.active_learning.annotation_queue import AnnotationQueue
from utils.logging import get_logger

logger = get_logger(__name__)


def list_quarantined(queue_path: str | None = None) -> None:
    """List all quarantined samples with details."""
    queue = AnnotationQueue(queue_path or "data/annotation_queue.json")
    quarantined = queue.quarantined_samples()

    if not quarantined:
        print("No quarantined samples found.")
        return

    print(f"\n{'='*100}")
    print(f"QUARANTINED SAMPLES ({len(quarantined)} total)")
    print(f"{'='*100}\n")

    for i, record in enumerate(quarantined, 1):
        print(f"{i}. Wallet: {record.get('wallet', 'N/A')}")
        print(f"   Asset Pair: {record.get('asset_pair', 'N/A')}")
        print(f"   Label: {record.get('label', 'N/A')} ({'clean' if record.get('label') == 0 else 'wash trade'})")
        print(f"   Annotator: {record.get('annotator_id', 'N/A')}")
        print(f"   Annotated: {record.get('annotated_at', 'N/A')}")
        print(f"   Quarantine Reason: {record.get('quarantine_reason', 'N/A')}")
        print(f"   Notes: {record.get('notes', 'N/A')}")
        print()


def print_summary(queue_path: str | None = None) -> None:
    """Print summary statistics of quarantined samples."""
    queue = AnnotationQueue(queue_path or "data/annotation_queue.json")
    quarantined = queue.quarantined_samples()

    if not quarantined:
        print("No quarantined samples.")
        return

    # Group by reason
    by_reason: dict = {}
    for record in quarantined:
        reason = record.get("quarantine_reason", "unknown")
        if reason not in by_reason:
            by_reason[reason] = 0
        by_reason[reason] += 1

    # Group by label
    by_label: dict = {}
    for record in quarantined:
        label = record.get("label", -1)
        label_name = "clean" if label == 0 else ("wash_trade" if label == 1 else "unknown")
        if label_name not in by_label:
            by_label[label_name] = 0
        by_label[label_name] += 1

    print(f"\n{'='*100}")
    print(f"QUARANTINE SUMMARY")
    print(f"{'='*100}\n")

    print(f"Total quarantined: {len(quarantined)}")
    print(f"\nBy Quarantine Reason:")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason}: {count}")

    print(f"\nBy Label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")

    print()


def dismiss_quarantine(wallet: str, queue_path: str | None = None) -> None:
    """Dismiss quarantine flag for a wallet (operator override)."""
    queue = AnnotationQueue(queue_path or "data/annotation_queue.json")
    quarantined = queue.quarantined_samples()

    # Find the wallet
    found = None
    for record in quarantined:
        if record.get("wallet") == wallet:
            found = record
            break

    if found is None:
        print(f"Wallet {wallet} not found in quarantine.")
        return

    print(f"\nDismissing quarantine for wallet: {wallet}")
    print(f"Original reason: {found.get('quarantine_reason', 'N/A')}")

    # Confirm
    response = input("Confirm dismissal? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("Cancelled.")
        return

    queue.dismiss_quarantine(wallet)
    print(f"Quarantine dismissed for {wallet}.")
    logger.info("Quarantine override: wallet=%s by operator", wallet)


def export_quarantined(output_path: str, queue_path: str | None = None) -> None:
    """Export quarantined samples to JSON for analysis."""
    queue = AnnotationQueue(queue_path or "data/annotation_queue.json")
    quarantined = queue.quarantined_samples()

    export_data = {
        "exported_at": datetime.now(UTC).isoformat(),
        "n_quarantined": len(quarantined),
        "samples": quarantined,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(quarantined)} quarantined samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Review and manage quarantined samples from backdoor detection"
    )
    parser.add_argument(
        "command",
        choices=["list", "summary", "dismiss", "export"],
        help="Command to run",
    )
    parser.add_argument(
        "--wallet",
        type=str,
        help="Wallet ID (required for dismiss command)",
    )
    parser.add_argument(
        "--queue-path",
        type=str,
        default="data/annotation_queue.json",
        help="Path to annotation queue file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/quarantined_export.json",
        help="Output path for export command",
    )

    args = parser.parse_args()

    try:
        if args.command == "list":
            list_quarantined(args.queue_path)
        elif args.command == "summary":
            print_summary(args.queue_path)
        elif args.command == "dismiss":
            if not args.wallet:
                print("Error: --wallet is required for dismiss command")
                return
            dismiss_quarantine(args.wallet, args.queue_path)
        elif args.command == "export":
            export_quarantined(args.output, args.queue_path)

    except Exception as e:
        logger.error("Command failed: %s", e)
        raise


if __name__ == "__main__":
    main()
