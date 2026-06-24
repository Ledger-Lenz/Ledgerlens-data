"""Interactive CLI annotation interface for the LedgerLens active learning pipeline.

Usage:
    # Annotate next pending wallets in the queue:
    python -m scripts.annotate --annotator-id alice

    # Re-annotate previously skipped wallets:
    python -m scripts.annotate --annotator-id alice --replay

    # Export annotation queue to parquet:
    python -m scripts.annotate --export data/annotated.parquet

    # Annotate from a specific queue file:
    python -m scripts.annotate --annotator-id alice --queue data/my_queue.json
"""

from __future__ import annotations

import argparse
import sys

from config import config
from detection.active_learning.annotation_queue import AnnotationQueue
from utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_MAP = {"w": 1, "c": 0}


def _prompt_label(wallet: str, item: dict) -> str | None:
    """Display wallet info and prompt for a label. Returns 'w', 'c', 's', or 'q'."""
    print("\n" + "=" * 60)
    print(f"Wallet : {wallet}")
    print(f"Score  : {item.get('score', 'N/A')}")
    print(f"Strategy: {item.get('query_strategy', 'N/A')}")
    print(f"Asset Pair: {item.get('asset_pair', 'N/A')}")

    shap_top = item.get("shap_top3", [])
    if shap_top:
        print("SHAP top-3 features:")
        for feat in shap_top:
            direction = "↑ wash" if feat.get("contribution", 0) > 0 else "↓ clean"
            print(
                f"  {feat['feature']}={feat.get('value', '?'):.3g}  "
                f"({direction}, contribution={feat.get('contribution', 0):+.3f})"
            )

    trades = item.get("recent_trades", [])
    if trades:
        print("Last trades:")
        for t in trades[:5]:
            print(f"  {t}")

    while True:
        answer = input("\nLabel [w=wash, c=clean, s=skip, q=quit]: ").strip().lower()
        if answer in ("w", "c", "s", "q"):
            return answer
        print("  Please enter w, c, s, or q.")


def run_annotation_loop(
    queue: AnnotationQueue,
    annotator_id: str,
    replay: bool = False,
    batch_size: int = 20,
) -> None:
    """Interactively annotate wallets from the queue."""
    if replay:
        skipped = queue.skipped_wallets()
        # Re-push skipped wallets as pending (re-annotate flow)
        items_to_annotate = []
        queue.pop_batch(0)  # load all pending
        all_items = _load_all(queue.queue_path)
        items_to_annotate = [i for i in all_items if i["wallet"] in skipped]
    else:
        items_to_annotate = queue.pop_batch(batch_size)

    if not items_to_annotate:
        print("No wallets to annotate. Run scripts/run_active_learning.py to populate the queue.")
        return

    for item in items_to_annotate:
        wallet = item["wallet"]
        answer = _prompt_label(wallet, item)

        if answer == "q":
            print("Exiting annotation session.")
            break
        elif answer == "s":
            queue.skip(wallet)
            print(f"  Skipped {wallet}")
        else:
            label = _label_map_get(answer)
            notes = input("  Notes (optional): ").strip()
            queue.annotate(wallet, label=label, annotator_id=annotator_id, notes=notes)
            print(f"  Annotated {wallet} → label={label}")

    print("\nAnnotation session complete.")


def _load_all(path: str) -> list[dict]:
    import json
    import os

    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _label_map_get(answer: str) -> int:
    return _LABEL_MAP[answer]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LedgerLens annotation CLI")
    parser.add_argument(
        "--annotator-id",
        default="",
        help="Non-empty annotator identifier (required unless --export)",
    )
    parser.add_argument(
        "--queue", default=config.AL_QUEUE_PATH, help="Path to annotation queue JSON"
    )
    parser.add_argument(
        "--replay", action="store_true", help="Re-annotate previously skipped wallets"
    )
    parser.add_argument("--export", default="", help="Export queue to this parquet path and exit")
    parser.add_argument("--batch-size", type=int, default=config.AL_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = AnnotationQueue(queue_path=args.queue)

    if args.export:
        df = queue.export_labelled(args.export)
        print(f"Exported {len(df)} annotated rows to {args.export}")
        return

    if not args.annotator_id:
        print("Error: --annotator-id is required for annotation sessions.", file=sys.stderr)
        sys.exit(1)

    run_annotation_loop(
        queue,
        annotator_id=args.annotator_id,
        replay=args.replay,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
