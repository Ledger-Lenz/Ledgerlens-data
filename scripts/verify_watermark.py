"""CLI tool for verifying the watermark in a LedgerLens model artifact.

Usage:
    python -m scripts.verify_watermark --model-path models/random_forest.joblib \\
        [--trigger-path models/watermark_triggers.enc] \\
        [--target-label 1] \\
        [--threshold 0.9]

The MODEL_WATERMARK_KEY environment variable must be set to the 32-byte AES-256
key used when the triggers were encrypted.  The trigger vectors are secrets —
they must never be logged or exposed via the API.
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the watermark in a LedgerLens model artifact"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the .joblib model file to verify",
    )
    parser.add_argument(
        "--trigger-path",
        default=None,
        help="Path to the encrypted trigger vector file (default: from config)",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=1,
        help="Expected prediction label for trigger vectors (default: 1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum agreement fraction to consider watermark present (default: 0.9)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from detection.model_training import load_trigger_vectors
    from detection.persistence import verify_watermark

    model = joblib.load(args.model_path)
    triggers = load_trigger_vectors(args.trigger_path)

    result = verify_watermark(
        model=model,
        trigger_set=triggers,
        target_label=args.target_label,
        agreement_threshold=args.threshold,
    )

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        status = "DETECTED" if result["watermark_detected"] else "NOT DETECTED"
        print(f"Watermark: {status}")
        print(f"  Agreement : {result['agreement']:.4f} ({result['agreement']*100:.1f}%)")
        print(f"  Threshold : {result['threshold']:.2f}")
        print(f"  Triggers  : {result['n_triggers']}")

    sys.exit(0 if result["watermark_detected"] else 1)


if __name__ == "__main__":
    main()
