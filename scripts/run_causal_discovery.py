"""CLI: run causal discovery on a feature Parquet file.

    python -m scripts.run_causal_discovery \\
        --data data/synthetic_dataset.parquet \\
        --significance-level 0.05 \\
        --max-cond-set-size 3

Output:
    analysis/feature_dag.dot
    analysis/feature_dag.json
"""

import argparse
import os
import sys

import pandas as pd

# Security: reject absolute paths outside the project data directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ALLOWED_ROOT = os.path.join(_PROJECT_ROOT, "data")


def _validate_data_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(_ALLOWED_ROOT):
        print(
            f"ERROR: Data path must be inside {_ALLOWED_ROOT!r}. "
            f"Got: {abs_path!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.exists(abs_path):
        print(f"ERROR: File not found: {abs_path!r}", file=sys.stderr)
        sys.exit(1)
    return abs_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PC-algorithm causal discovery on a LedgerLens feature Parquet file."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a feature Parquet file (must be inside the project data/ directory).",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Independence test significance level α (default: 0.05).",
    )
    parser.add_argument(
        "--max-cond-set-size",
        type=int,
        default=3,
        metavar="D",
        help="Maximum conditioning set size for the PC skeleton search (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        metavar="SECONDS",
        help="Hard time limit in seconds (default: 600). Partial DAG is returned on timeout.",
    )
    args = parser.parse_args()

    data_path = _validate_data_path(args.data)

    df = pd.read_parquet(data_path)

    # Drop non-feature columns
    _EXCLUDE = {"wallet", "label", "profile", "asset_pair"}
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    features_df = df[feature_cols]

    from analysis.causal_discovery import discover_feature_dag

    adj = discover_feature_dag(
        features_df,
        significance_level=args.significance_level,
        max_cond_set_size=args.max_cond_set_size,
        timeout_seconds=args.timeout,
    )

    edge_count = sum(len(v) for v in adj.values())
    print(f"Causal discovery complete: {edge_count} edge(s) found.")
    for src, targets in adj.items():
        for tgt in targets:
            print(f"  {src} → {tgt}")


if __name__ == "__main__":
    main()
