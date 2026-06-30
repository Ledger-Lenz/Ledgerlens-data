"""Derive empirical feature ranges from the synthetic dataset.

Reads ``data/synthetic_dataset.parquet`` (or a path supplied via ``--data``),
computes per-feature statistics (min, max, p1, p99, mean, std), and writes the
result to ``data/feature_ranges.json``.

The JSON file is consumed by ``detection.feature_engineering.validate_feature_ranges``
to flag out-of-range values at inference time, and by contributors updating the
``data/feature_dictionary.md`` range column.

Usage::

    python -m scripts.compute_feature_ranges
    python -m scripts.compute_feature_ranges --data data/synthetic_dataset.parquet \\
                                              --output data/feature_ranges.json

Exit codes:
    0  Success — ``data/feature_ranges.json`` written.
    1  Fatal error (file not found, empty dataset, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are not features and must be excluded from range computation.
_NON_FEATURE_COLS: frozenset[str] = frozenset({"wallet", "label", "profile"})


def compute_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return a dict mapping feature name → range statistics.

    For each numeric feature column (excluding label/wallet/profile) the
    following statistics are computed over all non-null values:

    ``min``    Empirical minimum observed in the dataset.
    ``max``    Empirical maximum observed in the dataset.
    ``p1``     1st percentile (robust lower bound; use for soft validation).
    ``p99``    99th percentile (robust upper bound; use for soft validation).
    ``mean``   Arithmetic mean.
    ``std``    Population standard deviation.
    ``n``      Number of non-null observations.

    The *hard* validation bounds used by ``validate_feature_ranges`` are the
    *theoretical* min/max stored in ``data/feature_dictionary.md`` (and
    re-exported by ``FEATURE_RANGES`` in ``detection.feature_engineering``).
    The empirical values here are informational and can be used to regenerate
    that table when the dataset changes.
    """
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in _NON_FEATURE_COLS
    ]

    ranges: dict[str, dict[str, float]] = {}
    for col in sorted(numeric_cols):
        series = df[col].dropna()
        if series.empty:
            continue
        ranges[col] = {
            "min":  round(float(series.min()), 8),
            "max":  round(float(series.max()), 8),
            "p1":   round(float(series.quantile(0.01)), 8),
            "p99":  round(float(series.quantile(0.99)), 8),
            "mean": round(float(series.mean()), 8),
            "std":  round(float(series.std()), 8),
            "n":    int(series.count()),
        }

    return ranges


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/synthetic_dataset.parquet",
        help="Path to the labelled feature parquet (default: data/synthetic_dataset.parquet)",
    )
    parser.add_argument(
        "--output",
        default="data/feature_ranges.json",
        help="Destination JSON path (default: data/feature_ranges.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Write indented JSON (default: True)",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        return 1

    logger.info("Loading dataset from %s …", data_path)
    try:
        df = pd.read_parquet(data_path)
    except Exception as exc:
        logger.error("Failed to read parquet: %s", exc)
        return 1

    if df.empty:
        logger.error("Dataset is empty: %s", data_path)
        return 1

    logger.info("Dataset shape: %s", df.shape)

    ranges = compute_ranges(df)
    logger.info("Computed ranges for %d features.", len(ranges))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(ranges, fh, indent=2 if args.pretty else None)
        fh.write("\n")

    logger.info("Wrote feature ranges to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
