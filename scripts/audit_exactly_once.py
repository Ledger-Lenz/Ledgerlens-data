#!/usr/bin/env python3
"""Audit exactly-once invariants at every pipeline stage boundary (Issue #918).

Builds each stage's dedup store from the same configuration the stage itself
reads, traces a sample of synthetic records through them in pipeline order,
and exits non-zero if any boundary has degraded to at-least-once or
at-most-once behaviour. Runs on a schedule against staging via
``.github/workflows/exactly-once-audit.yml``. See docs/exactly_once_audit.md.

Usage:
    python -m scripts.audit_exactly_once --redis-url redis://staging:6379/0 \\
        --db-url postgresql://.../ledgerlens --output reports/exactly_once_audit.json

Exit codes:
    0  All invariants hold at every boundary.
    1  At least one violation was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import config
from pipeline.exactly_once import (
    ExactlyOnceStore,
    RedisExactlyOnceBackend,
    SqlExactlyOnceBackend,
)
from pipeline.exactly_once_audit import StageBoundary, audit_pipeline
from utils.logging import get_logger

logger = get_logger(__name__)


def build_stage_boundaries(
    redis_url: str, db_url: str, min_ttl_seconds: float
) -> list[StageBoundary]:
    """Stage boundaries in pipeline order, configured as each stage configures them.

    ingestion        ingestion/trade_deduplicator.py  (Horizon trade dedup)
    feature_scoring  streaming/kafka_worker.py        (feature update + scoring per message)
    alerting         streaming/alert_ledger.py        (alert delivery outcomes)
    """
    ingestion = ExactlyOnceStore(
        RedisExactlyOnceBackend(redis_url, key_prefix=config.TRADE_DEDUP_CACHE_KEY_PREFIX),
        ttl_seconds=float(config.TRADE_DEDUP_TTL_SECONDS),
    )
    feature_scoring = ExactlyOnceStore(
        RedisExactlyOnceBackend(redis_url, key_prefix="ledgerlens:kafka_dedup:"),
        ttl_seconds=float(config.KAFKA_DEDUP_TTL_SECONDS),
    )
    alerting = ExactlyOnceStore(SqlExactlyOnceBackend(db_url))
    return [
        StageBoundary("ingestion", ingestion, "horizon_trade:audit", min_ttl_seconds),
        StageBoundary("feature_scoring", feature_scoring, "kafka_trade", min_ttl_seconds),
        StageBoundary("alerting", alerting, "alert_delivery", min_ttl_seconds),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--redis-url", default=None, help=f"Redis for stream dedup (default: {config.REDIS_URL})"
    )
    parser.add_argument(
        "--db-url", default=None, help="Database for the alert ledger (default: RISK_SCORE_DB_URL)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=5, help="Records traced end-to-end (default: 5)"
    )
    parser.add_argument(
        "--min-ttl-seconds",
        type=float,
        default=3600.0,
        help="Redelivery window every stage's dedup TTL must cover (default: 3600)",
    )
    parser.add_argument("--output", default=None, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)

    boundaries = build_stage_boundaries(
        args.redis_url or config.REDIS_URL,
        args.db_url or config.RISK_SCORE_DB_URL,
        args.min_ttl_seconds,
    )
    report = audit_pipeline(boundaries, sample_size=args.sample_size)

    rendered = json.dumps(report.to_dict(), indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
