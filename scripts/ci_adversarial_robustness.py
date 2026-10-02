"""Run a lightweight adversarial robustness regression gate for CI."""

from __future__ import annotations

import argparse
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from ci_metrics import CIRunRecord, MetricSnapshot, record_run
from scripts.adversarial_wash_trade_simulator import (
    _build_scorer,
    make_cross_chain_bridge_wash_trade,
    make_cross_venue_round_trip,
)


def evaluate(model_dir: str, cycles: int = 5) -> dict[str, float]:
    """Score both structural patterns against the latest available model."""
    scorer = _build_scorer(model_dir)
    venue = make_cross_venue_round_trip(cycles, seed=17)
    chain = make_cross_chain_bridge_wash_trade(cycles, seed=23)
    venue_score = float(scorer(venue))
    chain_score = float(scorer(chain))
    # Higher robustness means a higher adversarial risk score (the detector
    # still identifies the generated wash pattern).
    robustness = (venue_score + chain_score) / 2.0
    return {
        "adversarial_cross_venue_score": venue_score,
        "adversarial_cross_chain_score": chain_score,
        "adversarial_robustness_score": robustness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--store-path", default="ci_metrics/robustness_history.jsonl")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(args.model_dir, cycles=args.cycles)
    record = CIRunRecord(
        run_id=os.environ.get("GITHUB_RUN_ID", "local"),
        commit_sha=os.environ.get("GITHUB_SHA", "local"),
        branch=os.environ.get("GITHUB_REF_NAME", "local"),
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python_version=platform.python_version(),
        metrics=[
            MetricSnapshot(name=name, value=value, unit="score", higher_is_better=True)
            for name, value in metrics.items()
        ],
        extra={"suite": "cross-venue-and-cross-chain-wash-trade", "cycles": args.cycles},
    )
    alerts = record_run(
        record,
        store_path=Path(args.store_path),
        baseline_window=10,
        warning_pct=5.0,
        critical_pct=15.0,
        fail_on_critical=args.fail_on_critical,
    )
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")
    for alert in alerts:
        print(alert.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
