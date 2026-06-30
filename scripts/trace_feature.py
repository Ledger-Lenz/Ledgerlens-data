"""CLI to trace which trade IDs contributed to a specific feature value — Issue #244.

Reads the provenance JSON stored in the risk-score database for a wallet and
prints the Horizon paging-token trade IDs (and their Horizon explorer links)
that were used to compute the requested feature.

Usage
-----
    python -m scripts.trace_feature --wallet <STELLAR_ADDR> --feature benford_chi_square_24h
    python -m scripts.trace_feature --wallet <STELLAR_ADDR> --feature benford_mad_24h \\
        --asset-pair XLM/USDC

The command exits non-zero when:
  - No risk score record is found for the wallet.
  - The record has no provenance data (FEATURE_PROVENANCE_ENABLED was False).
  - The requested feature is not tracked (derived features are excluded).
"""

import argparse
import json
import sys

from config import config
from detection.persistence import get_session_factory
from detection.risk_score_store import RiskScoreStore
from utils.logging import get_logger

logger = get_logger(__name__)

HORIZON_BASE = config.HORIZON_URL.rstrip("/")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.trace_feature",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wallet", required=True, help="Stellar account address")
    parser.add_argument(
        "--feature",
        required=True,
        help="Feature name to trace (e.g. benford_chi_square_24h)",
    )
    parser.add_argument(
        "--asset-pair",
        default="XLM/native",
        help="Asset pair key stored in the risk score DB (default: XLM/native)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    store = RiskScoreStore(session_factory=get_session_factory())
    record = store.get(args.wallet, args.asset_pair)

    if record is None:
        print(
            f"No risk score record found for wallet {args.wallet!r} "
            f"and asset pair {args.asset_pair!r}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not record.provenance_json:
        print(
            "No provenance data stored for this record.\n"
            "Set FEATURE_PROVENANCE_ENABLED=true and re-score the wallet to collect provenance.",
            file=sys.stderr,
        )
        sys.exit(1)

    provenance: dict[str, list[str]] = json.loads(record.provenance_json)
    trade_ids = provenance.get(args.feature)

    if trade_ids is None:
        tracked = sorted(provenance.keys())
        print(
            f"Feature {args.feature!r} is not tracked in the provenance record.\n"
            f"Tracked features: {', '.join(tracked) if tracked else '(none)'}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not trade_ids:
        print(f"Feature {args.feature!r} has no contributing trade IDs recorded.")
        sys.exit(0)

    print(f"Feature:    {args.feature}")
    print(f"Wallet:     {args.wallet}")
    print(f"Asset pair: {args.asset_pair}")
    print(f"Trade IDs ({len(trade_ids)}):")
    for tid in trade_ids:
        print(f"  {tid}  {HORIZON_BASE}/trades/{tid}")


if __name__ == "__main__":
    main()
