"""Score a single wallet on a single asset pair on demand.

Usage:
    python -m scripts.score_wallet \
      --wallet GABC1234... \
      --pair "USDC:GA5Z.../XLM:native" \
      --since 2024-01-01

This CLI loads historical trades and order-book events for a specific wallet,
builds its feature vector, scores it using the trained ensemble, computes
SHAP feature attributions, and prints the result to stdout.
"""

import argparse
import json
import sys
from datetime import datetime

import pandas as pd
from stellar_sdk import Asset as SdkAsset

from config import config
from detection.feature_engineering import build_feature_vector
from detection.model_inference import RiskScorer
from detection.shap_explainer import ShapExplainer
from ingestion.historical_loader import load_trades, trades_to_dataframe
from ingestion.orderbook_loader import (
    load_orderbook_events,
    orderbook_events_to_dataframe,
)


def validate_wallet_id(wallet_id: str) -> None:
    """Validate that wallet_id is a 56-character Stellar public key."""
    if len(wallet_id) != 56 or not wallet_id.startswith("G"):
        print(f"Error: Invalid wallet ID format '{wallet_id}'.")
        print("Must be a 56-character Stellar public key starting with 'G'.")
        sys.exit(1)
    # Basic base32 check
    try:
        import base64

        base64.b32decode(wallet_id + "======")
    except Exception:
        print(f"Error: Invalid wallet ID format '{wallet_id}' (not base32).")
        sys.exit(1)


def parse_asset_pair(pair_str: str) -> tuple[SdkAsset, SdkAsset]:
    """Parse a pair string like 'CODE:ISSUER/CODE:ISSUER' or 'CODE:ISSUER' (assumes XLM counter)."""
    try:
        if "/" in pair_str:
            base_str, counter_str = pair_str.split("/")
        else:
            base_str, counter_str = pair_str, "XLM:native"

        def _to_sdk_asset(s: str) -> SdkAsset:
            code, _, issuer = s.partition(":")
            if issuer == "native" or code == "XLM":
                return SdkAsset.native()
            return SdkAsset(code, issuer)

        return _to_sdk_asset(base_str), _to_sdk_asset(counter_str)
    except Exception as e:
        print(f"Error: Invalid asset pair format '{pair_str}': {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a single wallet on demand")
    parser.add_argument("--wallet", required=True, help="Stellar wallet public key (G...)")
    parser.add_argument(
        "--pair",
        required=True,
        help="Asset pair to score (e.g. 'USDC:GA5Z.../XLM:native')",
    )
    parser.add_argument(
        "--since",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO date to start loading trades from",
    )
    parser.add_argument(
        "--no-orderbook",
        action="store_true",
        help="Skip loading order-book events",
    )
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_wallet_id(args.wallet)
    base_asset, counter_asset = parse_asset_pair(args.pair)

    # 1. Load models
    try:
        scorer = RiskScorer()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        if "No trained models" in str(e):
            print("Suggestion: Run 'python -m detection.model_training' first.", file=sys.stderr)
        sys.exit(1)

    # 2. Ingest
    try:
        trades = list(load_trades(base_asset, counter_asset, start_time=args.since))
        trades_df = trades_to_dataframe(trades)

        # Filter trades to only those involving the target wallet
        if not trades_df.empty:
            mask = (trades_df["base_account"] == args.wallet) | (
                trades_df["counter_account"] == args.wallet
            )
            trades_df = trades_df[mask]

        orderbook_events_df = None
        if not args.no_orderbook:
            events = list(load_orderbook_events(args.wallet))
            orderbook_events_df = orderbook_events_to_dataframe(events)

    except Exception as e:
        print(f"Error fetching data from Horizon: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Feature Engineering
    feature_vector = build_feature_vector(args.wallet, trades_df, orderbook_events=orderbook_events_df)
    feature_row = pd.Series(feature_vector)

    # 4. Score
    try:
        result = scorer.score(feature_row)
    except Exception as e:
        print(f"Error during scoring: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Explain
    try:
        explainer = ShapExplainer()
        models = scorer.models
        shap_explanations = explainer.explain_ensemble(feature_row, models, top_n=5)
    except Exception as e:
        # Fallback: empty explanations if SHAP fails
        shap_explanations = []

    # 6. Output
    if args.json:
        output = {
            "wallet": args.wallet,
            "asset_pair": args.pair,
            "score": result["score"],
            "benford_flag": result["benford_flag"],
            "ml_flag": result["ml_flag"],
            "confidence": result["confidence"],
            "shap_explanations": shap_explanations,
        }
        print(json.dumps(output, indent=2))
    else:
        status = "FLAGGED" if result["score"] >= config.RISK_SCORE_FLAG_THRESHOLD else "OK"
        print(f"Wallet:   {args.wallet}")
        print(f"Pair:     {args.pair}")
        print(f"Score:    {result['score']}  [{status}]")
        print(f"Benford:  {result['benford_flag']}")
        print(f"ML:       {result['ml_flag']} (confidence {result['confidence']})")
        print("\nTop 5 SHAP contributors:")
        for i, exp in enumerate(shap_explanations, 1):
            contrib = f"{exp['contribution']:+.2f}"
            print(f"  {i}. {exp['feature']:<25} {contrib:>6}  (value: {exp['value']:.4g})")


if __name__ == "__main__":
    main()
