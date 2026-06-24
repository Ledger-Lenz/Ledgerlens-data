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
from datetime import UTC, datetime

import pandas as pd
from stellar_sdk import Asset as SdkAsset

from config import config
from detection.causal_attribution import CounterfactualAttributor
from detection.feature_engineering import build_feature_vector
from detection.forensic_report import ForensicReportGenerator, write_report_secure
from detection.model_inference import RiskScorer
from detection.shap_explainer import ShapExplainer
from ingestion.historical_loader import load_trades, trades_to_dataframe
from ingestion.orderbook_loader import (
    load_orderbook_events,
    orderbook_events_to_dataframe,
)


def validate_wallet_id(wallet_id: str) -> None:
    """Validate that wallet_id looks like a Stellar public key (56 chars, starts with G)."""
    if len(wallet_id) != 56 or not wallet_id.startswith("G"):
        print(f"Error: Invalid wallet ID format '{wallet_id}'.")
        print("Must be a 56-character Stellar public key starting with 'G'.")
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
            try:
                return SdkAsset(code, issuer)
            except Exception:
                # Placeholder/test issuer — Horizon will reject at API call time.
                _DUMMY = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
                return SdkAsset(code, _DUMMY)

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
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Include causal attribution in the output",
    )
    parser.add_argument(
        "--what-if-remove",
        default=None,
        help="Comma-separated trade IDs to remove for a counterfactual score",
    )
    return parser.parse_args()


def _parse_remove_trade_ids(
    remove_trade_ids: str | None, trades_df: pd.DataFrame, wallet: str
) -> list[str]:
    if not remove_trade_ids:
        return []

    requested = [trade_id.strip() for trade_id in remove_trade_ids.split(",") if trade_id.strip()]
    if not requested:
        return []

    if trades_df.empty or "trade_id" not in trades_df.columns:
        raise ValueError("Cannot remove trades: wallet trade history is empty")

    wallet_trade_ids = set(trades_df["trade_id"].astype(str))
    invalid = [trade_id for trade_id in requested if trade_id not in wallet_trade_ids]
    if invalid:
        raise ValueError(f"Trade IDs not found in wallet history: {', '.join(sorted(invalid))}")

    return requested


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
            print(
                "Suggestion: train models first by running model_training.py:"
                " python -m detection.model_training",
                file=sys.stderr,
            )
        sys.exit(1)

    override_val = scorer.list_override.check(args.wallet)
    if override_val in (0, 100):
        result = {
            "score": override_val,
            "benford_flag": False,
            "ml_flag": bool(override_val >= 50),
            "confidence": 100,
        }
        trades_df = pd.DataFrame()
        feature_row = pd.Series({"wallet": args.wallet})
        shap_explanations = []
        causal_result = None
    else:
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
        feature_vector = build_feature_vector(
            args.wallet, trades_df, orderbook_events=orderbook_events_df
        )
        feature_row = pd.Series(feature_vector)

        # 4. Score
        try:
            result = scorer.score(feature_row)
        except Exception as e:
            print(f"Error during scoring: {e}", file=sys.stderr)
            sys.exit(1)

        remove_trade_ids = []
        causal_result = None
        if args.what_if_remove or args.causal:
            try:
                remove_trade_ids = _parse_remove_trade_ids(
                    args.what_if_remove, trades_df, args.wallet
                )
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                raise

            attributor = CounterfactualAttributor(scorer)
            if remove_trade_ids:
                causal_result = attributor.counterfactual_score(
                    args.wallet,
                    trades_df,
                    remove_trade_ids,
                    orderbook_events=orderbook_events_df,
                )
            elif args.causal:
                causal_result = attributor.counterfactual_score(
                    args.wallet,
                    trades_df,
                    [],
                    orderbook_events=orderbook_events_df,
                )

        # 5. Explain
        try:
            explainer = ShapExplainer()
            models = scorer.models
            shap_explanations = explainer.explain_ensemble(feature_row, models, top_n=5)
        except Exception:
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
        if causal_result is not None:
            output["causal_attribution"] = causal_result
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

        if causal_result is not None:
            print("\nCausal attribution:")
            print(f"  Original score:        {causal_result['original_score']}")
            print(f"  Counterfactual score:   {causal_result['counterfactual_score']}")
            print(f"  Score delta:           {causal_result['score_delta']}")
            if causal_result["features_changed"]:
                print("  Features changed:")
                for name, details in causal_result["features_changed"].items():
                    print(f"    - {name}: {details['original']} -> {details['counterfactual']}")


if __name__ == "__main__":
    main()


def _generate_report(args, result, shap_explanations, trades_df, feature_row, scorer) -> None:
    """Generate and optionally anchor a forensic report."""
    from pathlib import Path

    generator = ForensicReportGenerator()

    model_metadata = {}
    if scorer.metadata:
        model_metadata = {
            "name": "LedgerLens Ensemble",
            "version": scorer.metadata.get("model_version", "unknown"),
            "training_dataset_sha256": scorer.metadata.get("training_dataset_sha256", "unknown"),
            "feature_schema_version": scorer.metadata.get("feature_schema_hash", "unknown"),
        }

    report = generator.generate(
        wallet=args.wallet,
        wallet_trades=trades_df,
        risk_score_dict=result,
        shap_values=shap_explanations,
        asset_pair=args.pair,
        model_metadata=model_metadata or None,
    )

    # Optional on-chain anchoring — only when --anchor is set
    if args.anchor:
        try:
            from integrations.contract_client import LedgerLensContractClient

            client = LedgerLensContractClient()
            tx_hash = client.anchor_report(report)
            print(f"Anchored to Soroban: {tx_hash}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: on-chain anchoring failed: {e}", file=sys.stderr)

    # Determine output path and format
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_wallet = args.wallet[:12]
    out_dir = Path("reports/forensic")
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.report_format
    if fmt == "json":
        out_path = out_dir / f"{safe_wallet}_{ts}.json"
        write_report_secure(str(out_path), json.dumps(report.to_dict(), indent=2))
    elif fmt == "markdown":
        out_path = out_dir / f"{safe_wallet}_{ts}.md"
        write_report_secure(str(out_path), report.to_markdown())
    elif fmt == "pdf":
        out_path = out_dir / f"{safe_wallet}_{ts}.pdf"
        report.to_pdf(str(out_path))

    print(f"Forensic report written to: {out_path}", file=sys.stderr)
