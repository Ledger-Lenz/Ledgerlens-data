"""Score one wallet, or many, on a single asset pair.

Usage:
    python -m scripts.score_wallet \
      --wallet GABC1234... \
      --pair "USDC:GA5Z.../XLM:native" \
      --since 2024-01-01

    python -m scripts.score_wallet \
      --wallets-file wallets.txt \
      --pair "USDC:GA5Z.../XLM:native" \
      --workers 8

This CLI loads historical trades and order-book events for a specific wallet,
builds its feature vector, scores it using the trained ensemble, computes
SHAP feature attributions, and prints the result to stdout.

In batch mode (`--wallets-file`), order-book events and SHAP attribution are
skipped for speed; each wallet is scored concurrently and one NDJSON line
`{"wallet": ..., "score": ..., "error": null}` is printed per wallet as soon
as it finishes. A failure for one wallet is reported via its `error` field
and never aborts the rest of the batch.
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from stellar_sdk import Asset as SdkAsset

from config import config
from utils.logging import get_logger, set_level
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


logger = get_logger(__name__)


def validate_wallet_address(wallet_id: str) -> None:
    """Validate that wallet_id looks like a Stellar public key (56 chars, starts with G)."""
    if len(wallet_id) != 56 or not wallet_id.startswith("G"):
        raise ValueError(
            f"Invalid Stellar address '{wallet_id}'. Must be a 56-character public key starting with 'G'."
        )


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
        logger.error("Invalid asset pair format", exc_info=True, extra={
            "wallet": "unknown",
            "error_type": type(e).__name__,
            "error_message": f"Invalid asset pair format '{pair_str}': {e}"
        })
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one wallet, or many, on demand")
    wallet_group = parser.add_mutually_exclusive_group(required=True)
    wallet_group.add_argument("--wallet", help="Stellar wallet public key (G...)")
    wallet_group.add_argument(
        "--wallets-file",
        type=Path,
        help="Path to a file of wallet addresses (one per line; blank lines "
        "and lines starting with '#' are skipped) to score concurrently",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers to use with --wallets-file",
    )
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
        "-q",
        "--quiet",
        action="store_true",
        help=(
            "Suppress all log output and print only a single-line JSON result "
            "to stdout (implies --json). Useful for shell pipelines."
        ),
    )
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for status/progress messages, written to stderr (default: INFO)",
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


def _load_wallets_from_file(path: Path) -> list[str]:
    """Read wallet addresses from `path`, one per line.

    Blank lines and lines starting with '#' are skipped.
    """
    wallets = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            wallets.append(line)
    return wallets


def score_one(
    wallet: str,
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    scorer: RiskScorer,
    since: datetime | None,
) -> dict:
    """Score a single wallet for batch mode.

    Never raises: any failure (invalid address, ingestion error, scoring
    error) is captured in the returned dict's `error` field so one bad
    wallet can't abort the rest of the batch.
    """
    try:
        validate_wallet_address(wallet)

        override_val = scorer.list_override.check(wallet)
        if override_val in (0, 100):
            return {
                "wallet": wallet,
                "score": override_val,
                "benford_flag": False,
                "ml_flag": bool(override_val >= 50),
                "confidence": 100,
                "error": None,
            }

        trades = list(load_trades(base_asset, counter_asset, start_time=since))
        trades_df = trades_to_dataframe(trades)
        if not trades_df.empty:
            mask = (trades_df["base_account"] == wallet) | (
                trades_df["counter_account"] == wallet
            )
            trades_df = trades_df[mask]

        feature_vector = build_feature_vector(wallet, trades_df, orderbook_events=None)
        feature_row = pd.Series(feature_vector)
        result = scorer.score(feature_row)

        return {
            "wallet": wallet,
            "score": result["score"],
            "benford_flag": result["benford_flag"],
            "ml_flag": result["ml_flag"],
            "confidence": result["confidence"],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — per-wallet failures must not crash the batch
        return {"wallet": wallet, "score": None, "error": str(exc)}


def run_batch(args: argparse.Namespace) -> None:
    """Score every wallet in `args.wallets_file` concurrently, printing one
    NDJSON line per wallet to stdout as soon as it finishes."""
    wallets = _load_wallets_from_file(args.wallets_file)
    base_asset, counter_asset = parse_asset_pair(args.pair)

    try:
        scorer = RiskScorer()
    except RuntimeError as e:
        logger.error("Model load error", exc_info=True, extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
        })
        if "No trained models" in str(e):
            logger.info("Suggestion: train models first by running model_training.py: python -m detection.model_training")
        sys.exit(1)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_one, w, base_asset, counter_asset, scorer, args.since): w
            for w in wallets
        }
        for future in as_completed(futures):
            print(json.dumps(future.result()))


def main() -> None:
    args = parse_args()
    set_level(args.log_level)

    if args.quiet:
        logging.disable(logging.CRITICAL)

    if args.wallets_file:
        run_batch(args)
        return

    validate_wallet_address(args.wallet)
    base_asset, counter_asset = parse_asset_pair(args.pair)

    # 1. Load models
    try:
        scorer = RiskScorer()
    except RuntimeError as e:
        logger.error("Model load error", exc_info=True, extra={
            "wallet": args.wallet,
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        if "No trained models" in str(e):
            logger.info("Suggestion: train models first by running model_training.py: python -m detection.model_training", extra={"wallet": args.wallet})
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
            logger.error("Error fetching data from Horizon", exc_info=True, extra={
                "wallet": args.wallet,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            sys.exit(1)

        # 3. Feature Engineering
        feature_vector = build_feature_vector(
            args.wallet, trades_df, orderbook_events=orderbook_events_df
        )
        feature_row = pd.Series(feature_vector)
        logger.debug("Feature vector for %s: %s", args.wallet, feature_vector)

        # 4. Score
        try:
            t0 = time.time()
            result = scorer.score(feature_row)
            latency_ms = (time.time() - t0) * 1000
            model_version = scorer.metadata.get("model_version", "unknown") if scorer.metadata else "unknown"
            logger.info("Wallet scored", extra={
                "wallet": args.wallet,
                "score": result["score"],
                "latency_ms": latency_ms,
                "model_version": model_version,
                "asset_pair": args.pair
            })
        except Exception as e:
            logger.error("Error during scoring", exc_info=True, extra={
                "wallet": args.wallet,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            sys.exit(1)

        remove_trade_ids = []
        causal_result = None
        if args.what_if_remove or args.causal:
            try:
                remove_trade_ids = _parse_remove_trade_ids(
                    args.what_if_remove, trades_df, args.wallet
                )
            except ValueError as exc:
                logger.error("Error parsing what_if", exc_info=True, extra={
                    "wallet": args.wallet,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                })
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
    if args.json or args.quiet:
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
        if args.quiet:
            print(json.dumps(output, separators=(",", ":")))
        else:
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
            logger.info("Anchored to Soroban", extra={"tx_hash": tx_hash})
        except Exception as e:
            logger.warning("on-chain anchoring failed", exc_info=True, extra={
                "wallet": args.wallet,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

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

    logger.info("Forensic report written", extra={"out_path": str(out_path)})
