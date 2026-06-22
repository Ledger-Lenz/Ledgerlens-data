"""Bulk forensic report generator.

Reads a CSV of wallets and generates a forensic report for each one,
writing JSON output to reports/forensic/{wallet}_{timestamp}.json.

Usage:
    python -m scripts.generate_reports --input wallets.csv [--pair XLM:native] \\
        [--since 2024-01-01] [--anchor] [--output-dir reports/forensic]

CSV format (header row required):
    wallet[,asset_pair]

    wallet  — Stellar account ID (G...)
    pair    — optional asset pair; overridden by --pair if provided
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import config
from detection.feature_engineering import build_feature_vector
from detection.forensic_report import ForensicReportGenerator, write_report_secure
from detection.model_inference import RiskScorer
from detection.shap_explainer import ShapExplainer
from ingestion.historical_loader import load_trades, trades_to_dataframe
from ingestion.orderbook_loader import load_orderbook_events, orderbook_events_to_dataframe


def _score_wallet(
    wallet: str,
    pair: str,
    since: datetime | None,
    scorer: RiskScorer,
    generator: ForensicReportGenerator,
    output_dir: Path,
    anchor: bool,
) -> str:
    """Score one wallet and write its forensic report. Returns the output path."""
    # Ingest
    try:
        base_code, _, base_issuer = pair.split("/")[0].partition(":")
        from stellar_sdk import Asset as SdkAsset

        def _asset(s: str) -> SdkAsset:
            code, _, issuer = s.partition(":")
            return (
                SdkAsset.native()
                if issuer in ("native", "") and code in ("XLM", "")
                else SdkAsset(code, issuer)
            )

        parts = pair.split("/")
        base_asset = _asset(parts[0])
        counter_asset = _asset(parts[1]) if len(parts) > 1 else SdkAsset.native()

        trades = list(load_trades(base_asset, counter_asset, start_time=since))
        trades_df = trades_to_dataframe(trades)
        if not trades_df.empty:
            mask = (trades_df["base_account"] == wallet) | (trades_df["counter_account"] == wallet)
            trades_df = trades_df[mask]

        events = list(load_orderbook_events(wallet))
        orderbook_df = orderbook_events_to_dataframe(events)
    except Exception:
        trades_df = pd.DataFrame()
        orderbook_df = None

    # Feature engineering
    feature_vector = build_feature_vector(wallet, trades_df, orderbook_events=orderbook_df)
    feature_row = pd.Series(feature_vector)

    # Score
    result = scorer.score(feature_row)

    # SHAP
    try:
        explainer = ShapExplainer()
        shap_values = explainer.explain_ensemble(feature_row, scorer.models, top_n=10)
    except Exception:
        shap_values = []

    # Model metadata
    model_metadata = {}
    if scorer.metadata:
        model_metadata = {
            "name": "LedgerLens Ensemble",
            "version": scorer.metadata.get("model_version", "unknown"),
            "training_dataset_sha256": scorer.metadata.get("training_dataset_sha256", "unknown"),
            "feature_schema_version": scorer.metadata.get("feature_schema_hash", "unknown"),
        }

    report = generator.generate(
        wallet=wallet,
        wallet_trades=trades_df,
        risk_score_dict=result,
        shap_values=shap_values,
        asset_pair=pair,
        model_metadata=model_metadata or None,
    )

    # Optional on-chain anchor
    if anchor:
        try:
            from integrations.contract_client import LedgerLensContractClient

            client = LedgerLensContractClient()
            client.anchor_report(report)
        except Exception:
            pass  # anchoring failure must not abort report writing

    # Write report
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{wallet[:12]}_{ts}.json"
    write_report_secure(str(out_path), json.dumps(report.to_dict(), indent=2))
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk forensic report generator")
    parser.add_argument("--input", required=True, help="CSV file with wallet[,pair] rows")
    parser.add_argument("--pair", default="XLM:native", help="Default asset pair")
    parser.add_argument(
        "--since",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO date to start loading trades from",
    )
    parser.add_argument("--anchor", action="store_true", help="Anchor each report to Soroban")
    parser.add_argument(
        "--output-dir",
        default="reports/forensic",
        help="Output directory (default: reports/forensic)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load wallets from CSV
    wallets: list[tuple[str, str]] = []
    with open(args.input, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            wallet = row.get("wallet", "").strip()
            pair = row.get("pair", row.get("asset_pair", args.pair)).strip() or args.pair
            if wallet:
                wallets.append((wallet, pair))

    if not wallets:
        print("Error: no wallets found in input CSV", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        scorer = RiskScorer()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    generator = ForensicReportGenerator()

    try:
        from tqdm import tqdm

        progress = tqdm(total=len(wallets), unit="wallet")
    except ImportError:
        progress = None

    results: dict[str, str | Exception] = {}

    with ThreadPoolExecutor(max_workers=config.REPORT_CONCURRENCY) as pool:
        futures = {
            pool.submit(
                _score_wallet,
                wallet,
                pair,
                args.since,
                scorer,
                generator,
                output_dir,
                args.anchor,
            ): wallet
            for wallet, pair in wallets
        }
        for future in as_completed(futures):
            wallet = futures[future]
            try:
                out_path = future.result()
                results[wallet] = out_path
            except Exception as exc:
                results[wallet] = exc
                print(f"Error processing {wallet}: {exc}", file=sys.stderr)
            finally:
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    ok = sum(1 for v in results.values() if isinstance(v, str))
    print(f"Done: {ok}/{len(wallets)} reports written to {output_dir}")


if __name__ == "__main__":
    main()
