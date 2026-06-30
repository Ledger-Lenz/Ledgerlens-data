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
from detection.forensic_report import (
    ForensicReportGenerator,
    write_csv_report,
    write_report_secure,
)
from detection.model_inference import RiskScorer
from ingestion.historical_loader import load_trades, trades_to_dataframe
from ingestion.orderbook_loader import load_orderbook_events, orderbook_events_to_dataframe
from utils.logging import get_logger

logger = get_logger(__name__)


def _score_wallet(
    wallet: str,
    pair: str,
    since: datetime | None,
    scorer: RiskScorer,
    generator: ForensicReportGenerator,
    output_dir: Path,
    anchor: bool,
    output_format: str = "json",
    output_file: str | None = None,
) -> str:
    """Score one wallet and write its forensic report. Returns the output path.

    Args:
        output_format: ``"json"`` or ``"csv"``.
        output_file:   Explicit destination path, or ``"-"`` for stdout.
                       When *None* an auto-generated filename in *output_dir*
                       is used.
    """
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

    report = generator.generate(
        wallet=wallet,
        asset_pair=pair,
        feature_row=feature_row,
        wallet_trades=trades_df,
        orderbook_events=orderbook_df,
    )

    model_version = (
        scorer.metadata.get("model_version", "unknown") if scorer.metadata else "unknown"
    )
    try:
        from detection.audit_trail import commit_forensic_report

        commit_forensic_report(report, feature_vector, model_version)
    except Exception as exc:
        logger.warning("Audit trail commit skipped: %s", exc)

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
    _ext_map = {"csv": "csv", "html": "html"}
    ext = _ext_map.get(output_format, "json")

    if output_file == "-":
        # --- stdout ---
        if output_format == "csv":
            import csv
            import io
            from detection.forensic_report import CSV_COLUMNS

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(report.to_csv_rows())
            sys.stdout.write(buf.getvalue())
        else:
            sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
        return "<stdout>"

    if output_file:
        out_path = Path(output_file)
    else:
        out_path = output_dir / f"{wallet[:12]}_{ts}.{ext}"

    if output_format == "csv":
        write_csv_report(str(out_path), report)
    elif output_format == "html":
        from detection.forensic_report_interactive import generate_interactive_report

        generate_interactive_report(report.to_dict(), str(out_path))
    else:
        write_report_secure(str(out_path), json.dumps(report.to_dict(), indent=2))
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk forensic report generator")
    parser.add_argument("--input", default=None, help="CSV file with wallet[,pair] rows (required unless --simulate)")
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
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "html"],
        default="json",
        help="Export format for forensic reports: json (default), csv, or html (interactive)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Write output to this file path instead of the auto-generated path. "
            "Use '-' to write to stdout. "
            "When processing multiple wallets this path is used as a prefix "
            "(<path>_<wallet>_<ts>.<ext>)."
        ),
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Run the incident-response playbook against --wallet without waiting "
            "for a live alert.  Requires --wallet; ignores --input."
        ),
    )
    parser.add_argument(
        "--wallet",
        default=None,
        help="Stellar account ID to use with --simulate.",
    )
    return parser.parse_args()


def _run_simulate(args: argparse.Namespace) -> None:
    """Execute incident-response playbook simulation and generate a report."""
    if not args.wallet:
        print("Error: --simulate requires --wallet <WALLET_ADDRESS>", file=sys.stderr)
        sys.exit(1)

    from monitoring.incident_responder import IncidentResponder

    responder = IncidentResponder()
    incident = responder.simulate(
        wallet=args.wallet,
        risk_score=95,
        alert_type="high_risk_wallet",
    )
    if incident is None:
        print("Simulation produced no incident (duplicate suppressed or below threshold).", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    report_dict = incident.to_dict()

    if args.output_format == "html":
        from detection.forensic_report_interactive import generate_interactive_report

        out_path = output_dir / f"simulate_{args.wallet[:12]}_{ts}.html"
        generate_interactive_report(report_dict, str(out_path))
    else:
        out_path = output_dir / f"simulate_{args.wallet[:12]}_{ts}.json"
        out_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    print(f"Simulation complete: incident_id={incident.incident_id} report={out_path}")


def main() -> None:
    args = parse_args()

    if args.simulate:
        _run_simulate(args)
        return

    if not args.input:
        print("Error: --input is required when --simulate is not set", file=sys.stderr)
        sys.exit(1)

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
                args.output_format,
                args.output_file,
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
