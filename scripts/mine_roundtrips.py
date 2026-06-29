"""Detect round-trip trade pairs in Stellar SDEX trade history.

A round-trip is defined as a wallet pair (A, B) where:
  - A sells asset X to B
  - B sells asset X back to A
  - Both trades occur within `max_ledger_window` ledger closes (~5 s each)
  - The amounts are within `amount_tolerance` of each other

Usage:
    python -m scripts.mine_roundtrips \
        --input data/raw_trades.parquet \
        --output data/roundtrip_pairs.parquet \
        --max-ledger-window 100 \
        --amount-tolerance 0.05
"""

from __future__ import annotations

import argparse

import pandas as pd

# Approximate seconds per ledger close on the Stellar network.
SECONDS_PER_LEDGER = 5


def detect_roundtrip_pairs(
    trades: pd.DataFrame,
    max_ledger_window: int = 100,
    amount_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Detect round-trip wallet pairs within a sliding time window.

    Parameters
    ----------
    trades:
        DataFrame with columns: trade_id, ledger_close_time, base_account,
        counter_account, base_asset, counter_asset, amount (or base_amount).
    max_ledger_window:
        Maximum number of ledger closes allowed between the forward and
        return leg. At ~5 s/ledger this is ~8 minutes for the default of 100.
    amount_tolerance:
        Maximum fractional difference between the forward and return amounts
        (e.g. 0.05 = ±5%).

    Returns
    -------
    DataFrame with columns:
        wallet_a, wallet_b, forward_trade_id, return_trade_id,
        forward_time, return_time, forward_amount, return_amount,
        asset, elapsed_seconds
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "wallet_a",
                "wallet_b",
                "forward_trade_id",
                "return_trade_id",
                "forward_time",
                "return_time",
                "forward_amount",
                "return_amount",
                "asset",
                "elapsed_seconds",
            ]
        )

    df = trades.copy()
    df["ledger_close_time"] = pd.to_datetime(df["ledger_close_time"], utc=True)

    # Resolve the amount column (supports both column names used in the codebase)
    if "amount" not in df.columns and "base_amount" in df.columns:
        df["amount"] = df["base_amount"]

    max_seconds = max_ledger_window * SECONDS_PER_LEDGER
    records = []

    for _idx, fwd in df.iterrows():
        # Find candidate return legs: counter trades within the time window
        # where base_account / counter_account are swapped and asset matches.
        window_end = fwd["ledger_close_time"] + pd.Timedelta(seconds=max_seconds)
        candidates = df[
            (df["ledger_close_time"] > fwd["ledger_close_time"])
            & (df["ledger_close_time"] <= window_end)
            & (df["base_account"] == fwd["counter_account"])
            & (df["counter_account"] == fwd["base_account"])
            & (df["base_asset"] == fwd["base_asset"])
        ]

        for _, ret in candidates.iterrows():
            fwd_amt = float(fwd["amount"])
            ret_amt = float(ret["amount"])
            if fwd_amt == 0:
                continue
            if abs(fwd_amt - ret_amt) / fwd_amt <= amount_tolerance:
                elapsed = (ret["ledger_close_time"] - fwd["ledger_close_time"]).total_seconds()
                records.append(
                    {
                        "wallet_a": fwd["base_account"],
                        "wallet_b": fwd["counter_account"],
                        "forward_trade_id": fwd["trade_id"],
                        "return_trade_id": ret["trade_id"],
                        "forward_time": fwd["ledger_close_time"],
                        "return_time": ret["ledger_close_time"],
                        "forward_amount": fwd_amt,
                        "return_amount": ret_amt,
                        "asset": fwd["base_asset"],
                        "elapsed_seconds": elapsed,
                    }
                )

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input trades Parquet file")
    p.add_argument("--output", default="data/roundtrip_pairs.parquet")
    p.add_argument("--max-ledger-window", type=int, default=100)
    p.add_argument("--amount-tolerance", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trades = pd.read_parquet(args.input)
    pairs = detect_roundtrip_pairs(
        trades,
        max_ledger_window=args.max_ledger_window,
        amount_tolerance=args.amount_tolerance,
    )
    pairs.to_parquet(args.output, index=False)
    print(f"Detected {len(pairs)} round-trip pairs → {args.output}")


if __name__ == "__main__":
    main()
