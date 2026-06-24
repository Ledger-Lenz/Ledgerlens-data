"""Build a ground-truth labelled Stellar SDEX wash-trade dataset.

Combines three independent signals to assign labels:
  Signal 1 — Round-trip detection   (scripts/mine_roundtrips.py)
  Signal 2 — Funding-graph clustering (detection/wallet_graph.py)
  Signal 3 — Manual review sample   (data/labelling_notes.md)

Conservative labelling rule:
  label = 1  ← flagged by BOTH signals 1 AND 2
  label = 0  ← NO flags from either signal AND wallet has > 50 trades
               with > 5 distinct counterparties
  label = NaN ← everything else (grey zone, excluded)

Output Parquet adds to the feature matrix columns:
  label, labelling_signal, review_notes,
  data_window_start, data_window_end, n_trades

Usage:
    python -m scripts.build_labelled_dataset \
        --trades data/raw_trades.parquet \
        --output data/labelled_dataset.parquet \
        --config data/build_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from detection.feature_engineering import build_feature_matrix
from detection.wallet_graph import build_funding_graph, funding_source_similarity
from ingestion.data_models import AccountActivity
from scripts.mine_roundtrips import detect_roundtrip_pairs

# Jaccard similarity threshold above which two wallets are in the same cluster
GRAPH_SIMILARITY_THRESHOLD = 0.7
# Minimum trades for a wallet to be labelled 0 (legitimate)
MIN_TRADES_FOR_NEGATIVE = 50
# Minimum distinct counterparties for a wallet to be labelled 0
MIN_COUNTERPARTIES_FOR_NEGATIVE = 5


def apply_labelling_rules(
    wallets: list[str],
    roundtrip_flagged: set[str],
    graph_flagged: set[str],
    wallet_trades: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply conservative labelling rules and return a label DataFrame.

    Returns columns: wallet, label (int or NaN), labelling_signal.
    """
    rows = []
    for wallet in wallets:
        rt_flag = wallet in roundtrip_flagged
        gr_flag = wallet in graph_flagged
        wt = wallet_trades.get(wallet, pd.DataFrame())
        n_trades = len(wt)
        n_counterparties = (
            pd.unique(wt[["base_account", "counter_account"]].values.ravel())
            if not wt.empty
            else []
        )
        # Exclude self from counterparty count
        n_cp = sum(1 for w in n_counterparties if w != wallet)

        if rt_flag and gr_flag:
            label = 1
            signal = "roundtrip_and_graph"
        elif rt_flag and not gr_flag:
            label = np.nan
            signal = "roundtrip_only"
        elif gr_flag and not rt_flag:
            label = np.nan
            signal = "graph_only"
        elif (
            not rt_flag
            and not gr_flag
            and n_trades > MIN_TRADES_FOR_NEGATIVE
            and n_cp > MIN_COUNTERPARTIES_FOR_NEGATIVE
        ):
            label = 0
            signal = "clean"
        else:
            label = np.nan
            signal = "insufficient_data"

        rows.append({"wallet": wallet, "label": label, "labelling_signal": signal})

    return pd.DataFrame(rows)


def build_labelled_dataset(
    trades_df: pd.DataFrame,
    activities: list[AccountActivity] | None = None,
    max_ledger_window: int = 100,
    amount_tolerance: float = 0.05,
    graph_similarity_threshold: float = GRAPH_SIMILARITY_THRESHOLD,
    data_window_start: str | None = None,
    data_window_end: str | None = None,
) -> pd.DataFrame:
    """Orchestrate all three signals into a labelled feature Parquet.

    Parameters
    ----------
    trades_df:
        Raw trades DataFrame (columns from historical_loader / data_models).
    activities:
        Optional list of AccountActivity objects for the funding graph.
    max_ledger_window:
        Passed to detect_roundtrip_pairs.
    amount_tolerance:
        Passed to detect_roundtrip_pairs.
    graph_similarity_threshold:
        Jaccard threshold for graph-cluster flagging.
    data_window_start / data_window_end:
        ISO datetime strings for the data window provenance columns.

    Returns
    -------
    Labelled feature DataFrame with extra provenance columns.
    """
    if trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["ledger_close_time"] = pd.to_datetime(trades_df["ledger_close_time"], utc=True)

    # ── Signal 1: round-trip detection ──────────────────────────────────────
    roundtrip_pairs = detect_roundtrip_pairs(
        trades_df,
        max_ledger_window=max_ledger_window,
        amount_tolerance=amount_tolerance,
    )
    roundtrip_flagged: set[str] = set()
    if not roundtrip_pairs.empty:
        roundtrip_flagged.update(roundtrip_pairs["wallet_a"].tolist())
        roundtrip_flagged.update(roundtrip_pairs["wallet_b"].tolist())

    # ── Signal 2: funding-graph clustering ──────────────────────────────────
    graph_flagged: set[str] = set()
    if activities:
        funding_graph: nx.DiGraph = build_funding_graph(activities)
        all_wallets = list(pd.unique(trades_df[["base_account", "counter_account"]].values.ravel()))
        for wallet in all_wallets:
            sim = funding_source_similarity(wallet, funding_graph)
            if sim >= graph_similarity_threshold:
                graph_flagged.add(wallet)

    # ── Build per-wallet trade lookup ────────────────────────────────────────
    all_wallets_list = list(
        pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())
    )
    wallet_trades: dict[str, pd.DataFrame] = {}
    for wallet in all_wallets_list:
        mask = (trades_df["base_account"] == wallet) | (trades_df["counter_account"] == wallet)
        wallet_trades[wallet] = trades_df[mask]

    # ── Apply labelling rules ────────────────────────────────────────────────
    labels_df = apply_labelling_rules(
        all_wallets_list,
        roundtrip_flagged,
        graph_flagged,
        wallet_trades,
    )

    # ── Build feature matrix ─────────────────────────────────────────────────
    features_df = build_feature_matrix(trades_df)

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = features_df.merge(labels_df, on="wallet", how="left")

    # ── Add provenance columns ───────────────────────────────────────────────
    window_start = data_window_start or str(trades_df["ledger_close_time"].min())
    window_end = data_window_end or str(trades_df["ledger_close_time"].max())

    n_trades_map = {w: len(wt) for w, wt in wallet_trades.items()}
    merged["data_window_start"] = window_start
    merged["data_window_end"] = window_end
    merged["n_trades"] = merged["wallet"].map(n_trades_map).fillna(0).astype(int)
    merged["review_notes"] = ""

    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trades", required=True, help="Raw trades Parquet file")
    p.add_argument("--output", default="data/labelled_dataset.parquet")
    p.add_argument("--config", default="data/build_config.json")
    p.add_argument("--max-ledger-window", type=int, default=100)
    p.add_argument("--amount-tolerance", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    cfg: dict = {}
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)

    trades = pd.read_parquet(args.trades)
    result = build_labelled_dataset(
        trades,
        max_ledger_window=args.max_ledger_window
        or cfg.get("thresholds", {}).get("max_ledger_window", 100),
        amount_tolerance=args.amount_tolerance
        or cfg.get("thresholds", {}).get("amount_tolerance", 0.05),
        data_window_start=cfg.get("date_range_start"),
        data_window_end=cfg.get("date_range_end"),
    )

    # Drop grey-zone rows (label = NaN) for the released dataset
    labelled = result.dropna(subset=["label"])
    labelled["label"] = labelled["label"].astype(int)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(args.output, index=False)

    n_pos = int((labelled["label"] == 1).sum())
    n_neg = int((labelled["label"] == 0).sum())
    print(
        f"Wrote {len(labelled)} labelled rows ({n_pos} positive, {n_neg} negative) → {args.output}"
    )


if __name__ == "__main__":
    main()
