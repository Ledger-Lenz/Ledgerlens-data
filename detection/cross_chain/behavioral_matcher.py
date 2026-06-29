"""Behavioral fingerprint matcher.

Links wallet addresses across chains by analyzing trade amounts and timing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def to_timestamp(ts: Any) -> float:
    """Convert datetime, float, int, or ISO-string to POSIX timestamp."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    if isinstance(ts, str):
        # Handle trailing Z or offsets
        s = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            # Fall back to trying to parse common formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt).timestamp()
                except ValueError:
                    continue
            raise
    raise ValueError(f"Unsupported timestamp type: {type(ts)}")


class BehavioralMatcher:
    """Matches cross-chain wallets using behavioral patterns."""

    @staticmethod
    def match_amount_fingerprints(
        stellar_txs: list[dict[str, Any]],
        external_txs: list[dict[str, Any]],
        tolerance: float = 0.001,  # 0.1%
        window_seconds: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Match Stellar and EVM/Solana wallets based on identical trade amounts.

        stellar_txs and external_txs should have:
        {
            "wallet": "...",
            "timestamp": ... (datetime, float, or string),
            "amount": ... (float),
            "chain": "..." (optional for stellar, required for external to identify type)
        }
        """
        links = []

        # Convert timestamps
        s_records = []
        for tx in stellar_txs:
            try:
                s_records.append({
                    "wallet": tx["wallet"],
                    "timestamp": to_timestamp(tx["timestamp"]),
                    "amount": float(tx["amount"]),
                    "id": tx.get("id", tx.get("tx_id", ""))
                })
            except Exception as e:
                logger.warning("Skipping invalid Stellar record: %s. Error: %s", tx, e)

        ext_records = []
        for tx in external_txs:
            try:
                ext_records.append({
                    "wallet": tx["wallet"],
                    "timestamp": to_timestamp(tx["timestamp"]),
                    "amount": float(tx["amount"]),
                    "chain": tx.get("chain", "ethereum").lower(),
                    "id": tx.get("id", tx.get("tx_id", ""))
                })
            except Exception as e:
                logger.warning("Skipping invalid external record: %s. Error: %s", tx, e)

        # Match pairs
        for s_tx in s_records:
            s_amt = s_tx["amount"]
            s_time = s_tx["timestamp"]
            if s_amt <= 0:
                continue

            for ext_tx in ext_records:
                ext_amt = ext_tx["amount"]
                ext_time = ext_tx["timestamp"]

                # Check time window
                if abs(s_time - ext_time) > window_seconds:
                    continue

                # Check amount tolerance: abs(s_amt - ext_amt) / s_amt <= tolerance
                diff = abs(s_amt - ext_amt)
                if (diff / s_amt) <= tolerance:
                    # Match found! Calculate confidence
                    confidence = 1.0 - (diff / s_amt) if diff > 0 else 1.0
                    links.append({
                        "stellar_address": s_tx["wallet"],
                        "linked_address": ext_tx["wallet"],
                        "chain": ext_tx["chain"],
                        "confidence": float(confidence),
                        "metadata": {
                            "stellar_tx_id": s_tx["id"],
                            "external_tx_id": ext_tx["id"],
                            "stellar_amount": s_amt,
                            "external_amount": ext_amt,
                            "stellar_timestamp": s_time,
                            "external_timestamp": ext_time,
                            "type": "amount_fingerprint"
                        }
                    })

        return links

    @staticmethod
    def match_timing_correlation(
        stellar_txs: list[dict[str, Any]],
        external_txs: list[dict[str, Any]],
        bin_size_seconds: float = 3600.0,  # 1 hour
        min_common_bins: int = 5,
        threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Calculate Pearson correlation of binned transaction counts.

        Links wallets if correlation >= threshold.
        """
        links = []

        # 1. Parse and extract timestamps per wallet
        stellar_wallets: dict[str, list[float]] = {}
        for tx in stellar_txs:
            try:
                w = tx["wallet"]
                ts = to_timestamp(tx["timestamp"])
                stellar_wallets.setdefault(w, []).append(ts)
            except Exception:
                continue

        external_wallets: dict[str, tuple[str, list[float]]] = {}
        for tx in external_txs:
            try:
                w = tx["wallet"]
                ts = to_timestamp(tx["timestamp"])
                chain = tx.get("chain", "ethereum").lower()
                if w not in external_wallets:
                    external_wallets[w] = (chain, [])
                external_wallets[w][1].append(ts)
            except Exception:
                continue

        if not stellar_wallets or not external_wallets:
            return []

        # Find global min/max timestamps to define bin grid
        all_timestamps = []
        for times in stellar_wallets.values():
            all_timestamps.extend(times)
        for _, times in external_wallets.values():
            all_timestamps.extend(times)

        global_min = min(all_timestamps)
        global_max = max(all_timestamps)

        # If all transactions happen at the exact same instant, we can't correlate
        if global_max == global_min:
            return []

        # Create bin edges
        bins = np.arange(global_min, global_max + bin_size_seconds, bin_size_seconds)
        n_bins = len(bins) - 1

        if n_bins < min_common_bins:
            # Not enough bins to run correlation
            return []

        # 2. Build activity count histograms for each wallet
        stellar_histograms = {}
        for w, times in stellar_wallets.items():
            hist, _ = np.histogram(times, bins=bins)
            stellar_histograms[w] = hist

        external_histograms = {}
        for w, (chain, times) in external_wallets.items():
            hist, _ = np.histogram(times, bins=bins)
            external_histograms[w] = (chain, hist)

        # 3. Compute Pearson correlation for each pair
        for s_w, s_hist in stellar_histograms.items():
            s_std = np.std(s_hist)
            if s_std == 0:
                continue  # No variance, correlation undefined

            for ext_w, (chain, ext_hist) in external_histograms.items():
                ext_std = np.std(ext_hist)
                if ext_std == 0:
                    continue  # No variance

                # Compute Pearson correlation
                r = np.corrcoef(s_hist, ext_hist)[0, 1]
                if np.isnan(r):
                    continue

                if r >= threshold:
                    links.append({
                        "stellar_address": s_w,
                        "linked_address": ext_w,
                        "chain": chain,
                        "confidence": float(r),
                        "metadata": {
                            "pearson_r": float(r),
                            "n_bins": int(n_bins),
                            "bin_size_seconds": float(bin_size_seconds),
                            "type": "timing_correlation"
                        }
                    })

        return links
