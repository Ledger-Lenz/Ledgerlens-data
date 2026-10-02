"""Behavioral fingerprint matcher.

Links wallet addresses across chains by analyzing trade amounts and timing.

``match_jitter_robust`` (#882) is the adversarially robust variant: it scores
source/destination legs of a cross-chain transfer by a fee-aware amount
fingerprint and a heavy-tailed delay prior instead of a hard time window, so
an adversary adding random delay between the legs no longer defeats matching.
See docs/cross_chain_jitter_matching.md for assumptions and limitations.
"""

from __future__ import annotations

import bisect
import logging
import math
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
                s_records.append(
                    {
                        "wallet": tx["wallet"],
                        "timestamp": to_timestamp(tx["timestamp"]),
                        "amount": float(tx["amount"]),
                        "id": tx.get("id", tx.get("tx_id", "")),
                    }
                )
            except Exception as e:
                logger.warning("Skipping invalid Stellar record: %s. Error: %s", tx, e)

        ext_records = []
        for tx in external_txs:
            try:
                ext_records.append(
                    {
                        "wallet": tx["wallet"],
                        "timestamp": to_timestamp(tx["timestamp"]),
                        "amount": float(tx["amount"]),
                        "chain": tx.get("chain", "ethereum").lower(),
                        "id": tx.get("id", tx.get("tx_id", "")),
                    }
                )
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
                    links.append(
                        {
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
                                "type": "amount_fingerprint",
                            },
                        }
                    )

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
                    links.append(
                        {
                            "stellar_address": s_w,
                            "linked_address": ext_w,
                            "chain": chain,
                            "confidence": float(r),
                            "metadata": {
                                "pearson_r": float(r),
                                "n_bins": int(n_bins),
                                "bin_size_seconds": float(bin_size_seconds),
                                "type": "timing_correlation",
                            },
                        }
                    )

        return links

    @staticmethod
    def match_jitter_robust(
        stellar_txs: list[dict[str, Any]],
        external_txs: list[dict[str, Any]],
        fee_models: list[tuple[float, float]] | None = None,
        amount_tolerance: float = 0.0005,
        max_delay_seconds: float = 6 * 3600.0,
        max_clock_skew_seconds: float = 120.0,
        typical_delay_seconds: float = 120.0,
        adversarial_mix: float = 0.5,
        min_confidence: float = 0.6,
        density_band: float = 20.0,
        match_prior: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Match cross-chain transfer legs robustly to deliberate timing jitter.

        Treat each Stellar leg ``s`` as the source and each external leg ``e``
        arriving within ``[-max_clock_skew_seconds, max_delay_seconds]`` as a
        candidate destination. The likelihood that ``e`` is ``s``'s other leg is

            L(s, e) = A(rel) * D(dt)

        * ``A`` — amount fingerprint: ``rel = |e.amount - expected| / s.amount``
          where ``expected = s.amount * (1 - rate) - fixed`` for the best
          fitting ``(fixed, rate)`` in ``fee_models`` (default: no fee). ``A``
          is a Gaussian in ``rel`` with sigma ``amount_tolerance / 2``,
          truncated at ``amount_tolerance``.
        * ``D`` — delay prior: a mixture of an exponential with mean
          ``typical_delay_seconds`` (honest bridge latency) and a uniform over
          the whole window (an adversary may pick any delay).
          ``adversarial_mix`` is the uniform's weight.

        Candidates are normalised against each other *and* against a null
        hypothesis (the true destination is not among them, i.e. every
        candidate is background traffic), giving a posterior per pair. The
        null weight is distribution-aware: it grows with the number of chance
        matches expected in the window for this amount, estimated from how
        often the amount occurs across the whole external set. Popular round
        amounts therefore need tighter timing to be linked than distinctive
        ones. ``match_prior`` is the prior probability that a source leg's
        destination is present in ``external_txs`` at all; the null weight is
        scaled by the prior odds ``(1 - match_prior) / match_prior``. Set it
        from the observed base rate (e.g. the share of bridge deposits whose
        destination chain is ingested); a value that is too high trades
        precision for recall in dense traffic. Pairs
        are then assigned one-to-one greedily by posterior, and kept when the
        posterior is ``>= min_confidence``.

        Input records use the same shape as :meth:`match_amount_fingerprints`.
        """
        fee_models = fee_models or [(0.0, 0.0)]
        sigma = amount_tolerance / 2.0
        window = max_delay_seconds + max_clock_skew_seconds
        uniform_density = 1.0 / window
        # Null likelihood: a borderline (2-sigma) amount match at a delay the
        # honest-latency component considers implausible.
        null_likelihood = math.exp(-2.0) * adversarial_mix * uniform_density

        if not 0.0 < match_prior < 1.0:
            raise ValueError("match_prior must be in (0, 1)")
        prior_odds = (1.0 - match_prior) / match_prior
        skew_scale = max(max_clock_skew_seconds / 10.0, 1e-9)

        def delay_density(dt: float) -> float:
            # The destination leg cannot precede the source except by clock
            # skew, so negative delays decay quickly instead of scoring as 0 s.
            decay = dt / typical_delay_seconds if dt >= 0 else -dt / skew_scale
            honest = math.exp(-decay) / typical_delay_seconds
            return (1 - adversarial_mix) * honest + adversarial_mix * uniform_density

        s_records = []
        for tx in stellar_txs:
            try:
                s_records.append(
                    (
                        tx["wallet"],
                        to_timestamp(tx["timestamp"]),
                        float(tx["amount"]),
                        tx.get("id", tx.get("tx_id", "")),
                    )
                )
            except Exception as e:
                logger.warning("Skipping invalid Stellar record: %s. Error: %s", tx, e)
        ext_records = []
        for tx in external_txs:
            try:
                ext_records.append(
                    (
                        to_timestamp(tx["timestamp"]),
                        float(tx["amount"]),
                        tx["wallet"],
                        tx.get("chain", "ethereum").lower(),
                        tx.get("id", tx.get("tx_id", "")),
                    )
                )
            except Exception as e:
                logger.warning("Skipping invalid external record: %s. Error: %s", tx, e)
        ext_records.sort(key=lambda r: r[0])
        ext_times = [r[0] for r in ext_records]
        ext_amounts = sorted(r[1] for r in ext_records)
        span = max(ext_times[-1] - ext_times[0], window) if ext_times else window

        def count_near(center: float, half_width: float) -> int:
            return bisect.bisect_right(ext_amounts, center + half_width) - bisect.bisect_left(
                ext_amounts, center - half_width
            )

        def chance_matches(expected: float, s_amt: float) -> float:
            """Expected background amount collisions inside one window.

            Takes the larger of the exact-band count (minus the true leg; this
            catches spikes at popular round amounts) and the smoothed local
            density from a band ``density_band`` times wider (which catches
            dense traffic where a lone collision is still likely chance).
            """
            tol = amount_tolerance * s_amt
            exact = max(count_near(expected, tol) - 1, 0)
            smoothed = count_near(expected, tol * density_band) / density_band
            return max(exact, smoothed) * window / span

        candidates: list[tuple[float, int, int, float, float]] = []
        for si, (_, s_time, s_amt, _) in enumerate(s_records):
            if s_amt <= 0:
                continue
            lo = bisect.bisect_left(ext_times, s_time - max_clock_skew_seconds)
            hi = bisect.bisect_right(ext_times, s_time + max_delay_seconds)
            scored = []
            chance = min(
                chance_matches(s_amt * (1 - rate) - fixed, s_amt) for fixed, rate in fee_models
            )
            for ei in range(lo, hi):
                e_time, e_amt = ext_records[ei][0], ext_records[ei][1]
                rel = min(
                    abs(e_amt - (s_amt * (1 - rate) - fixed)) / s_amt for fixed, rate in fee_models
                )
                if rel > amount_tolerance:
                    continue
                dt = e_time - s_time
                likelihood = math.exp(-0.5 * (rel / sigma) ** 2) * delay_density(dt)
                scored.append((likelihood, ei, rel, dt))
            null = (null_likelihood + chance * uniform_density) * prior_odds
            total = sum(lk for lk, *_ in scored) + null
            for likelihood, ei, rel, dt in scored:
                candidates.append((likelihood / total, si, ei, rel, dt))

        links = []
        used_s: set[int] = set()
        used_e: set[int] = set()
        for posterior, si, ei, rel, dt in sorted(candidates, key=lambda c: -c[0]):
            if posterior < min_confidence:
                break
            if si in used_s or ei in used_e:
                continue
            used_s.add(si)
            used_e.add(ei)
            s_wallet, s_time, s_amt, s_id = s_records[si]
            e_time, e_amt, e_wallet, chain, e_id = ext_records[ei]
            links.append(
                {
                    "stellar_address": s_wallet,
                    "linked_address": e_wallet,
                    "chain": chain,
                    "confidence": float(posterior),
                    "metadata": {
                        "stellar_tx_id": s_id,
                        "external_tx_id": e_id,
                        "stellar_amount": s_amt,
                        "external_amount": e_amt,
                        "stellar_timestamp": s_time,
                        "external_timestamp": e_time,
                        "delay_seconds": dt,
                        "amount_residual": rel,
                        "type": "jitter_robust_fingerprint",
                    },
                }
            )
        return links
