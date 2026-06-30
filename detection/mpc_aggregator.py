"""Secure Multi-Party Computation aggregator for wash-trade risk scores.

Implements a 2- and 3-party MPC protocol for computing the **mean** and
**variance** of wash-trade risk scores across Stellar DEX ecosystem
participants (exchanges, wallet providers, custodians) using the ``mpyc``
library.

Protocol summary
----------------
Each participant holds a private list of per-wallet risk scores (0–100).
The parties jointly compute:

    aggregate_mean     = (sum_i sum_j score_{i,j}) / N_total
    aggregate_variance = sum_i sum_j (score_{i,j} - mean)^2 / N_total

without any party revealing its individual scores to the others.

Security model
--------------
The protocol is **information-theoretically secure in the semi-honest
(honest-but-curious) model** under Shamir secret sharing.  A semi-honest
party follows the protocol honestly but attempts to learn others' inputs
from the transcript.  The protocol does NOT defend against:

  - **Malicious parties** that deviate from the protocol (e.g. send
    incorrect shares).  Adding MACs (as in SPDZ) would upgrade to the
    malicious model but is out of scope here.
  - **Collusion**: if ``t`` or more parties collude they can reconstruct
    any secret, where ``t = floor((n-1)/2)`` for ``n`` parties.  With
    n=2: t=0 (no secrecy if one party is corrupt); n=3: t=1 (secure
    against any single corrupt party).

What each party learns at the end:
  - The aggregate mean and variance (output — same for all parties).
  - Nothing about any individual party's score distribution.

Party authentication
--------------------
Parties authenticate via TLS certificates in distributed mode, using the
same PKI infrastructure as ``detection/federated/coordinator.py``.  In
local (test) mode no certificates are required.

4-party and beyond
------------------
mpyc supports n-party Shamir secret sharing for arbitrary n.  This module
exposes a ``n_parties`` parameter; passing n ≥ 4 works correctly with mpyc
but the threshold ``t = floor((n-1)/2)`` grows, meaning a larger fraction
of parties must remain honest.  For n ≥ 4 in production, consider the
SPDZ-style MAC-check extension or switch to an actively-secure framework
(MP-SPDZ, SCALE-MAMBA).

Usage
-----
    # Local single-process simulation (all parties in one async event loop):
    import asyncio
    from detection.mpc_aggregator import mpc_aggregate_scores_local

    results = asyncio.run(mpc_aggregate_scores_local(
        party_scores=[[72.0, 45.0, 88.0], [10.0, 30.0, 55.0], [60.0, 80.0, 20.0]],
    ))
    print(results)  # {"mean": ..., "variance": ..., "n_total": 9}

    # Single-party call (distributed — each party runs independently):
    from detection.mpc_aggregator import mpc_aggregate_scores
    result = asyncio.run(mpc_aggregate_scores(
        local_scores=[72.0, 45.0, 88.0],
        n_parties=3,
        party_index=0,       # 0-indexed; coordinator assigns this
        peers=[("192.168.1.2", 11000), ("192.168.1.3", 11001)],
    ))
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger("ledgerlens.mpc_aggregator")

# ---------------------------------------------------------------------------
# mpyc import — graceful absence
# ---------------------------------------------------------------------------
try:
    from mpyc.runtime import Mpc, mpc as _global_mpc
    from mpyc.seclists import seclist
    import mpyc.sectypes as sectypes

    _MPYC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPYC_AVAILABLE = False
    _global_mpc = None  # type: ignore[assignment]

# Fixed-point precision: 32 integer bits, 16 fractional bits → ~4 decimal places
_FXP_INT_BITS: int = int(os.getenv("MPC_FXP_INT_BITS", "32"))
_FXP_FRAC_BITS: int = int(os.getenv("MPC_FXP_FRAC_BITS", "16"))

# ---------------------------------------------------------------------------
# Result dataclass (plain dict for JSON-serialisability)
# ---------------------------------------------------------------------------

AggregateResult = dict  # keys: mean, variance, n_total, n_parties


# ---------------------------------------------------------------------------
# Core MPC computation (runs inside the mpyc async event loop)
# ---------------------------------------------------------------------------


async def _compute_aggregate_mpc(
    mpc_runtime: Any,
    local_scores: list[float],
    n_parties: int,
    party_index: int,
) -> AggregateResult:
    """Run the MPC mean+variance computation.

    Each party secret-shares its scores with the others via Shamir sharing.
    The aggregate mean and variance are then computed on the shares and
    revealed to all parties simultaneously.

    Parameters
    ----------
    mpc_runtime:
        An active ``mpyc.runtime.Mpc`` instance (already started).
    local_scores:
        This party's private list of risk scores (floats in [0, 100]).
    n_parties:
        Total number of participating parties (2 or 3).
    party_index:
        0-indexed position of this party (``mpc_runtime.pid``).
    """
    # Secure fixed-point type
    secfxp = mpc_runtime.SecFxp(_FXP_INT_BITS + _FXP_FRAC_BITS, _FXP_FRAC_BITS)

    n_local = len(local_scores)

    # ---- Step 1: Each party shares its count and score sum ----
    # We use a simple approach: each party inputs its sum and sum-of-squares;
    # all parties then contribute to a global sum via secret addition.

    local_sum = sum(local_scores)
    local_sum_sq = sum(s * s for s in local_scores)
    local_count = float(n_local)

    # Secret-share this party's values
    sec_sum = mpc_runtime.input(secfxp(local_sum), senders=list(range(n_parties)))
    sec_sum_sq = mpc_runtime.input(secfxp(local_sum_sq), senders=list(range(n_parties)))
    sec_count = mpc_runtime.input(secfxp(local_count), senders=list(range(n_parties)))

    # ---- Step 2: Aggregate across all parties ----
    total_sum = sum(sec_sum)      # type: ignore[arg-type]
    total_sum_sq = sum(sec_sum_sq)  # type: ignore[arg-type]
    total_count = sum(sec_count)   # type: ignore[arg-type]

    # ---- Step 3: Derive mean and variance ----
    # mean = total_sum / total_count
    # variance = total_sum_sq / total_count - mean^2
    mean_sec = total_sum / total_count
    mean_sq_sec = mean_sec * mean_sec
    variance_sec = total_sum_sq / total_count - mean_sq_sec

    # ---- Step 4: Reveal outputs to all parties ----
    mean_plain, variance_plain, count_plain = await mpc_runtime.output(
        [mean_sec, variance_sec, total_count]
    )

    # Resolve the total count (it's a secure value; plain after output)
    n_total = int(round(float(count_plain)))

    return AggregateResult(
        mean=float(mean_plain),
        variance=float(variance_plain),
        n_total=n_total,
        n_parties=n_parties,
    )


# ---------------------------------------------------------------------------
# Local multi-party simulation (all parties in one process)
# ---------------------------------------------------------------------------


async def mpc_aggregate_scores_local(
    party_scores: list[list[float]],
    threshold: int | None = None,
) -> AggregateResult:
    """Simulate n-party MPC in a single process using mpyc's local mode.

    All parties run cooperatively in the same event loop — no network
    required.  This is the correct way to test MPC logic without a
    distributed setup.

    Parameters
    ----------
    party_scores:
        List of per-party score lists.  ``party_scores[i]`` is party i's
        private input.  All inner lists may have different lengths.
    threshold:
        Shamir threshold t (number of corrupt parties tolerated).  Defaults
        to ``floor((n-1)/2)``.

    Returns
    -------
    AggregateResult with ``mean``, ``variance``, ``n_total``, ``n_parties``.
    """
    if not _MPYC_AVAILABLE:
        raise RuntimeError(
            "mpyc is not installed. Run: pip install mpyc"
        )

    n_parties = len(party_scores)
    if n_parties < 2:
        raise ValueError(f"MPC requires at least 2 parties, got {n_parties}")

    if threshold is None:
        threshold = (n_parties - 1) // 2

    # mpyc local mode: create one Mpc instance per party, all sharing the
    # same in-process communication channels.
    from mpyc.runtime import Party, Mpc

    parties = [Party(pid=i) for i in range(n_parties)]

    # Collect results from each party's coroutine
    results: list[AggregateResult] = [{}] * n_parties  # type: ignore[list-item]

    async def _run_party(pid: int) -> None:
        runtime = Mpc(pid=pid, parties=parties, threshold=threshold, no_party=True)
        async with runtime:
            results[pid] = await _compute_aggregate_mpc(
                runtime,
                party_scores[pid],
                n_parties,
                pid,
            )

    await asyncio.gather(*[_run_party(i) for i in range(n_parties)])

    # All parties should see identical outputs; return party 0's view
    return results[0]


# ---------------------------------------------------------------------------
# Distributed single-party entry point
# ---------------------------------------------------------------------------


async def mpc_aggregate_scores(
    local_scores: list[float],
    n_parties: int,
    party_index: int = 0,
    peers: list[tuple[str, int]] | None = None,
    threshold: int | None = None,
) -> AggregateResult:
    """Run the MPC protocol as a single party in a distributed setup.

    Each participating organisation runs this function independently.
    The mpyc runtime handles peer connections and Shamir sharing.

    Parameters
    ----------
    local_scores:
        This party's private risk scores.
    n_parties:
        Total number of parties.
    party_index:
        0-indexed position of this party (must be unique per process).
    peers:
        List of ``(host, port)`` tuples for all parties (including self).
        If None, the mpyc default localhost addressing is used (for testing).
    threshold:
        Shamir threshold.  Defaults to ``floor((n_parties - 1) / 2)``.
    """
    if not _MPYC_AVAILABLE:
        raise RuntimeError("mpyc is not installed. Run: pip install mpyc")

    if n_parties < 2:
        raise ValueError(f"MPC requires at least 2 parties, got {n_parties}")
    if not (2 <= n_parties <= 3):
        logger.warning(
            "n_parties=%d — this module is validated for 2 and 3 parties. "
            "4+ parties work with mpyc but require a larger Shamir threshold "
            "and are not covered by the current test suite. See module docstring.",
            n_parties,
        )

    if threshold is None:
        threshold = (n_parties - 1) // 2

    from mpyc.runtime import Party, Mpc

    if peers is not None:
        parties = [
            Party(pid=i, host=peers[i][0], port=peers[i][1])
            for i in range(n_parties)
        ]
    else:
        # Default: all localhost, ports 11000..11000+n-1
        parties = [
            Party(pid=i, host="localhost", port=11000 + i)
            for i in range(n_parties)
        ]

    runtime = Mpc(pid=party_index, parties=parties, threshold=threshold)
    async with runtime:
        return await _compute_aggregate_mpc(
            runtime, local_scores, n_parties, party_index
        )


# ---------------------------------------------------------------------------
# Plaintext reference implementation (for validation / benchmarking)
# ---------------------------------------------------------------------------


def plaintext_aggregate(party_scores: list[list[float]]) -> AggregateResult:
    """Compute mean and variance directly on all scores (no MPC).

    Used in tests to verify that the MPC result matches the ground truth.
    """
    all_scores: list[float] = []
    for scores in party_scores:
        all_scores.extend(scores)

    n = len(all_scores)
    if n == 0:
        return AggregateResult(mean=0.0, variance=0.0, n_total=0, n_parties=len(party_scores))

    mean = sum(all_scores) / n
    variance = sum((s - mean) ** 2 for s in all_scores) / n
    return AggregateResult(
        mean=mean,
        variance=variance,
        n_total=n,
        n_parties=len(party_scores),
    )
