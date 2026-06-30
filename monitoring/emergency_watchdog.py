"""Emergency watchdog: proposes an automatic pause when score distribution is anomalous.

Monitors the stream of risk scores produced by the local scoring pipeline.
If more than ``ANOMALY_RATE_THRESHOLD`` (default 90%) of scores in a
rolling one-minute window exceed ``ANOMALY_SCORE_THRESHOLD`` (default 95),
the watchdog proposes an emergency pause to the two human emergency keyholders
by calling ``LedgerLensContractClient.initiate_emergency_pause``.

The watchdog does NOT apply the pause itself — it only submits the *proposal*.
Two human keyholders must independently call ``approve_emergency_pause`` for
the pause to take effect, preventing the automated system from being used as a
DoS vector by a compromised pipeline.

Usage::

    watchdog = EmergencyWatchdog(
        pause_contract_id="C...",
        signing_key="S...",   # one emergency keyholder secret
    )
    watchdog.record_score(wallet_hash, score)   # call from scoring loop
    watchdog.check()                             # call periodically (e.g. every 5 s)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

from utils.logging import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS = 60
ANOMALY_SCORE_THRESHOLD = 95
ANOMALY_RATE_THRESHOLD = 0.90


class EmergencyWatchdog:
    """Watches rolling score distribution and proposes a pause on anomaly.

    Parameters
    ----------
    pause_contract_id:
        Soroban contract ID of the EmergencyPauseContract.
    signing_key:
        One of the 3 emergency keyholder Stellar secret keys.  The key is
        used only to sign the ``initiate_pause`` transaction; it is never
        stored persistently.
    rpc_url:
        Optional Soroban RPC override; falls back to ``config.SOROBAN_RPC_URL``.
    on_pause_proposed:
        Optional callback invoked with ``(proposal_id, reason)`` after a
        pause is successfully proposed on-chain (useful for alerting).
    window_seconds:
        Rolling window size in seconds (default 60).
    anomaly_score_threshold:
        A score is considered anomalous when it exceeds this value.
    anomaly_rate_threshold:
        Pause is proposed when the anomalous fraction exceeds this rate.
    """

    def __init__(
        self,
        pause_contract_id: str,
        signing_key: str,
        rpc_url: str | None = None,
        on_pause_proposed: Callable[[int, str], None] | None = None,
        window_seconds: int = _WINDOW_SECONDS,
        anomaly_score_threshold: int = ANOMALY_SCORE_THRESHOLD,
        anomaly_rate_threshold: float = ANOMALY_RATE_THRESHOLD,
    ) -> None:
        self.pause_contract_id = pause_contract_id
        self._signing_key = signing_key
        self._rpc_url = rpc_url
        self._on_pause_proposed = on_pause_proposed
        self._window_seconds = window_seconds
        self._anomaly_score_threshold = anomaly_score_threshold
        self._anomaly_rate_threshold = anomaly_rate_threshold
        # Deque of (timestamp, score) tuples
        self._window: deque[tuple[float, int]] = deque()
        self._pause_proposed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_score(self, wallet_id_hash: str, score: int) -> None:
        """Record a new score observation from the pipeline."""
        self._window.append((time.monotonic(), score))

    def check(self) -> bool:
        """Evaluate the rolling window and propose a pause if anomalous.

        Returns True if a pause was proposed during this call.
        """
        if self._pause_proposed:
            return False

        self._evict_old()
        if len(self._window) < 10:
            # Not enough data yet
            return False

        anomalous = sum(1 for _, s in self._window if s > self._anomaly_score_threshold)
        rate = anomalous / len(self._window)

        if rate > self._anomaly_rate_threshold:
            reason = (
                f"Anomalous score distribution: {rate:.0%} of scores in the last "
                f"{self._window_seconds}s exceed {self._anomaly_score_threshold} "
                f"(threshold: {self._anomaly_rate_threshold:.0%})"
            )
            logger.warning("EmergencyWatchdog: %s — proposing pause", reason)
            self._propose_pause(reason)
            return True
        return False

    @property
    def anomaly_rate(self) -> float:
        """Current fraction of scores in the window exceeding the threshold."""
        self._evict_old()
        if not self._window:
            return 0.0
        return sum(1 for _, s in self._window if s > self._anomaly_score_threshold) / len(
            self._window
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_old(self) -> None:
        cutoff = time.monotonic() - self._window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _propose_pause(self, reason: str) -> None:
        try:
            from integrations.contract_client import LedgerLensContractClient

            client = LedgerLensContractClient(
                contract_id="",   # not used for pause calls
                rpc_url=self._rpc_url,
            )
            proposal_id = client.initiate_emergency_pause(
                pause_contract_id=self.pause_contract_id,
                reason=reason,
                signing_key=self._signing_key,
            )
            self._pause_proposed = True
            logger.warning(
                "EmergencyWatchdog: pause proposed on-chain (proposal_id=%d)", proposal_id
            )
            if self._on_pause_proposed is not None:
                self._on_pause_proposed(proposal_id, reason)
        except Exception:
            logger.exception("EmergencyWatchdog: failed to propose emergency pause")
