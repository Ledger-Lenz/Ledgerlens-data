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

Self-test / canary
------------------
A silently-dead watchdog is worse than no watchdog because it creates false
confidence.  To detect that, the watchdog emits a periodic *heartbeat* to an
external, independently-monitored dead-man's-switch endpoint (outside the
primary monitoring stack, so a failure of that stack cannot mask a dead
watchdog).  If the heartbeat is missed, the external service fires an alert.

Configure via the ``heartbeat_url`` constructor argument (or the
``WATCHDOG_HEARTBEAT_URL`` environment variable).  The heartbeat is emitted
from :meth:`check` on a fixed interval (``heartbeat_interval_seconds``,
default 30 s) and is best-effort: a failed heartbeat is logged but never
raises, so it cannot take down the watchdog itself.

Escalation runbook for a missed watchdog heartbeat
--------------------------------------------------
1. External dead-man's-switch fires (heartbeat missed for > 2 intervals).
2. On-call acknowledges within 5 minutes and checks whether the watchdog
   process is running (``systemctl status ledgerlens-emergency-watchdog``).
3. If the process is down, restart it and inspect logs for the crash cause.
4. If the process is up but not heartbeating, treat the watchdog as blind:
   manually review the score distribution and, if anomalous, initiate the
   emergency pause by hand.
5. Escalate to the security lead if the watchdog cannot be restored within
   15 minutes; the pause proposal path must not be left unmonitored.

Usage::

    watchdog = EmergencyWatchdog(
        pause_contract_id="C...",
        signing_key="S...",   # one emergency keyholder secret
        heartbeat_url="https://deadman.example.com/heartbeat/ledgerlens-watchdog",
    )
    watchdog.record_score(wallet_hash, score)   # call from scoring loop
    watchdog.check()                             # call periodically (e.g. every 5 s)
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from collections.abc import Callable

from utils.logging import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS = 60
ANOMALY_SCORE_THRESHOLD = 95
ANOMALY_RATE_THRESHOLD = 0.90

# Default interval between external heartbeat emissions, in seconds.
HEARTBEAT_INTERVAL_SECONDS = 30
# Environment variable used to configure the external dead-man's-switch URL.
HEARTBEAT_URL_ENV = "WATCHDOG_HEARTBEAT_URL"


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
    latency_budget_ms:
        End-to-end latency threshold in milliseconds. If the rate of events exceeding this budget surpasses `anomaly_rate_threshold`, a pause is proposed.
    heartbeat_url:
        External, independently-monitored dead-man's-switch endpoint.  When
        set (or when ``WATCHDOG_HEARTBEAT_URL`` is present in the
        environment), the watchdog emits a periodic heartbeat so a missed
        heartbeat can be alerted on externally.
    heartbeat_interval_seconds:
        Fixed interval between heartbeat emissions (default 30 s).
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
        latency_budget_ms: int = 2000,
        heartbeat_url: str | None = None,
        heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.pause_contract_id = pause_contract_id
        self._signing_key = signing_key
        self._rpc_url = rpc_url
        self._on_pause_proposed = on_pause_proposed
        self._window_seconds = window_seconds
        self._anomaly_score_threshold = anomaly_score_threshold
        self._anomaly_rate_threshold = anomaly_rate_threshold
        self._latency_budget_ms = latency_budget_ms
        # Deque of (timestamp, score, e2e_latency_ms) tuples
        self._window: deque[tuple[float, int, float | None]] = deque()
        self._pause_proposed = False
        # Self-test / canary state
        self._heartbeat_url = heartbeat_url or os.environ.get(HEARTBEAT_URL_ENV) or None
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._last_heartbeat = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_score(
        self, wallet_id_hash: str, score: int, e2e_latency_ms: float | None = None
    ) -> None:
        """Record a new score observation from the pipeline."""
        self._window.append((time.monotonic(), score, e2e_latency_ms))

    def check(self) -> bool:
        """Evaluate the rolling window and propose a pause if anomalous.

        Also emits the external heartbeat on its fixed interval so a dead
        watchdog is detected by the dead-man's-switch.

        Returns True if a pause was proposed during this call.
        """
        self._maybe_heartbeat()

        if self._pause_proposed:
            return False

        self._evict_old()
        if len(self._window) < 10:
            # Not enough data yet
            return False

        anomalous = sum(1 for _, s, _ in self._window if s > self._anomaly_score_threshold)
        rate = anomalous / len(self._window)

        latency_records = sum(1 for _, _, lat in self._window if lat is not None)
        if latency_records > 0:
            anomalous_latency = sum(
                1 for _, _, lat in self._window if lat is not None and lat > self._latency_budget_ms
            )
            latency_rate = anomalous_latency / latency_records
        else:
            latency_rate = 0.0

        if rate > self._anomaly_rate_threshold:
            reason = (
                f"Anomalous score distribution: {rate:.0%} of scores in the last "
                f"{self._window_seconds}s exceed {self._anomaly_score_threshold} "
                f"(threshold: {self._anomaly_rate_threshold:.0%})"
            )
            logger.warning("EmergencyWatchdog: %s — proposing pause", reason)
            self._propose_pause(reason)
            return True

        if latency_rate > self._anomaly_rate_threshold:
            reason = (
                f"Latency budget breached: {latency_rate:.0%} of events in the last "
                f"{self._window_seconds}s exceed {self._latency_budget_ms}ms "
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
        return sum(1 for _, s, _ in self._window if s > self._anomaly_score_threshold) / len(
            self._window
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_old(self) -> None:
        cutoff = time.monotonic() - self._window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _maybe_heartbeat(self) -> None:
        """Emit the external heartbeat if the fixed interval has elapsed.

        Best-effort: failures are logged and swallowed so the canary can
        never crash the watchdog it is meant to protect.
        """
        if not self._heartbeat_url:
            return
        now = time.monotonic()
        if now - self._last_heartbeat < self._heartbeat_interval_seconds:
            return
        self._last_heartbeat = now
        try:
            req = urllib.request.Request(
                self._heartbeat_url,
                data=b"watchdog-alive",
                method="POST",
                headers={"Content-Type": "text/plain"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception:
            logger.exception("EmergencyWatchdog: failed to emit external heartbeat")

    def _propose_pause(self, reason: str) -> None:
        try:
            from integrations.contract_client import LedgerLensContractClient

            client = LedgerLensContractClient(
                contract_id="",  # not used for pause calls
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
