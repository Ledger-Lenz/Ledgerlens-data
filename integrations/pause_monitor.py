"""Off-chain monitoring parity check for the LedgerLens emergency-pause system (#949).

Overview
--------
The on-chain emergency-pause contract (``integrations/emergency_pause_contract.rs``)
is the *source of truth* for whether the scoring oracle should be paused.  The
off-chain scoring pipeline must honour it.  However, there is a window — bounded
by ``poll_interval_seconds`` (default 30 s) — in which the on-chain state could
diverge from the pipeline's operational state without anyone noticing.

:class:`PauseMonitor` closes this gap by periodically reconciling both states
and raising a paging-severity alert whenever they disagree.

Latency bound
~~~~~~~~~~~~~
A mismatch is **guaranteed to be detected within** ``poll_interval_seconds``
(default 30 s) of it occurring, assuming the monitor is running.  This bound is
coarser than Stellar's ~5 s ledger finality by design: the pause contract emits
events, which are consumed by :class:`~integrations.soroban_event_listener.SorobanEventListener`
with its own polling loop.  ``PauseMonitor`` is a *belt-and-suspenders* check
that catches cases where the event listener missed an event (network blip, RPC
error) or the pipeline failed to act on it.

Alert severity
~~~~~~~~~~~~~~
The ``alert_fn`` is called with a ``mismatch_details: dict`` containing at
minimum:

.. code-block:: python

    {
        "severity": "CRITICAL",     # always CRITICAL for pause mismatches
        "onchain_paused": bool,
        "pipeline_paused": bool,
        "detected_at": "<ISO timestamp>",
        "message": "<human-readable description>",
    }

The caller is responsible for routing this dict to their alerting system
(PagerDuty, Opsgenie, Slack, etc.).  See
``docs/runbooks/pause_monitor_mismatch.md`` for the response procedure.

Usage
-----
::

    from integrations.pause_monitor import PauseMonitor

    monitor = PauseMonitor(
        contract_client=my_client,
        pipeline_state_fn=lambda: my_pipeline.is_running(),
        alert_fn=pagerduty_alert,
    )
    # Blocking loop — run in a background thread or asyncio task:
    import asyncio
    asyncio.run(monitor.start())

    # Or run a single reconciliation tick (useful for testing):
    monitor.run_once()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


class PauseMonitor:
    """Periodically reconciles the on-chain pause state with the pipeline's
    operational state and raises a paging-severity alert on mismatch.

    Parameters
    ----------
    contract_client:
        Any object with an ``is_paused() -> bool`` method that queries the
        emergency-pause contract.  In production this is
        :class:`~integrations.contract_client.LedgerLensContractClient`;
        in tests it is a :class:`~unittest.mock.MagicMock`.
    pipeline_state_fn:
        Callable that returns ``True`` if the scoring pipeline is currently
        *running* (not paused), ``False`` if it is paused.  The monitor
        expects the pipeline to be paused (return ``False``) whenever the
        on-chain contract is paused.
    alert_fn:
        Callable with signature ``alert_fn(mismatch_details: dict) -> None``.
        Invoked with ``severity="CRITICAL"`` whenever the on-chain and
        pipeline states disagree.  The caller decides how to route it.
    poll_interval_seconds:
        How often to reconcile states.  Mismatch detection is bounded by
        this value.  Default 30 s.
    pause_contract_id:
        Optional contract ID passed to ``contract_client.is_paused()``.
        If ``None``, the call is made without arguments (for clients that
        already have the contract ID baked in).
    """

    def __init__(
        self,
        contract_client: Any,
        pipeline_state_fn: Callable[[], bool],
        alert_fn: Callable[[dict], None],
        poll_interval_seconds: int = 30,
        pause_contract_id: str | None = None,
    ) -> None:
        self.contract_client = contract_client
        self.pipeline_state_fn = pipeline_state_fn
        self.alert_fn = alert_fn
        self.poll_interval_seconds = poll_interval_seconds
        self.pause_contract_id = pause_contract_id
        self._running: bool = False

    # ------------------------------------------------------------------
    # State queries (individually mockable in tests)
    # ------------------------------------------------------------------

    def get_onchain_pause_state(self) -> bool:
        """Query the emergency-pause contract and return ``True`` if paused.

        Calls ``contract_client.is_paused(pause_contract_id)`` when a contract
        ID is configured, otherwise ``contract_client.is_paused()``.

        Returns
        -------
        bool
            ``True`` if the on-chain contract reports the system is paused.
        """
        try:
            if self.pause_contract_id is not None:
                return bool(self.contract_client.is_paused(self.pause_contract_id))
            return bool(self.contract_client.is_paused())
        except Exception:
            logger.exception("PauseMonitor: failed to query on-chain pause state")
            # Treat query failure as "unknown" — don't alert, don't clear.
            raise

    def get_pipeline_operational_state(self) -> bool:
        """Return ``True`` if the scoring pipeline is currently running (not paused).

        Delegates to ``pipeline_state_fn``.  ``True`` means operational/running;
        ``False`` means halted/paused.
        """
        return bool(self.pipeline_state_fn())

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def check_parity(self) -> bool:
        """Compare on-chain and pipeline states and alert on mismatch.

        Returns
        -------
        bool
            ``True`` if the states are in parity (no alert fired).
            ``False`` if a mismatch was detected and ``alert_fn`` was called.
        """
        onchain_paused = self.get_onchain_pause_state()
        pipeline_running = self.get_pipeline_operational_state()

        # Parity: on-chain paused ↔ pipeline halted (not running)
        # Equivalently: on-chain running ↔ pipeline running
        in_parity = onchain_paused != pipeline_running
        # Explanation: if on-chain is paused (True) the pipeline should NOT be
        # running (False), so paused==True and running==False is parity
        # (True != False → True).  If on-chain is not paused (False) and pipeline
        # is running (True), False != True → True, also parity.

        if in_parity:
            logger.debug(
                "PauseMonitor: states in parity (onchain_paused=%s, pipeline_running=%s)",
                onchain_paused,
                pipeline_running,
            )
            return True

        # Mismatch detected
        now = datetime.now(UTC).isoformat()
        if onchain_paused and pipeline_running:
            message = (
                "CRITICAL: On-chain emergency pause is ACTIVE but the scoring "
                "pipeline is still RUNNING.  The pipeline must be halted immediately. "
                "See docs/runbooks/pause_monitor_mismatch.md."
            )
        else:
            message = (
                "CRITICAL: On-chain emergency pause is NOT active but the scoring "
                "pipeline is HALTED.  If the pause was lifted intentionally, "
                "restart the pipeline.  See docs/runbooks/pause_monitor_mismatch.md."
            )

        mismatch_details: dict = {
            "severity": "CRITICAL",
            "onchain_paused": onchain_paused,
            "pipeline_paused": not pipeline_running,
            "detected_at": now,
            "message": message,
        }
        logger.critical(
            "PauseMonitor mismatch detected: onchain_paused=%s pipeline_running=%s",
            onchain_paused,
            pipeline_running,
        )
        self.alert_fn(mismatch_details)
        return False

    def run_once(self) -> bool:
        """Perform a single reconciliation tick.

        Returns
        -------
        bool
            ``True`` if states are in parity, ``False`` if a mismatch was found.
        """
        return self.check_parity()

    # ------------------------------------------------------------------
    # Async loop management
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the reconciliation loop indefinitely (async).

        Polls every ``poll_interval_seconds`` seconds.  Exits when
        :meth:`stop` is called.  Errors from individual ticks are logged
        and do not terminate the loop.
        """
        self._running = True
        logger.info(
            "PauseMonitor started (poll_interval=%ds)", self.poll_interval_seconds
        )
        while self._running:
            try:
                self.run_once()
            except Exception:
                logger.exception("PauseMonitor: error during reconciliation tick")
            await asyncio.sleep(self.poll_interval_seconds)
        logger.info("PauseMonitor stopped")

    def stop(self) -> None:
        """Signal the async loop to stop after the current tick completes."""
        self._running = False
        logger.info("PauseMonitor: stop requested")
