"""Soroban event listener for governance and emergency-pause contract events.

Polls the Soroban RPC ``getEvents`` endpoint for two event types:

- ``threshold_changed`` (topic symbol ``t_changed``) — updates the local
  ``config.RISK_SCORE_FLAG_THRESHOLD`` at runtime when M-of-N keyholders
  approve a new value via the governance contract.
- ``contract_paused`` (topic symbol ``c_paused``) — signals the local scoring
  pipeline to halt until an ``contract_unpaused`` event is received.

Usage::

    listener = SorobanEventListener(
        governance_contract_id="C...",
        pause_contract_id="C...",
    )
    listener.run_forever()          # blocking loop
    # Or use asyncio:
    await listener.poll_once()
"""

from __future__ import annotations

import time
from typing import Callable

import requests

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_POLL_INTERVAL = 5  # seconds (≈1 Stellar ledger close)


class SorobanEventListener:
    """Long-polls Soroban RPC for governance and pause events.

    Parameters
    ----------
    governance_contract_id:
        Contract ID of the ThresholdGovernanceContract (issue #238).
    pause_contract_id:
        Contract ID of the EmergencyPauseContract (issue #241).
    rpc_url:
        Soroban RPC endpoint; defaults to ``config.SOROBAN_RPC_URL``.
    poll_interval:
        Seconds between ``getEvents`` calls.
    on_threshold_changed:
        Optional callback invoked with the new threshold value (int)
        whenever a ``threshold_changed`` event is received.
    on_contract_paused:
        Optional callback invoked with the pause reason (str) whenever
        a ``contract_paused`` event is received.
    on_contract_unpaused:
        Optional callback invoked (no args) when an ``contract_unpaused``
        event is received.
    """

    def __init__(
        self,
        governance_contract_id: str,
        pause_contract_id: str,
        rpc_url: str | None = None,
        poll_interval: int = _DEFAULT_POLL_INTERVAL,
        on_threshold_changed: Callable[[int], None] | None = None,
        on_contract_paused: Callable[[str], None] | None = None,
        on_contract_unpaused: Callable[[], None] | None = None,
    ) -> None:
        self.governance_contract_id = governance_contract_id
        self.pause_contract_id = pause_contract_id
        self.rpc_url = rpc_url or config.SOROBAN_RPC_URL
        self.poll_interval = poll_interval
        self._on_threshold_changed = on_threshold_changed or self._default_threshold_handler
        self._on_contract_paused = on_contract_paused or self._default_pause_handler
        self._on_contract_unpaused = on_contract_unpaused or self._default_unpause_handler
        self._start_ledger: int = 0
        self._paused: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    def run_forever(self) -> None:
        """Block and poll indefinitely."""
        logger.info(
            "SorobanEventListener starting (governance=%s pause=%s)",
            self.governance_contract_id,
            self.pause_contract_id,
        )
        while True:
            try:
                self.poll_once()
            except Exception:
                logger.exception("SorobanEventListener: poll error")
            time.sleep(self.poll_interval)

    def poll_once(self) -> list[dict]:
        """Fetch and dispatch new events since the last seen ledger."""
        events = self._fetch_events()
        for event in events:
            self._dispatch(event)
        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_events(self) -> list[dict]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getEvents",
            "params": {
                "startLedger": self._start_ledger,
                "filters": [
                    {
                        "type": "contract",
                        "contractIds": [
                            self.governance_contract_id,
                            self.pause_contract_id,
                        ],
                    }
                ],
                "pagination": {"limit": 200},
            },
        }
        resp = requests.post(self.rpc_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events: list[dict] = data.get("result", {}).get("events", [])
        if events:
            # Advance cursor to avoid reprocessing
            self._start_ledger = events[-1].get("ledger", self._start_ledger) + 1
        return events

    def _dispatch(self, event: dict) -> None:
        topics: list = event.get("topic", [])
        if not topics:
            return

        # Soroban topic[0] is the event name symbol
        event_name = self._sym_to_str(topics[0]) if topics else ""
        contract_id = event.get("contractId", "")
        value = event.get("value")

        if event_name == "t_changed" and contract_id == self.governance_contract_id:
            new_threshold = self._extract_u32(value)
            if new_threshold is not None:
                logger.info("Governance: threshold_changed → %d", new_threshold)
                self._on_threshold_changed(new_threshold)

        elif event_name == "c_paused" and contract_id == self.pause_contract_id:
            reason = self._extract_string(value) or "unknown"
            logger.warning("Emergency pause received: %s", reason)
            self._paused = True
            self._on_contract_paused(reason)

        elif event_name == "c_unpaused" and contract_id == self.pause_contract_id:
            logger.info("Contract unpaused")
            self._paused = False
            self._on_contract_unpaused()

    @staticmethod
    def _sym_to_str(scval: dict) -> str:
        return scval.get("sym", scval.get("str", ""))

    @staticmethod
    def _extract_u32(scval: dict | None) -> int | None:
        if scval is None:
            return None
        return scval.get("u32")

    @staticmethod
    def _extract_string(scval: dict | None) -> str | None:
        if scval is None:
            return None
        return scval.get("str") or scval.get("sym")

    # ------------------------------------------------------------------
    # Default handlers (update config in-process)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_threshold_handler(new_threshold: int) -> None:
        config.RISK_SCORE_FLAG_THRESHOLD = new_threshold
        logger.info(
            "RISK_SCORE_FLAG_THRESHOLD updated to %d via on-chain governance",
            new_threshold,
        )

    @staticmethod
    def _default_pause_handler(reason: str) -> None:
        logger.critical(
            "Scoring pipeline HALTED by emergency pause: %s", reason
        )

    @staticmethod
    def _default_unpause_handler() -> None:
        logger.info("Scoring pipeline resumed after emergency unpause")
