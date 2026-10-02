"""Wallet list override check for allowlisting and denylisting.

Resolution Strategy
-------------------
If a wallet appears in both the allowlist and denylist (a conflict), the
allowlist takes precedence and the wallet is treated as allowed (score 0).
This prioritizes operator intent to explicitly allow over deny decisions.

Audit Logging
-------------
Administrative list mutations (add/remove on the allowlist or denylist) emit
structured, append-only audit events via ``audit_trail`` so that every override
is reconstructable (actor, action, before/after state, timestamp) and
hash-chained for tamper-evidence.
"""

from __future__ import annotations

import json
import os
import time

from config import config
from utils.logging import get_logger

try:
    from audit_trail import record_audit_event
except Exception:  # pragma: no cover - audit trail is optional at import time
    record_audit_event = None

logger = get_logger(__name__)


class ListOverride:
    """Manages hot-reloading allowlists and denylists to override risk scores."""

    def __init__(
        self,
        allowlist_path: str = "data/allowlist.json",
        denylist_path: str = "data/denylist.json",
    ):
        self.allowlist_path = allowlist_path
        self.denylist_path = denylist_path
        self._allowlist: set[str] = set()
        self._denylist: set[str] = set()
        self._last_loaded: float = 0.0
        self._reload()

    def _reload(self) -> None:
        """Reload allowlist and denylist files from disk."""
        # Allowlist
        if os.path.exists(self.allowlist_path):
            try:
                with open(self.allowlist_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._allowlist = set(data)
                    else:
                        logger.warning(
                            "Allowlist file at %s is not a list. Ignoring.",
                            self.allowlist_path,
                        )
                        self._allowlist = set()
            except Exception as e:
                logger.warning("Failed to load allowlist from %s: %s", self.allowlist_path, e)
                self._allowlist = set()
        else:
            self._allowlist = set()

        # Denylist
        if os.path.exists(self.denylist_path):
            try:
                with open(self.denylist_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._denylist = set(data)
                    else:
                        logger.warning(
                            "Denylist file at %s is not a list. Ignoring.",
                            self.denylist_path,
                        )
                        self._denylist = set()
            except Exception as e:
                logger.warning("Failed to load denylist from %s: %s", self.denylist_path, e)
                self._denylist = set()
        else:
            self._denylist = set()

        self._last_loaded = time.time()

    def _audit(self, action: str, wallet: str, before: bool, after: bool) -> None:
        """Emit a structured audit event for an administrative list mutation."""
        if record_audit_event is None:
            return
        try:
            record_audit_event(
                actor=os.environ.get("AUDIT_ACTOR", "system"),
                action=action,
                target=wallet,
                before={"listed": before},
                after={"listed": after},
            )
        except Exception as e:  # pragma: no cover - never break overrides on audit failure
            logger.warning("Failed to record audit event for %s: %s", action, e)

    def add_to_allowlist(self, wallet: str) -> None:
        """Administratively add a wallet to the allowlist (audited)."""
        before = wallet in self._allowlist
        self._allowlist.add(wallet)
        self._audit("allowlist.add", wallet, before, True)

    def remove_from_allowlist(self, wallet: str) -> None:
        """Administratively remove a wallet from the allowlist (audited)."""
        before = wallet in self._allowlist
        self._allowlist.discard(wallet)
        self._audit("allowlist.remove", wallet, before, False)

    def add_to_denylist(self, wallet: str) -> None:
        """Administratively add a wallet to the denylist (audited)."""
        before = wallet in self._denylist
        self._denylist.add(wallet)
        self._audit("denylist.add", wallet, before, True)

    def remove_from_denylist(self, wallet: str) -> None:
        """Administratively remove a wallet from the denylist (audited)."""
        before = wallet in self._denylist
        self._denylist.discard(wallet)
        self._audit("denylist.remove", wallet, before, False)

    def check(self, wallet: str) -> int | None:
        """Returns 0 (allowlist), 100 (denylist), or None (not listed)."""
        now = time.time()
        interval = getattr(config, "LIST_RELOAD_INTERVAL_SECONDS", 60)
        if now - self._last_loaded >= interval:
            self._reload()

        if wallet in self._allowlist:
            logger.warning("Wallet %s overridden to 0 (source: allowlist)", wallet)
            return 0
        if wallet in self._denylist:
            logger.warning("Wallet %s overridden to 100 (source: denylist)", wallet)
            return 100
        return None
