"""Wallet list override check for allowlisting and denylisting."""

import json
import os
import time

from config import config
from utils.logging import get_logger

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
