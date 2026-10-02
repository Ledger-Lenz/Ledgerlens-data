"""Structured, tamper-evident audit trail for administrative/override actions.

This module provides an append-only audit log distinct from general
application logs. Each entry records the actor, action, before/after state
and a timestamp, and is linked to the previous entry via a SHA-256 hash chain
so that any modification of a past entry is detectable.

Administrative endpoints (threshold overrides, manual score overrides,
tenant config changes) should call :func:`record_audit_event` for every
state-changing action.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Durable store for audit entries. Kept separate from general application logs.
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "audit_trail.log")

# Genesis hash used as the ``prev_hash`` of the first entry in the chain.
GENESIS_HASH = "0" * 64

_lock = threading.Lock()


def _canonical(entry: Dict[str, Any]) -> str:
    """Return a deterministic JSON encoding of an entry for hashing."""
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


def _compute_hash(entry: Dict[str, Any]) -> str:
    """Compute the SHA-256 hash of an entry (excluding its own ``hash``)."""
    payload = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _read_entries() -> List[Dict[str, Any]]:
    """Read all audit entries from the durable store, in append order."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    entries: List[Dict[str, Any]] = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _last_hash(entries: List[Dict[str, Any]]) -> str:
    """Return the hash of the most recent entry, or the genesis hash."""
    if not entries:
        return GENESIS_HASH
    return entries[-1].get("hash", GENESIS_HASH)


def record_audit_event(
    actor: str,
    action: str,
    before: Optional[Any] = None,
    after: Optional[Any] = None,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a structured audit event to the tamper-evident audit trail.

    Args:
        actor: Identity of the principal performing the action.
        action: Machine-readable action name (e.g. ``threshold.override``).
        before: State prior to the action (any JSON-serialisable value).
        after: State after the action (any JSON-serialisable value).
        target: Optional identifier of the affected resource.
        metadata: Optional additional structured context.

    Returns:
        The persisted audit entry, including its ``hash`` and ``prev_hash``.
    """
    with _lock:
        entries = _read_entries()
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "target": target,
            "before": before,
            "after": after,
            "metadata": metadata or {},
            "prev_hash": _last_hash(entries),
        }
        entry["hash"] = _compute_hash(entry)

        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(_canonical(entry) + "\n")

        return entry


def verify_audit_trail() -> bool:
    """Verify the integrity of the audit trail hash chain.

    Returns:
        ``True`` if every entry's hash matches its contents and links to the
        previous entry. ``False`` if any entry has been tampered with.
    """
    entries = _read_entries()
    prev_hash = GENESIS_HASH
    for entry in entries:
        if entry.get("prev_hash") != prev_hash:
            return False
        if entry.get("hash") != _compute_hash(entry):
            return False
        prev_hash = entry["hash"]
    return True


def query_audit_trail(
    actor: Optional[str] = None,
    action: Optional[str] = None,
    target: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the audit trail, optionally filtering by actor/action/target."""
    results = []
    for entry in _read_entries():
        if actor is not None and entry.get("actor") != actor:
            continue
        if action is not None and entry.get("action") != action:
            continue
        if target is not None and entry.get("target") != target:
            continue
        results.append(entry)
    return results
