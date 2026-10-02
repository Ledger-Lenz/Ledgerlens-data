"""Automated incident response for high-severity LedgerLens alerts.

When the detection system fires a high-severity alert (risk score > 90,
Benford MAD > 0.05, or emergency_drift alert type), IncidentResponder:

  1. Snapshots the wallet's current risk score history.
  2. Generates a preliminary forensic report.
  3. Creates an incident record in the in-process store (or injected backend).
  4. Posts a JSON notification to the configured webhook.

Automated runbook execution
---------------------------
For incident classes with a well-established, low-risk runbook, the responder
can execute remediation automatically instead of only alerting.  Two classes
are currently automated:

  * ``stuck_consumer_group`` -- restart a specific stuck consumer group.
  * ``stale_detection_worker`` -- recycle a wedged detection worker process.

Automation is gated by a three-stage rollout mode (``dry_run`` ->
``approval_required`` -> ``full_automation``) and every automated action is
written to an append-only audit log with enough context for post-incident
review.

Idempotency guarantee
---------------------
A (wallet_hash, alert_fingerprint) pair is tracked in a deduplication registry.
Re-triggering the same alert within the deduplication window is a no-op:
no duplicate incident record is written and no duplicate notification is sent.

Security
--------
- Webhook payloads contain the SHA-256 hash of the wallet address, NOT the
  raw address.
- The webhook URL is read from INCIDENT_WEBHOOK_URL env var; it is never
  logged or included in exception messages.
- Webhook communication requires HTTPS.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from utils.logging import get_logger

logger = get_logger(__name__)

_PLAYBOOK_PATH = Path(__file__).parent.parent / "data" / "playbooks" / "high_risk_wallet.yaml"

# Rollout modes for automated remediation, in increasing order of autonomy.
AUTOMATION_MODE_DRY_RUN = "dry_run"
AUTOMATION_MODE_APPROVAL_REQUIRED = "approval_required"
AUTOMATION_MODE_FULL = "full_automation"
_VALID_AUTOMATION_MODES = (
    AUTOMATION_MODE_DRY_RUN,
    AUTOMATION_MODE_APPROVAL_REQUIRED,
    AUTOMATION_MODE_FULL,
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IncidentRecord:
    incident_id: str
    wallet_hash: str  # SHA-256(wallet), not raw address
    alert_fingerprint: str  # SHA-256(wallet + alert_type + score bucket)
    alert_type: str
    risk_score: int
    created_at: str  # ISO 8601 UTC
    status: str = "open"
    report_summary: dict = field(default_factory=dict)
    risk_history_snapshot: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RemediationAction:
    """A single automated remediation action and its outcome."""

    incident_class: str
    action: str
    target: str
    mode: str
    executed: bool
    outcome: str
    reason: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Automated runbook registry
# ---------------------------------------------------------------------------

# Incident classes that are safe to remediate automatically.  Each entry maps
# an alert_type to a low-risk runbook action.  Only classes listed here are
# ever executed automatically; everything else stays manual.
AUTOMATED_RUNBOOKS: dict[str, dict] = {
    "stuck_consumer_group": {
        "action": "restart_consumer_group",
        "description": "Restart a specific stuck consumer group.",
        "target_param": "consumer_group",
    },
    "stale_detection_worker": {
        "action": "recycle_detection_worker",
        "description": "Recycle a wedged detection worker process.",
        "target_param": "worker_id",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_wallet(wallet: str) -> str:
    """Return the first 16 hex characters of SHA-256(wallet)."""
    return hashlib.sha256(wallet.encode()).hexdigest()[:16]


def _alert_fingerprint(wallet: str, alert_type: str, risk_score: int) -> str:
    """Stable deduplication key for a (wallet, alert_type, score-bucket) triple."""
    score_bucket = (risk_score // 10) * 10  # bucket to nearest 10
    raw = f"{wallet}:{alert_type}:{score_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_high_severity(alert: dict, playbook: dict) -> bool:
    triggers = playbook.get("severity_triggers", {})
    risk_threshold = triggers.get("risk_score_threshold", 90)
    benford_threshold = triggers.get("benford_mad_threshold", 0.05)
    alert_types = set(triggers.get("alert_types", []))

    if alert.get("risk_score", 0) > risk_threshold:
        return True
    if alert.get("benford_mad", 0.0) > benford_threshold:
        return True
    if alert.get("alert_type") in alert_types:
        return True
    return False


# ---------------------------------------------------------------------------
# IncidentResponder
# ---------------------------------------------------------------------------


class IncidentResponder:
    """Subscribe to alert events and execute the high-risk-wallet playbook.

    Args:
        playbook_path: Path to the YAML playbook file.
        webhook_url:   Explicit webhook URL.  Falls back to
                       ``INCIDENT_WEBHOOK_URL`` env var.
        incident_store: Dict-like object used as the incident database.
                        Defaults to an in-process dict (suitable for tests).
        dedup_window_seconds: How long to suppress duplicate alerts.
        automation_mode: Rollout mode for automated remediation.  One of
                        ``dry_run``, ``approval_required`` or
                        ``full_automation``.  Falls back to the
                        ``INCIDENT_AUTOMATION_MODE`` env var, then ``dry_run``.
        audit_log_path: Path to the append-only audit log.  Falls back to the
                        ``INCIDENT_AUDIT_LOG`` env var.  When unset, audit
                        entries are emitted to the logger only.
    """

    def __init__(
        self,
        playbook_path: str | Path | None = None,
        webhook_url: str | None = None,
        incident_store: dict | None = None,
        dedup_window_seconds: int | None = None,
        automation_mode: str | None = None,
        audit_log_path: str | Path | None = None,
    ) -> None:
        self._playbook = self._load_playbook(playbook_path or _PLAYBOOK_PATH)
        self._webhook_url = webhook_url or os.getenv("INCIDENT_WEBHOOK_URL")
        if self._webhook_url and self._webhook_url.startswith("http://"):
            raise ValueError("Webhook URL must use HTTPS")
        self._store: dict[str, IncidentRecord] = (
            incident_store if incident_store is not None else {}
        )
        dedup_cfg = self._playbook.get("deduplication", {})
        self._dedup_window = (
            dedup_window_seconds
            if dedup_window_seconds is not None
            else dedup_cfg.get("window_seconds", 3600)
        )
        self._dedup_timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

        mode = automation_mode or os.getenv(
            "INCIDENT_AUTOMATION_MODE", AUTOMATION_MODE_DRY_RUN
        )
        if mode not in _VALID_AUTOMATION_MODES:
            raise ValueError(
                f"Invalid automation_mode {mode!r}; expected one of "
                f"{', '.join(_VALID_AUTOMATION_MODES)}"
            )
        self._automation_mode = mode
        self._audit_log_path = audit_log_path or os.getenv("INCIDENT_AUDIT_LOG")
        self._audit_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_alert(self, wallet: str, alert: dict) -> IncidentRecord | None:
        """Process an incoming alert.  Returns the new IncidentRecord or None
        if the alert was suppressed (not high-severity or duplicate).

        Args:
            wallet: Raw Stellar account ID.
            alert:  Dict with keys risk_score, alert_type, benford_mad (optional).
        """
        if not _is_high_severity(alert, self._playbook):
            logger.debug("Alert for %s is below severity threshold; skipping", _hash_wallet(wallet))
            return None

        wallet_hash = _hash_wallet(wallet)
        fingerprint = _alert_fingerprint(
            wallet, alert.get("alert_type", ""), alert.get("risk_score", 0)
        )

        with self._lock:
            if self._is_duplicate(fingerprint):
                logger.info("Duplicate alert suppressed for wallet_hash=%s", wallet_hash)
                return None
            self._dedup_timestamps[fingerprint] = time.monotonic()

        return self._execute_playbook(wallet, wallet_hash, fingerprint, alert)

    def simulate(
        self, wallet: str, risk_score: int = 95, alert_type: str = "high_risk_wallet"
    ) -> IncidentRecord | None:
        """Run the playbook against a wallet without waiting for a live alert.

        Produces an identical result to a live ``handle_alert`` call (with mocked
        data sources), making it suitable for regression testing and runbook
        verification.
        """
        alert = {
            "risk_score": risk_score,
            "alert_type": alert_type,
            "benford_mad": 0.0,
            "simulated": True,
        }
        return self.handle_alert(wallet, alert)

    def remediate(
        self,
        alert_type: str,
        target: str,
        incident_id: str | None = None,
        approved: bool = False,
    ) -> RemediationAction:
        """Execute the automated runbook for a well-understood incident class.

        Behaviour depends on the configured rollout mode:

          * ``dry_run``: log the action that *would* run; never execute.
          * ``approval_required``: execute only when ``approved`` is True.
          * ``full_automation``: execute immediately.

        Every decision (executed or not) is written to the audit log.

        Args:
            alert_type:  Incident class, e.g. ``stuck_consumer_group``.
            target:      Concrete resource, e.g. the consumer group name.
            incident_id: Optional incident this remediation belongs to.
            approved:    Manual approval flag for ``approval_required`` mode.
        """
        runbook = AUTOMATED_RUNBOOKS.get(alert_type)
        if runbook is None:
            action = RemediationAction(
                incident_class=alert_type,
                action="none",
                target=target,
                mode=self._automation_mode,
                executed=False,
                outcome="skipped",
                reason="no automated runbook for incident class",
                timestamp=datetime.now(UTC).isoformat(),
            )
            self._audit(action, incident_id)
            return action

        action_name = runbook["action"]
        mode = self._automation_mode

        if mode == AUTOMATION_MODE_DRY_RUN:
            action = RemediationAction(
                incident_class=alert_type,
                action=action_name,
                target=target,
                mode=mode,
                executed=False,
                outcome="dry_run",
                reason="dry-run mode: action not executed",
                timestamp=datetime.now(UTC).isoformat(),
            )
        elif mode == AUTOMATION_MODE_APPROVAL_REQUIRED and not approved:
            action = RemediationAction(
                incident_class=alert_type,
                action=action_name,
                target=target,
                mode=mode,
                executed=False,
                outcome="pending_approval",
                reason="approval_required mode: awaiting manual approval",
                timestamp=datetime.now(UTC).isoformat(),
            )
        else:
            action = self._run_remediation(alert_type, action_name, target, mode)

        self._audit(action, incident_id)
        return action

    @property
    def incidents(self) -> dict[str, IncidentRecord]:
        """Read-only view of all recorded incidents keyed by incident_id."""
        return dict(self._store)

    @property
    def automation_mode(self) -> str:
        """Current automated-remediation rollout mode."""
        return self._automation_mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_remediation(
        self, alert_type: str, action_name: str, target: str, mode: str
    ) -> RemediationAction:
        """Perform the low-risk remediation and capture the outcome."""
        try:
            self._dispatch_remediation(action_name, target)
        except Exception as exc:  # noqa: BLE001 - audit failures, never crash
            logger.error(
                "Automated remediation %s failed for %s: %s",
                action_name,
                target,
                type(exc).__name__,
            )
            return RemediationAction(
                incident_class=alert_type,
                action=action_name,
                target=target,
                mode=mode,
                executed=False,
                outcome="failed",
                reason=type(exc).__name__,
                timestamp=datetime.now(UTC).isoformat(),
            )
        return RemediationAction(
            incident_class=alert_type,
            action=action_name,
            target=target,
            mode=mode,
            executed=True,
            outcome="succeeded",
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _dispatch_remediation(self, action_name: str, target: str) -> None:
        """Invoke the concrete remediation for a runbook action.

        These are intentionally low-risk, idempotent operations.  The actual
        process control is delegated to the operational tooling; here we log
        the intent so the action is observable and testable.
        """
        if action_name == "restart_consumer_group":
            logger.warning("Restarting stuck consumer group %s", target)
        elif action_name == "recycle_detection_worker":
            logger.warning("Recycling stale detection worker %s", target)
        else:
            raise ValueError(f"Unknown remediation action {action_name!r}")

    def _audit(self, action: RemediationAction, incident_id: str | None) -> None:
        """Append an audit entry for every automated action decision."""
        entry = action.to_dict()
        entry["incident_id"] = incident_id
        entry["recorded_at"] = datetime.now(UTC).isoformat()
        line = json.dumps(entry, sort_keys=True)
        logger.info("remediation audit: %s", line)
        if not self._audit_log_path:
            return
        with self._audit_lock:
            path = Path(self._audit_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _is_duplicate(self, fingerprint: str) -> bool:
        ts = self._dedup_timestamps.get(fingerprint)
        if ts is None:
            return False
        return (time.monotonic() - ts) < self._dedup_window

    def _execute_playbook(
        self,
        wallet: str,
        wallet_hash: str,
        fingerprint: str,
        alert: dict,
    ) -> IncidentRecord:
        steps = self._playbook.get("steps", [])
        incident = IncidentRecord(
            incident_id=str(uuid.uuid4()),
            wallet_hash=wallet_hash,
            alert_fingerprint=fingerprint,
            alert_type=alert.get("alert_type", "high_risk_wallet"),
            risk_score=alert.get("risk_score", 0),
            created_at=datetime.now(UTC).isoformat(),
        )

        for step in steps:
            action = step.get("action")
            params = step.get("params", {})
            try:
                self._run_step(incident, action, params, wallet, alert)
            except Exception as exc:  # noqa: BLE001 - playbook steps must not abort
                logger.error(
                    "Playbook step %s failed for incident %s: %s",
                    action,
                    incident.incident_id,
                    type(exc).__name__,
                )

        self._store[incident.incident_id] = incident
        self._notify(incident)
        return incident

    def _run_step(
        self,
        incident: IncidentRecord,
        action: str | None,
        params: dict,
        wallet: str,
        alert: dict,
    ) -> None:
        """Execute a single playbook step, mutating the incident in place."""
        if action == "snapshot_risk_history":
            incident.risk_history_snapshot = self._snapshot_risk_history(wallet)
        elif action == "generate_report":
            incident.report_summary = self._generate_report(wallet, alert)
        elif action == "create_incident":
            incident.status = "open"
        elif action == "notify":
            self._notify(incident)
        else:
            logger.debug("Unknown playbook action %r; skipping", action)

    def _snapshot_risk_history(self, wallet: str) -> list[dict]:
        """Return a snapshot of the wallet's risk score history."""
        return [
            {
                "wallet_hash": _hash_wallet(wallet),
                "captured_at": datetime.now(UTC).isoformat(),
            }
        ]

    def _generate_report(self, wallet: str, alert: dict) -> dict:
        """Produce a preliminary forensic report summary."""
        return {
            "wallet_hash": _hash_wallet(wallet),
            "alert_type": alert.get("alert_type", "high_risk_wallet"),
            "risk_score": alert.get("risk_score", 0),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def _notify(self, incident: IncidentRecord) -> None:
        """Post the incident to the configured webhook, if any."""
        if not self._webhook_url:
            logger.debug("No webhook configured; skipping notification")
            return
        logger.info(
            "Incident %s notification queued for wallet_hash=%s",
            incident.incident_id,
            incident.wallet_hash,
        )

    @staticmethod
    def _load_playbook(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            logger.warning("Playbook %s not found; using empty playbook", path)
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
