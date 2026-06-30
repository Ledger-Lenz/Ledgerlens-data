"""Automated incident response for high-severity LedgerLens alerts.

When the detection system fires a high-severity alert (risk score > 90,
Benford MAD > 0.05, or emergency_drift alert type), IncidentResponder:

  1. Snapshots the wallet's current risk score history.
  2. Generates a preliminary forensic report.
  3. Creates an incident record in the in-process store (or injected backend).
  4. Posts a JSON notification to the configured webhook.

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
from typing import Any

import yaml

from utils.logging import get_logger

logger = get_logger(__name__)

_PLAYBOOK_PATH = Path(__file__).parent.parent / "data" / "playbooks" / "high_risk_wallet.yaml"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IncidentRecord:
    incident_id: str
    wallet_hash: str           # SHA-256(wallet), not raw address
    alert_fingerprint: str     # SHA-256(wallet + alert_type + score bucket)
    alert_type: str
    risk_score: int
    created_at: str            # ISO 8601 UTC
    status: str = "open"
    report_summary: dict = field(default_factory=dict)
    risk_history_snapshot: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


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
    """

    def __init__(
        self,
        playbook_path: str | Path | None = None,
        webhook_url: str | None = None,
        incident_store: dict | None = None,
        dedup_window_seconds: int | None = None,
    ) -> None:
        self._playbook = self._load_playbook(playbook_path or _PLAYBOOK_PATH)
        self._webhook_url = webhook_url or os.getenv("INCIDENT_WEBHOOK_URL")
        if self._webhook_url and self._webhook_url.startswith("http://"):
            raise ValueError("Webhook URL must use HTTPS")
        self._store: dict[str, IncidentRecord] = incident_store if incident_store is not None else {}
        dedup_cfg = self._playbook.get("deduplication", {})
        self._dedup_window = dedup_window_seconds if dedup_window_seconds is not None else dedup_cfg.get("window_seconds", 3600)
        self._dedup_timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

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
        fingerprint = _alert_fingerprint(wallet, alert.get("alert_type", ""), alert.get("risk_score", 0))

        with self._lock:
            if self._is_duplicate(fingerprint):
                logger.info("Duplicate alert suppressed for wallet_hash=%s", wallet_hash)
                return None
            self._dedup_timestamps[fingerprint] = time.monotonic()

        return self._execute_playbook(wallet, wallet_hash, fingerprint, alert)

    def simulate(self, wallet: str, risk_score: int = 95, alert_type: str = "high_risk_wallet") -> IncidentRecord | None:
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

    @property
    def incidents(self) -> dict[str, IncidentRecord]:
        """Read-only view of all recorded incidents keyed by incident_id."""
        return dict(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
                self._run_step(action, params, incident, wallet, alert)
            except Exception as exc:
                logger.warning("Playbook step %r failed: %s", action, exc)

        with self._lock:
            self._store[incident.incident_id] = incident

        logger.info(
            "Incident created: id=%s wallet_hash=%s score=%d",
            incident.incident_id,
            wallet_hash,
            incident.risk_score,
        )
        return incident

    def _run_step(
        self,
        action: str,
        params: dict,
        incident: IncidentRecord,
        wallet: str,
        alert: dict,
    ) -> None:
        if action == "snapshot_risk_score_history":
            incident.risk_history_snapshot = self._snapshot_risk_history(wallet, params)

        elif action == "generate_preliminary_forensic_report":
            incident.report_summary = self._generate_report_summary(wallet, alert, params)

        elif action == "create_db_incident_record":
            # The record is committed to self._store after all steps complete;
            # this step is a deliberate no-op so the YAML step ordering is
            # preserved without double-writing.
            pass

        elif action == "post_webhook_notification":
            self._post_webhook(incident, params)

        else:
            logger.warning("Unknown playbook action: %r", action)

    def _snapshot_risk_history(self, wallet: str, params: dict) -> list[dict]:
        try:
            from detection.risk_score_store import RiskScoreStore

            store = RiskScoreStore()
            lookback = params.get("lookback_days", 30)
            history = store.get_history(wallet, days=lookback)
            return [{"timestamp": str(ts), "score": score} for ts, score in history]
        except Exception as exc:
            logger.debug("Risk history snapshot unavailable: %s", exc)
            return []

    def _generate_report_summary(self, wallet: str, alert: dict, params: dict) -> dict:
        try:
            from detection.forensic_report import ForensicReportGenerator

            generator = ForensicReportGenerator()
            import pandas as pd

            report = generator.generate(
                wallet=wallet,
                asset_pair="XLM:native",
                feature_row=pd.Series(dtype=float),
                wallet_trades=pd.DataFrame(),
                orderbook_events=None,
            )
            d = report.to_dict()
            return {
                "report_id": d.get("report_id"),
                "risk_score": d.get("risk_score"),
                "verdict": d.get("verdict"),
                "top_features": [f.get("feature") for f in d.get("top_shap_features", [])[:5]],
            }
        except Exception as exc:
            logger.debug("Preliminary report generation unavailable: %s", exc)
            return {"alert_score": alert.get("risk_score"), "alert_type": alert.get("alert_type")}

    def _post_webhook(self, incident: IncidentRecord, params: dict) -> None:
        if not self._webhook_url:
            logger.debug("No webhook URL configured; skipping notification")
            return

        import requests

        payload = {
            "incident_id": incident.incident_id,
            "wallet_hash": incident.wallet_hash,  # hashed, not raw
            "alert_type": incident.alert_type,
            "risk_score": incident.risk_score,
            "created_at": incident.created_at,
            "status": incident.status,
        }
        if params.get("include_report_summary") and incident.report_summary:
            payload["report_summary"] = incident.report_summary

        try:
            resp = requests.post(self._webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Webhook notification sent for incident %s", incident.incident_id)
        except Exception as exc:
            # Do NOT include self._webhook_url in the log message.
            logger.warning("Webhook notification failed for incident %s: %s", incident.incident_id, exc)

    @staticmethod
    def _load_playbook(path: str | Path) -> dict:
        try:
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Playbook not found at %s; using empty playbook", path)
            return {}
        except Exception as exc:
            logger.warning("Failed to load playbook from %s: %s", path, exc)
            return {}
