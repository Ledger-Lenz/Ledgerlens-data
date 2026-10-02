"""Privacy budget accounting dashboard and alerting.

Provides visibility into cumulative epsilon/delta consumption per tenant
and time window, with configurable alerts when thresholds are exceeded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from detection.privacy.budget_tracker import BudgetTracker
from monitoring.metrics_collector import MetricsCollector
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PrivacyBudgetSnapshot:
    """Snapshot of privacy budget consumption at a point in time."""

    timestamp: str
    tenant_id: str
    epsilon_consumed: float
    delta_consumed: float
    epsilon_budget: float
    delta_budget: float
    epsilon_remaining: float
    delta_remaining: float
    epsilon_utilization_pct: float
    delta_utilization_pct: float
    window_name: str


@dataclass
class BudgetAlertThresholds:
    """Configuration for privacy budget alerts."""

    epsilon_warning_threshold_pct: float = 80.0
    epsilon_critical_threshold_pct: float = 95.0
    delta_warning_threshold_pct: float = 80.0
    delta_critical_threshold_pct: float = 95.0


class PrivacyBudgetDashboard:
    """Dashboard for privacy budget monitoring and alerting."""

    def __init__(
        self,
        budget_tracker: BudgetTracker,
        metrics_collector: MetricsCollector,
        alert_thresholds: BudgetAlertThresholds | None = None,
    ) -> None:
        """Initialize budget dashboard.

        Args:
            budget_tracker: BudgetTracker instance tracking epsilon/delta consumption.
            metrics_collector: MetricsCollector for exporting metrics.
            alert_thresholds: Thresholds for triggering alerts.
        """
        self.budget_tracker = budget_tracker
        self.metrics_collector = metrics_collector
        self.alert_thresholds = alert_thresholds or BudgetAlertThresholds()

    def get_current_budget_snapshot(
        self,
        tenant_id: str,
        window_name: str = "global",
    ) -> PrivacyBudgetSnapshot:
        """Get current privacy budget snapshot for a tenant.

        Args:
            tenant_id: Tenant identifier.
            window_name: Time window name (e.g., 'daily', 'monthly', 'global').

        Returns:
            PrivacyBudgetSnapshot with current consumption and remaining budget.
        """
        budget_info = self.budget_tracker.get_budget_info(tenant_id, window_name)

        epsilon_consumed = budget_info.get("epsilon_consumed", 0.0)
        delta_consumed = budget_info.get("delta_consumed", 0.0)
        epsilon_budget = budget_info.get("epsilon_budget", 1.0)
        delta_budget = budget_info.get("delta_budget", 1.0)

        epsilon_remaining = max(0.0, epsilon_budget - epsilon_consumed)
        delta_remaining = max(0.0, delta_budget - delta_consumed)

        epsilon_utilization_pct = (epsilon_consumed / epsilon_budget * 100) if epsilon_budget > 0 else 0
        delta_utilization_pct = (delta_consumed / delta_budget * 100) if delta_budget > 0 else 0

        snapshot = PrivacyBudgetSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            tenant_id=tenant_id,
            epsilon_consumed=epsilon_consumed,
            delta_consumed=delta_consumed,
            epsilon_budget=epsilon_budget,
            delta_budget=delta_budget,
            epsilon_remaining=epsilon_remaining,
            delta_remaining=delta_remaining,
            epsilon_utilization_pct=epsilon_utilization_pct,
            delta_utilization_pct=delta_utilization_pct,
            window_name=window_name,
        )

        return snapshot

    def export_metrics(self, tenant_id: str, window_name: str = "global") -> None:
        """Export budget metrics to monitoring system.

        Args:
            tenant_id: Tenant identifier.
            window_name: Time window name.
        """
        snapshot = self.get_current_budget_snapshot(tenant_id, window_name)

        try:
            self.metrics_collector.record_gauge(
                "privacy_budget_epsilon_consumed_total",
                snapshot.epsilon_consumed,
                tags={"tenant": tenant_id, "window": window_name},
            )
            self.metrics_collector.record_gauge(
                "privacy_budget_epsilon_remaining",
                snapshot.epsilon_remaining,
                tags={"tenant": tenant_id, "window": window_name},
            )
            self.metrics_collector.record_gauge(
                "privacy_budget_epsilon_utilization_pct",
                snapshot.epsilon_utilization_pct,
                tags={"tenant": tenant_id, "window": window_name},
            )

            self.metrics_collector.record_gauge(
                "privacy_budget_delta_consumed_total",
                snapshot.delta_consumed,
                tags={"tenant": tenant_id, "window": window_name},
            )
            self.metrics_collector.record_gauge(
                "privacy_budget_delta_remaining",
                snapshot.delta_remaining,
                tags={"tenant": tenant_id, "window": window_name},
            )
            self.metrics_collector.record_gauge(
                "privacy_budget_delta_utilization_pct",
                snapshot.delta_utilization_pct,
                tags={"tenant": tenant_id, "window": window_name},
            )

            logger.debug(
                "Exported privacy budget metrics for tenant=%s window=%s",
                tenant_id,
                window_name,
            )
        except Exception as exc:
            logger.error(
                "Failed to export privacy budget metrics for tenant=%s: %s",
                tenant_id,
                exc,
            )

    def check_budget_alerts(self, tenant_id: str, window_name: str = "global") -> list[dict]:
        """Check if budget consumption exceeds alert thresholds.

        Args:
            tenant_id: Tenant identifier.
            window_name: Time window name.

        Returns:
            List of alert dictionaries with severity, message, and snapshot data.
        """
        snapshot = self.get_current_budget_snapshot(tenant_id, window_name)
        alerts = []

        if snapshot.epsilon_utilization_pct >= self.alert_thresholds.epsilon_critical_threshold_pct:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "epsilon",
                "message": f"Epsilon budget {snapshot.epsilon_utilization_pct:.1f}% consumed "
                f"(threshold: {self.alert_thresholds.epsilon_critical_threshold_pct}%) "
                f"for tenant={tenant_id} window={window_name}",
                "snapshot": asdict(snapshot),
            })
        elif snapshot.epsilon_utilization_pct >= self.alert_thresholds.epsilon_warning_threshold_pct:
            alerts.append({
                "severity": "WARNING",
                "metric": "epsilon",
                "message": f"Epsilon budget {snapshot.epsilon_utilization_pct:.1f}% consumed "
                f"(threshold: {self.alert_thresholds.epsilon_warning_threshold_pct}%) "
                f"for tenant={tenant_id} window={window_name}",
                "snapshot": asdict(snapshot),
            })

        if snapshot.delta_utilization_pct >= self.alert_thresholds.delta_critical_threshold_pct:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "delta",
                "message": f"Delta budget {snapshot.delta_utilization_pct:.1f}% consumed "
                f"(threshold: {self.alert_thresholds.delta_critical_threshold_pct}%) "
                f"for tenant={tenant_id} window={window_name}",
                "snapshot": asdict(snapshot),
            })
        elif snapshot.delta_utilization_pct >= self.alert_thresholds.delta_warning_threshold_pct:
            alerts.append({
                "severity": "WARNING",
                "metric": "delta",
                "message": f"Delta budget {snapshot.delta_utilization_pct:.1f}% consumed "
                f"(threshold: {self.alert_thresholds.delta_warning_threshold_pct}%) "
                f"for tenant={tenant_id} window={window_name}",
                "snapshot": asdict(snapshot),
            })

        return alerts

    def generate_dashboard_json(self, tenant_ids: list[str]) -> dict[str, Any]:
        """Generate dashboard data as JSON for visualization/export.

        Args:
            tenant_ids: List of tenant IDs to include in dashboard.

        Returns:
            Dict with dashboard data for all tenants and windows.
        """
        dashboard = {
            "generated_at": datetime.now(UTC).isoformat(),
            "alert_thresholds": asdict(self.alert_thresholds),
            "tenants": {},
        }

        for tenant_id in tenant_ids:
            tenant_data = {
                "windows": {},
                "alerts": [],
            }

            for window_name in ["global", "daily", "monthly"]:
                snapshot = self.get_current_budget_snapshot(tenant_id, window_name)
                tenant_data["windows"][window_name] = asdict(snapshot)

                alerts = self.check_budget_alerts(tenant_id, window_name)
                tenant_data["alerts"].extend(alerts)

            dashboard["tenants"][tenant_id] = tenant_data

        return dashboard

    def generate_grafana_panel_json(self) -> dict[str, Any]:
        """Generate Grafana dashboard panel configuration.

        Returns:
            Grafana panel JSON for privacy budget burn-down visualization.
        """
        panel = {
            "title": "Privacy Budget Consumption",
            "type": "graph",
            "targets": [
                {
                    "expr": 'privacy_budget_epsilon_utilization_pct{job="ledgerlens"}',
                    "legendFormat": "Epsilon - {{tenant}}",
                    "refId": "A",
                },
                {
                    "expr": 'privacy_budget_delta_utilization_pct{job="ledgerlens"}',
                    "legendFormat": "Delta - {{tenant}}",
                    "refId": "B",
                },
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 80},
                            {"color": "red", "value": 95},
                        ],
                    },
                },
            },
            "alert": {
                "conditions": [
                    {
                        "evaluator": {"params": [95], "type": "gt"},
                        "operator": {"type": "and"},
                        "query": {"params": ["A", "5m", "now"]},
                        "type": "query",
                    },
                ],
                "executionErrorState": "alerting",
                "frequency": "5m",
                "handler": 1,
                "message": "Privacy budget critical threshold exceeded",
                "name": "Privacy Budget Critical",
                "noDataState": "no_data",
                "notifications": [{"uid": "privacy-alerts"}],
            },
        }
        return panel
