"""Automated post-deployment rollback trigger — Issue #940.

Monitors key production metrics in a configurable post-deployment window.
If any metric regresses beyond a documented threshold the trigger automatically
rolls back to the last known-good model version via
:class:`~detection.artifact_lifecycle.ModelArtifactRegistry`.

Every automated rollback (and every suppressed rollback) is written to a
structured audit log so the event is fully reconstructable.

Typical usage::

    from detection.rollback_trigger import RollbackConfig, PostDeploymentMonitor

    config = RollbackConfig(
        window_seconds=3600,         # monitor for 1 hour post-deploy
        thresholds={"auc_roc": 0.02, "f1": 0.03},  # max allowed drop
    )
    monitor = PostDeploymentMonitor(registry=registry, config=config)

    # On every metric sample collected after deployment:
    monitor.record_baseline("rf", version, {"auc_roc": 0.94, "f1": 0.89})
    result = monitor.evaluate("rf", version, {"auc_roc": 0.91, "f1": 0.85})
    if result.rolled_back:
        print("Rolled back to", result.previous_active_version)

Manual override::

    monitor.set_override("rf", version, suppress=True, reason="Expected degradation on holiday data")
    result = monitor.evaluate("rf", version, {"auc_roc": 0.91, "f1": 0.85})
    # result.rolled_back is False, result.suppressed is True

Audit log entries are JSONL records appended to
``PostDeploymentMonitor.audit_log_path`` (default: ``reports/rollback_audit.jsonl``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (absolute metric-value drop that triggers rollback)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "auc_roc": 0.02,  # max allowed drop in AUC-ROC
    "f1": 0.03,  # max allowed drop in F1
    "precision": 0.03,
    "recall": 0.03,
}

DEFAULT_WINDOW_SECONDS: int = 3600  # 1 hour post-deployment monitoring window


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RollbackConfig:
    """Configuration for the post-deployment rollback trigger.

    Attributes:
        window_seconds: Duration of the post-deployment monitoring window.
            Metric regression checks are only acted upon within this window.
        thresholds: Mapping of metric name → maximum allowed absolute drop.
            A metric that drops by more than its threshold triggers rollback.
        audit_log_path: Path to the JSONL audit log file. Defaults to
            ``reports/rollback_audit.jsonl``.
    """

    window_seconds: int = DEFAULT_WINDOW_SECONDS
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    audit_log_path: str = "reports/rollback_audit.jsonl"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionDetail:
    """A single metric that exceeded its regression threshold."""

    metric: str
    baseline_value: float
    observed_value: float
    drop: float
    threshold: float


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of a single post-deployment metric evaluation."""

    model_name: str
    version: str
    evaluated_at: str
    within_window: bool
    regressions: tuple[RegressionDetail, ...]
    rolled_back: bool
    suppressed: bool
    suppression_reason: str | None
    previous_active_version: str | None


# ---------------------------------------------------------------------------
# Override record
# ---------------------------------------------------------------------------


@dataclass
class _OverrideRecord:
    suppress: bool
    reason: str | None
    set_at: float


# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------


def _append_audit_entry(log_path: str, entry: dict[str, Any]) -> None:
    """Append a JSONL audit entry atomically (append mode is atomic on POSIX)."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------


class PostDeploymentMonitor:
    """Monitors post-deployment metrics and triggers automated rollbacks.

    Parameters
    ----------
    registry:
        A :class:`~detection.artifact_lifecycle.ModelArtifactRegistry`
        instance used to execute rollbacks. The registry must have a
        ``trust_verifier`` configured if ``promote()`` will be called later;
        for rollback-only usage the verifier is not required.
    config:
        A :class:`RollbackConfig` instance. Defaults are used when omitted.
    """

    def __init__(
        self,
        registry: Any,  # ModelArtifactRegistry — typed as Any to avoid a circular import
        config: RollbackConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or RollbackConfig()
        self._baselines: dict[tuple[str, str], dict[str, float]] = {}
        self._deploy_times: dict[tuple[str, str], float] = {}
        self._overrides: dict[tuple[str, str], _OverrideRecord] = {}

    @property
    def audit_log_path(self) -> str:
        return self._config.audit_log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_baseline(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        deploy_time: float | None = None,
    ) -> None:
        """Record the baseline metrics for a freshly promoted model version.

        Call this immediately after promotion, before the first evaluation.

        Parameters
        ----------
        model_name:
            Artifact name in the registry (e.g. ``"rf"``).
        version:
            Version string returned by :meth:`~ModelArtifactRegistry.register`.
        metrics:
            Dict of metric-name → float value at deployment time.
        deploy_time:
            Unix timestamp of deployment (defaults to ``time.time()``).
        """
        key = (model_name, version)
        self._baselines[key] = {k: float(v) for k, v in metrics.items()}
        self._deploy_times[key] = deploy_time if deploy_time is not None else time.time()
        logger.info(
            "Rollback monitor: baseline recorded for %s:%s — %s",
            model_name,
            version,
            metrics,
        )

    def set_override(
        self,
        model_name: str,
        version: str,
        *,
        suppress: bool,
        reason: str | None = None,
    ) -> None:
        """Set or clear a manual override for automated rollback.

        When ``suppress=True`` the automated trigger will not roll back this
        version even if metric regression is detected.  This is intended for
        cases where a metric change is expected/intentional (e.g. model
        re-calibrated for a seasonal traffic shift).

        Parameters
        ----------
        suppress:
            ``True`` to suppress automated rollback; ``False`` to re-enable it.
        reason:
            Human-readable explanation logged in the audit trail.
        """
        key = (model_name, version)
        self._overrides[key] = _OverrideRecord(
            suppress=suppress,
            reason=reason,
            set_at=time.time(),
        )
        logger.info(
            "Rollback monitor: override set for %s:%s suppress=%s reason=%r",
            model_name,
            version,
            suppress,
            reason,
        )
        _append_audit_entry(
            self._config.audit_log_path,
            {
                "event": "override_set",
                "model_name": model_name,
                "version": version,
                "suppress": suppress,
                "reason": reason,
                "set_at": datetime.now(UTC).isoformat(),
            },
        )

    def evaluate(
        self,
        model_name: str,
        version: str,
        observed_metrics: dict[str, float],
        *,
        evaluation_time: float | None = None,
    ) -> EvaluationResult:
        """Evaluate observed metrics against the baseline and trigger rollback if needed.

        Parameters
        ----------
        model_name:
            Artifact name in the registry.
        version:
            Version string to evaluate.
        observed_metrics:
            Current production metrics (e.g. from a canary evaluation batch).
        evaluation_time:
            Unix timestamp of the observation (defaults to ``time.time()``).

        Returns
        -------
        EvaluationResult
            A frozen record describing the outcome. Inspect ``.rolled_back``
            and ``.regressions`` to understand what happened.
        """
        eval_time = evaluation_time if evaluation_time is not None else time.time()
        key = (model_name, version)

        # Check monitoring window
        deploy_time = self._deploy_times.get(key)
        within_window = (
            deploy_time is not None
            and (eval_time - deploy_time) <= self._config.window_seconds
        )

        # Detect regressions against baseline
        baseline = self._baselines.get(key, {})
        regressions: list[RegressionDetail] = []
        for metric, threshold in self._config.thresholds.items():
            if metric not in baseline or metric not in observed_metrics:
                continue
            drop = baseline[metric] - float(observed_metrics[metric])
            if drop > threshold:
                regressions.append(
                    RegressionDetail(
                        metric=metric,
                        baseline_value=baseline[metric],
                        observed_value=float(observed_metrics[metric]),
                        drop=drop,
                        threshold=threshold,
                    )
                )

        # Check override
        override = self._overrides.get(key)
        suppressed = bool(override and override.suppress)
        suppression_reason = override.reason if suppressed else None

        # Decide whether to roll back
        should_rollback = bool(regressions) and within_window and not suppressed
        rolled_back = False
        previous_active_version: str | None = None

        if should_rollback:
            try:
                rolled = self._registry.rollback(
                    model_name,
                    version,
                    reason=f"Automated rollback: metric regression detected — "
                    f"{[r.metric for r in regressions]}",
                )
                rolled_back = True
                if rolled.parent_version:
                    previous_active_version = rolled.parent_version
                logger.warning(
                    "Rollback monitor: AUTO-ROLLBACK executed for %s:%s — regressions: %s",
                    model_name,
                    version,
                    regressions,
                )
            except Exception as exc:
                logger.error(
                    "Rollback monitor: rollback attempted but failed for %s:%s — %s",
                    model_name,
                    version,
                    exc,
                )

        elif regressions and not within_window:
            logger.info(
                "Rollback monitor: regression detected for %s:%s but outside monitoring window"
                " (%ds); no action taken.",
                model_name,
                version,
                self._config.window_seconds,
            )
        elif regressions and suppressed:
            logger.info(
                "Rollback monitor: regression detected for %s:%s but override suppresses"
                " rollback — reason: %r",
                model_name,
                version,
                suppression_reason,
            )

        result = EvaluationResult(
            model_name=model_name,
            version=version,
            evaluated_at=datetime.fromtimestamp(eval_time, tz=UTC).isoformat(),
            within_window=within_window,
            regressions=tuple(regressions),
            rolled_back=rolled_back,
            suppressed=suppressed,
            suppression_reason=suppression_reason,
            previous_active_version=previous_active_version,
        )

        # Always write an audit entry
        self._write_audit_entry(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_audit_entry(self, result: EvaluationResult) -> None:
        entry: dict[str, Any] = {
            "event": "rollback_triggered" if result.rolled_back else "evaluation",
            "model_name": result.model_name,
            "version": result.version,
            "evaluated_at": result.evaluated_at,
            "within_window": result.within_window,
            "regressions": [asdict(r) for r in result.regressions],
            "rolled_back": result.rolled_back,
            "suppressed": result.suppressed,
            "suppression_reason": result.suppression_reason,
            "previous_active_version": result.previous_active_version,
        }
        _append_audit_entry(self._config.audit_log_path, entry)
