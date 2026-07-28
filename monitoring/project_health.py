"""Dashboard-ready project health summary (Issue #605).

This module defines a **stable contract** for project-health data that dashboards,
CLI tools, and CI gates can consume without depending on ad-hoc file layouts.

Contract (schema_version ``1.0``)
---------------------------------
A summary document always contains:

- ``schema_version`` — contract version string
- ``generated_at`` — UTC ISO-8601 timestamp
- ``overall_status`` — aggregate status: ``healthy`` | ``degraded`` | ``critical`` | ``unknown``
- ``components`` — list of component health records (see ``ComponentHealth``)
- ``signals`` — machine-readable rollup buckets for dashboards
- ``diagnostics`` — actionable maintainer messages (highest severity first)
- ``thresholds`` — thresholds used for status evaluation (for transparency)

Design notes
------------
- Collectors are pure / filesystem-based so unit tests and offline CI work without
  Redis, Kafka, or a live DB.
- Status roll-up is worst-of: any ``critical`` component makes overall ``critical``.
- Missing optional artifacts degrade with diagnostics rather than hard-failing
  the whole summary (so the document remains dashboard-ready).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
UTC = timezone.utc

VALID_STATUSES = frozenset({"healthy", "degraded", "critical", "unknown"})

STATUS_RANK = {
    "healthy": 0,
    "unknown": 1,
    "degraded": 2,
    "critical": 3,
}

# Default evaluation thresholds (overridable via build_project_health_summary)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_auc_degradation": 0.05,
    "max_mia_success_rate": 0.65,
    "max_adversarial_evasion_rate": 0.25,
    "min_component_score": 50.0,
}


@dataclass(frozen=True)
class ComponentHealth:
    """Health record for a single project subsystem."""

    id: str
    name: str
    status: str
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectHealthSummary:
    """Dashboard-ready project health document (schema 1.0)."""

    schema_version: str
    generated_at: str
    overall_status: str
    components: list[ComponentHealth]
    signals: dict[str, Any]
    diagnostics: list[str]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "components": [c.to_dict() for c in self.components],
            "signals": self.signals,
            "diagnostics": list(self.diagnostics),
            "thresholds": dict(self.thresholds),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


class ProjectHealthContractError(ValueError):
    """Raised when a summary document fails contract validation."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, float(score)))


def worst_status(statuses: Iterable[str]) -> str:
    """Return the worst status among *statuses* (unknown if empty)."""
    worst = "unknown"
    worst_rank = -1
    for status in statuses:
        if status not in STATUS_RANK:
            continue
        rank = STATUS_RANK[status]
        if rank > worst_rank:
            worst = status
            worst_rank = rank
    return worst if worst_rank >= 0 else "unknown"


def validate_project_health_summary(document: Mapping[str, Any]) -> list[str]:
    """Validate a summary dict against the schema 1.0 contract.

    Returns a list of human-readable errors (empty list means valid).
    Does not raise — callers that need hard failure should use
    ``assert_valid_project_health_summary``.
    """
    errors: list[str] = []

    if not isinstance(document, Mapping):
        return ["summary must be a mapping/object"]

    for key in (
        "schema_version",
        "generated_at",
        "overall_status",
        "components",
        "signals",
        "diagnostics",
        "thresholds",
    ):
        if key not in document:
            errors.append(f"missing required field: {key}")

    schema = document.get("schema_version")
    if schema is not None and schema != SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {schema!r}; expected {SCHEMA_VERSION!r}"
        )

    overall = document.get("overall_status")
    if overall is not None and overall not in VALID_STATUSES:
        errors.append(
            f"invalid overall_status {overall!r}; "
            f"must be one of {sorted(VALID_STATUSES)}"
        )

    components = document.get("components")
    if components is not None:
        if not isinstance(components, list):
            errors.append("components must be a list")
        else:
            seen_ids: set[str] = set()
            for i, component in enumerate(components):
                prefix = f"components[{i}]"
                if not isinstance(component, Mapping):
                    errors.append(f"{prefix} must be an object")
                    continue
                for field_name in ("id", "name", "status", "score", "metrics", "diagnostics"):
                    if field_name not in component:
                        errors.append(f"{prefix} missing field: {field_name}")
                cid = component.get("id")
                if isinstance(cid, str):
                    if cid in seen_ids:
                        errors.append(f"duplicate component id: {cid}")
                    seen_ids.add(cid)
                status = component.get("status")
                if status is not None and status not in VALID_STATUSES:
                    errors.append(f"{prefix}.status invalid: {status!r}")
                score = component.get("score")
                if score is not None:
                    try:
                        score_f = float(score)
                        if score_f < 0 or score_f > 100:
                            errors.append(
                                f"{prefix}.score out of range [0, 100]: {score_f}"
                            )
                    except (TypeError, ValueError):
                        errors.append(f"{prefix}.score must be numeric")

    if "signals" in document and not isinstance(document.get("signals"), Mapping):
        errors.append("signals must be an object")

    if "diagnostics" in document and not isinstance(document.get("diagnostics"), list):
        errors.append("diagnostics must be a list")

    if "thresholds" in document and not isinstance(document.get("thresholds"), Mapping):
        errors.append("thresholds must be an object")

    return errors


def assert_valid_project_health_summary(document: Mapping[str, Any]) -> None:
    """Raise ``ProjectHealthContractError`` if *document* is not contract-valid."""
    errors = validate_project_health_summary(document)
    if errors:
        raise ProjectHealthContractError(
            "project health summary failed contract validation:\n- "
            + "\n- ".join(errors)
        )


# ---------------------------------------------------------------------------
# Component collectors
# ---------------------------------------------------------------------------


def _component(
    *,
    id: str,
    name: str,
    status: str,
    score: float,
    metrics: dict[str, Any] | None = None,
    diagnostics: list[str] | None = None,
    checked_at: str | None = None,
) -> ComponentHealth:
    if status not in VALID_STATUSES:
        status = "unknown"
    return ComponentHealth(
        id=id,
        name=name,
        status=status,
        score=_clamp_score(score),
        metrics=metrics or {},
        diagnostics=list(diagnostics or []),
        checked_at=checked_at or _utc_now(),
    )


def collect_model_artifacts(repo_root: Path) -> ComponentHealth:
    """Check that model artifact directory and baseline files are present."""
    model_dir = repo_root / "models"
    required = [
        model_dir / "metrics.json",
        model_dir / "label_distribution_baseline.json",
        model_dir / "README.md",
    ]
    present = [p for p in required if p.is_file()]
    missing = [str(p.relative_to(repo_root)) for p in required if not p.is_file()]
    has_weights = any(model_dir.glob("*.pt")) or any(model_dir.glob("*.joblib"))

    score = 100.0
    status = "healthy"
    diagnostics: list[str] = []

    if not model_dir.is_dir():
        return _component(
            id="model_artifacts",
            name="Model artifacts",
            status="critical",
            score=0.0,
            metrics={"model_dir": str(model_dir), "exists": False},
            diagnostics=[
                f"MODEL_DIR missing at {model_dir}: train or restore models before scoring."
            ],
        )

    if missing:
        score -= 25.0 * len(missing)
        status = "degraded"
        diagnostics.append(
            "Missing model support files: " + ", ".join(missing) + ". Restore from release artifacts."
        )
    if not has_weights:
        score -= 20.0
        status = worst_status([status, "degraded"])
        diagnostics.append(
            "No model weight files (*.pt / *.joblib) under models/; inference may be unavailable."
        )

    if score < 50:
        status = "critical"

    return _component(
        id="model_artifacts",
        name="Model artifacts",
        status=status,
        score=score,
        metrics={
            "model_dir": str(model_dir),
            "required_present": len(present),
            "required_total": len(required),
            "missing": missing,
            "has_weight_files": has_weights,
        },
        diagnostics=diagnostics,
    )


def collect_training_metrics(
    repo_root: Path, thresholds: Mapping[str, float]
) -> ComponentHealth:
    """Evaluate privacy / quality metrics from models/metrics.json."""
    metrics_path = repo_root / "models" / "metrics.json"
    if not metrics_path.is_file():
        return _component(
            id="training_metrics",
            name="Training metrics",
            status="degraded",
            score=40.0,
            metrics={"path": str(metrics_path), "exists": False},
            diagnostics=[
                f"Missing {metrics_path.relative_to(repo_root)}; "
                "re-run training/publish scripts to emit quality metrics."
            ],
        )

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _component(
            id="training_metrics",
            name="Training metrics",
            status="critical",
            score=10.0,
            metrics={"path": str(metrics_path), "parse_error": str(exc)},
            diagnostics=[
                f"models/metrics.json is invalid JSON ({exc}); fix or regenerate the file."
            ],
        )

    max_deg = float(thresholds.get("max_auc_degradation", DEFAULT_THRESHOLDS["max_auc_degradation"]))
    max_mia = float(
        thresholds.get("max_mia_success_rate", DEFAULT_THRESHOLDS["max_mia_success_rate"])
    )

    diagnostics: list[str] = []
    status = "healthy"
    score = 100.0
    model_rows: list[dict[str, Any]] = []

    dp = payload.get("differential_privacy")
    if isinstance(dp, Mapping) and dp:
        for model_name, row in dp.items():
            if not isinstance(row, Mapping):
                continue
            degradation = float(row.get("auc_roc_degradation", 0.0) or 0.0)
            mia = float(row.get("membership_inference_success_rate", 0.0) or 0.0)
            model_rows.append(
                {
                    "model": model_name,
                    "auc_roc_degradation": degradation,
                    "membership_inference_success_rate": mia,
                    "achieved_epsilon": row.get("achieved_epsilon"),
                }
            )
            if degradation > max_deg:
                status = worst_status([status, "degraded"])
                score -= 15.0
                diagnostics.append(
                    f"{model_name}: AUC degradation {degradation:.4f} exceeds "
                    f"threshold {max_deg:.4f}."
                )
            if mia > max_mia:
                status = worst_status([status, "degraded"])
                score -= 15.0
                diagnostics.append(
                    f"{model_name}: membership-inference success rate {mia:.3f} exceeds "
                    f"threshold {max_mia:.3f}; review DP settings."
                )
    else:
        status = "unknown"
        score = 55.0
        diagnostics.append(
            "models/metrics.json has no differential_privacy block; "
            "privacy quality signals unavailable."
        )

    if score < 50:
        status = worst_status([status, "critical"])

    return _component(
        id="training_metrics",
        name="Training metrics",
        status=status,
        score=score,
        metrics={
            "path": str(metrics_path.relative_to(repo_root)),
            "models": model_rows,
            "max_auc_degradation_threshold": max_deg,
            "max_mia_success_rate_threshold": max_mia,
        },
        diagnostics=diagnostics,
    )


def collect_data_contracts(repo_root: Path) -> ComponentHealth:
    """Validate presence of core data contracts and feature dictionaries."""
    required = [
        repo_root / "data" / "feature_ranges.json",
        repo_root / "data" / "feature_dictionary.md",
        repo_root / "data" / "trade_avro_schema.json",
        repo_root / "data" / "dataset_card.md",
    ]
    present = [p for p in required if p.is_file()]
    missing = [str(p.relative_to(repo_root)) for p in required if not p.is_file()]

    # Light schema sanity for feature_ranges.json
    parse_errors: list[str] = []
    feature_count = 0
    ranges_path = repo_root / "data" / "feature_ranges.json"
    if ranges_path.is_file():
        try:
            ranges = json.loads(ranges_path.read_text(encoding="utf-8"))
            if isinstance(ranges, Mapping):
                feature_count = len(ranges)
            else:
                parse_errors.append("feature_ranges.json root must be an object")
        except json.JSONDecodeError as exc:
            parse_errors.append(f"feature_ranges.json invalid JSON: {exc}")

    score = 100.0 * (len(present) / max(len(required), 1))
    status = "healthy"
    diagnostics: list[str] = []

    if missing:
        status = "degraded"
        score -= 10.0 * len(missing)
        diagnostics.append("Missing data contracts: " + ", ".join(missing))
    if parse_errors:
        status = worst_status([status, "critical"])
        score = min(score, 25.0)
        diagnostics.extend(parse_errors)
    if feature_count == 0 and ranges_path.is_file() and not parse_errors:
        status = worst_status([status, "degraded"])
        score = min(score, 60.0)
        diagnostics.append(
            "feature_ranges.json is empty; run scripts/compute_feature_ranges.py."
        )

    return _component(
        id="data_contracts",
        name="Data contracts",
        status=status,
        score=score,
        metrics={
            "required_present": len(present),
            "required_total": len(required),
            "missing": missing,
            "feature_range_keys": feature_count,
        },
        diagnostics=diagnostics,
    )


def collect_adversarial_posture(
    repo_root: Path, thresholds: Mapping[str, float]
) -> ComponentHealth:
    """Read adversarial benchmark report if present."""
    path = repo_root / "reports" / "adversarial_benchmark.json"
    max_evasion = float(
        thresholds.get(
            "max_adversarial_evasion_rate",
            DEFAULT_THRESHOLDS["max_adversarial_evasion_rate"],
        )
    )

    if not path.is_file():
        return _component(
            id="adversarial_posture",
            name="Adversarial posture",
            status="unknown",
            score=50.0,
            metrics={"path": str(path.relative_to(repo_root)), "exists": False},
            diagnostics=[
                "reports/adversarial_benchmark.json missing; "
                "run scripts/run_adversarial_eval.py to populate."
            ],
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _component(
            id="adversarial_posture",
            name="Adversarial posture",
            status="critical",
            score=15.0,
            metrics={"parse_error": str(exc)},
            diagnostics=[f"adversarial_benchmark.json is invalid JSON: {exc}"],
        )

    evasion = float(payload.get("evasion_rate", 0.0) or 0.0)
    gradient = payload.get("gradient_attack") if isinstance(payload.get("gradient_attack"), Mapping) else {}
    gradient_evasion = float(gradient.get("evasion_rate", 0.0) or 0.0) if gradient else 0.0

    status = "healthy"
    score = 100.0
    diagnostics: list[str] = []

    if evasion > max_evasion:
        status = "degraded"
        score -= 30.0
        diagnostics.append(
            f"Adversarial evasion_rate {evasion:.3f} exceeds threshold {max_evasion:.3f}."
        )
    if gradient_evasion > max_evasion:
        status = worst_status([status, "degraded"])
        score -= 20.0
        diagnostics.append(
            f"Gradient attack evasion_rate {gradient_evasion:.3f} exceeds "
            f"threshold {max_evasion:.3f}."
        )

    if score < 50:
        status = worst_status([status, "critical"])

    return _component(
        id="adversarial_posture",
        name="Adversarial posture",
        status=status,
        score=score,
        metrics={
            "path": str(path.relative_to(repo_root)),
            "evasion_rate": evasion,
            "gradient_attack_evasion_rate": gradient_evasion,
            "max_adversarial_evasion_rate_threshold": max_evasion,
        },
        diagnostics=diagnostics,
    )


def collect_ops_config(repo_root: Path) -> ComponentHealth:
    """Lightweight ops / monitoring surface checks."""
    required = [
        repo_root / "monitoring" / "prometheus.yml",
        repo_root / "monitoring" / "alert_rules.yml",
        repo_root / "docker-compose.yml",
    ]
    dashboards = list((repo_root / "monitoring" / "grafana" / "dashboards").glob("*.json"))
    missing = [str(p.relative_to(repo_root)) for p in required if not p.is_file()]

    score = 100.0
    status = "healthy"
    diagnostics: list[str] = []

    if missing:
        status = "degraded"
        score -= 20.0 * len(missing)
        diagnostics.append("Missing ops config: " + ", ".join(missing))
    if not dashboards:
        status = worst_status([status, "degraded"])
        score -= 15.0
        diagnostics.append(
            "No Grafana dashboards under monitoring/grafana/dashboards/."
        )

    # Validate dashboard JSON lightly
    bad_dashboards: list[str] = []
    for dash in dashboards:
        try:
            data = json.loads(dash.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping) or "panels" not in data:
                bad_dashboards.append(dash.name)
        except json.JSONDecodeError:
            bad_dashboards.append(dash.name)
    if bad_dashboards:
        status = worst_status([status, "critical"])
        score = min(score, 30.0)
        diagnostics.append(
            "Invalid Grafana dashboard JSON: " + ", ".join(sorted(bad_dashboards))
        )

    return _component(
        id="ops_surface",
        name="Ops & dashboard surface",
        status=status,
        score=score,
        metrics={
            "missing_config": missing,
            "dashboard_count": len(dashboards),
            "invalid_dashboards": bad_dashboards,
        },
        diagnostics=diagnostics,
    )


def collect_pipeline_entrypoints(repo_root: Path) -> ComponentHealth:
    """Ensure primary pipeline/API entrypoints remain present."""
    required = [
        repo_root / "run_pipeline.py",
        repo_root / "api" / "app.py",
        repo_root / "scripts" / "stream.py",
        repo_root / "requirements.txt",
    ]
    missing = [str(p.relative_to(repo_root)) for p in required if not p.is_file()]
    if missing:
        return _component(
            id="pipeline_entrypoints",
            name="Pipeline entrypoints",
            status="critical",
            score=max(0.0, 100.0 - 30.0 * len(missing)),
            metrics={"missing": missing},
            diagnostics=[
                "Critical entrypoints missing: "
                + ", ".join(missing)
                + ". Restore before deploying."
            ],
        )
    return _component(
        id="pipeline_entrypoints",
        name="Pipeline entrypoints",
        status="healthy",
        score=100.0,
        metrics={"missing": [], "checked": [str(p.relative_to(repo_root)) for p in required]},
        diagnostics=[],
    )


DEFAULT_COLLECTORS: tuple[Callable[[Path, Mapping[str, float]], ComponentHealth], ...] = (
    lambda root, _t: collect_model_artifacts(root),
    collect_training_metrics,
    lambda root, _t: collect_data_contracts(root),
    collect_adversarial_posture,
    lambda root, _t: collect_ops_config(root),
    lambda root, _t: collect_pipeline_entrypoints(root),
)


def _build_signals(components: list[ComponentHealth]) -> dict[str, Any]:
    by_id = {c.id: c for c in components}
    return {
        "model": {
            "status": by_id.get("model_artifacts", _unknown("model_artifacts")).status,
            "score": by_id.get("model_artifacts", _unknown("model_artifacts")).score,
            "training_status": by_id.get(
                "training_metrics", _unknown("training_metrics")
            ).status,
        },
        "data_quality": {
            "status": by_id.get("data_contracts", _unknown("data_contracts")).status,
            "score": by_id.get("data_contracts", _unknown("data_contracts")).score,
        },
        "security": {
            "status": by_id.get(
                "adversarial_posture", _unknown("adversarial_posture")
            ).status,
            "score": by_id.get(
                "adversarial_posture", _unknown("adversarial_posture")
            ).score,
        },
        "ops": {
            "status": by_id.get("ops_surface", _unknown("ops_surface")).status,
            "score": by_id.get("ops_surface", _unknown("ops_surface")).score,
            "pipeline_status": by_id.get(
                "pipeline_entrypoints", _unknown("pipeline_entrypoints")
            ).status,
        },
        "component_count": len(components),
        "healthy_count": sum(1 for c in components if c.status == "healthy"),
        "degraded_count": sum(1 for c in components if c.status == "degraded"),
        "critical_count": sum(1 for c in components if c.status == "critical"),
        "unknown_count": sum(1 for c in components if c.status == "unknown"),
    }


def _unknown(cid: str) -> ComponentHealth:
    return _component(
        id=cid,
        name=cid,
        status="unknown",
        score=0.0,
        diagnostics=[f"Component {cid} was not evaluated."],
    )


def build_project_health_summary(
    repo_root: str | Path | None = None,
    *,
    thresholds: Mapping[str, float] | None = None,
    collectors: Iterable[Callable[[Path, Mapping[str, float]], ComponentHealth]]
    | None = None,
) -> ProjectHealthSummary:
    """Build a contract-valid project health summary for *repo_root*.

    Args:
        repo_root: Repository root (defaults to package parent of ``monitoring/``).
        thresholds: Optional overrides for evaluation thresholds.
        collectors: Optional custom collectors for tests or extensions.

    Returns:
        A validated ``ProjectHealthSummary``.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    root = root.resolve()

    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            merged_thresholds[str(key)] = float(value)

    active_collectors = list(collectors) if collectors is not None else list(DEFAULT_COLLECTORS)
    components: list[ComponentHealth] = []
    for collector in active_collectors:
        try:
            components.append(collector(root, merged_thresholds))
        except Exception as exc:  # pragma: no cover - defensive for flaky FS
            logger.exception("project health collector failed: %s", collector)
            components.append(
                _component(
                    id=getattr(collector, "__name__", "collector"),
                    name=getattr(collector, "__name__", "collector"),
                    status="critical",
                    score=0.0,
                    diagnostics=[f"Collector crashed: {exc}"],
                )
            )

    overall = worst_status(c.status for c in components)
    diagnostics: list[str] = []
    # Highest severity first
    for severity in ("critical", "degraded", "unknown", "healthy"):
        for component in components:
            if component.status != severity:
                continue
            for message in component.diagnostics:
                diagnostics.append(f"[{component.id}/{severity}] {message}")

    if not diagnostics and overall == "healthy":
        diagnostics.append("All project-health components are healthy.")

    summary = ProjectHealthSummary(
        schema_version=SCHEMA_VERSION,
        generated_at=_utc_now(),
        overall_status=overall,
        components=components,
        signals=_build_signals(components),
        diagnostics=diagnostics,
        thresholds=merged_thresholds,
    )
    assert_valid_project_health_summary(summary.to_dict())
    return summary


def write_project_health_summary(
    path: str | Path,
    summary: ProjectHealthSummary | None = None,
    *,
    repo_root: str | Path | None = None,
    indent: int | None = 2,
) -> ProjectHealthSummary:
    """Build (if needed) and write a summary JSON document to *path*."""
    document = summary or build_project_health_summary(repo_root=repo_root)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document.to_json(indent=indent) + "\n", encoding="utf-8")
    return document


def load_project_health_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate a summary JSON file; return the dict."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_valid_project_health_summary(payload)
    return payload
