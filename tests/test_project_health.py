"""Unit tests for dashboard-ready project health summary (Issue #605)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoring.project_health import (
    SCHEMA_VERSION,
    ComponentHealth,
    ProjectHealthContractError,
    assert_valid_project_health_summary,
    build_project_health_summary,
    collect_adversarial_posture,
    collect_data_contracts,
    collect_model_artifacts,
    collect_training_metrics,
    load_project_health_summary,
    validate_project_health_summary,
    worst_status,
    write_project_health_summary,
)
from scripts.project_health_summary import main as cli_main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_worst_status_rollups():
    assert worst_status([]) == "unknown"
    assert worst_status(["healthy"]) == "healthy"
    assert worst_status(["healthy", "degraded"]) == "degraded"
    assert worst_status(["unknown", "critical", "healthy"]) == "critical"


def test_validate_rejects_missing_fields():
    errors = validate_project_health_summary({"schema_version": SCHEMA_VERSION})
    assert any("missing required field" in e for e in errors)


def test_validate_rejects_bad_status_and_score():
    errors = validate_project_health_summary(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": "2026-01-01T00:00:00Z",
            "overall_status": "fine",
            "components": [
                {
                    "id": "x",
                    "name": "X",
                    "status": "broken",
                    "score": 150,
                    "metrics": {},
                    "diagnostics": [],
                }
            ],
            "signals": {},
            "diagnostics": [],
            "thresholds": {},
        }
    )
    assert any("overall_status" in e for e in errors)
    assert any("status invalid" in e for e in errors)
    assert any("score out of range" in e for e in errors)


def test_assert_valid_raises():
    with pytest.raises(ProjectHealthContractError):
        assert_valid_project_health_summary({})


def test_build_summary_against_real_repo_is_contract_valid():
    summary = build_project_health_summary(REPO_ROOT)
    payload = summary.to_dict()
    assert validate_project_health_summary(payload) == []
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["overall_status"] in {"healthy", "degraded", "critical", "unknown"}
    assert len(payload["components"]) >= 5
    ids = {c["id"] for c in payload["components"]}
    assert "model_artifacts" in ids
    assert "data_contracts" in ids
    assert "pipeline_entrypoints" in ids
    assert "signals" in payload
    assert "thresholds" in payload
    assert payload["diagnostics"]


def test_model_artifacts_critical_when_dir_missing(tmp_path: Path):
    component = collect_model_artifacts(tmp_path)
    assert component.status == "critical"
    assert component.score == 0.0
    assert any("missing" in d.lower() or "MODEL_DIR" in d for d in component.diagnostics)


def test_training_metrics_invalid_json(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "metrics.json").write_text("{not-json", encoding="utf-8")
    component = collect_training_metrics(tmp_path, {})
    assert component.status == "critical"
    assert "invalid JSON" in " ".join(component.diagnostics)


def test_training_metrics_flags_high_degradation(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "metrics.json").write_text(
        json.dumps(
            {
                "differential_privacy": {
                    "demo_model": {
                        "auc_roc_degradation": 0.5,
                        "membership_inference_success_rate": 0.9,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    component = collect_training_metrics(
        tmp_path,
        {"max_auc_degradation": 0.05, "max_mia_success_rate": 0.65},
    )
    assert component.status in {"degraded", "critical"}
    assert component.score < 100
    assert any("AUC degradation" in d for d in component.diagnostics)
    assert any("membership-inference" in d for d in component.diagnostics)


def test_data_contracts_degraded_when_missing(tmp_path: Path):
    component = collect_data_contracts(tmp_path)
    assert component.status in {"degraded", "critical"}
    assert component.metrics["required_present"] == 0


def test_adversarial_posture_threshold(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "adversarial_benchmark.json").write_text(
        json.dumps({"evasion_rate": 0.9, "gradient_attack": {"evasion_rate": 0.1}}),
        encoding="utf-8",
    )
    component = collect_adversarial_posture(
        tmp_path, {"max_adversarial_evasion_rate": 0.25}
    )
    assert component.status in {"degraded", "critical"}
    assert any("evasion_rate" in d for d in component.diagnostics)


def test_write_and_load_roundtrip(tmp_path: Path):
    out = tmp_path / "out" / "project_health.json"
    summary = write_project_health_summary(out, repo_root=REPO_ROOT)
    loaded = load_project_health_summary(out)
    assert loaded["schema_version"] == summary.schema_version
    assert loaded["overall_status"] == summary.overall_status
    assert len(loaded["components"]) == len(summary.components)


def test_custom_collectors_injected(tmp_path: Path):
    def fake_collector(root: Path, thresholds: dict) -> ComponentHealth:
        return ComponentHealth(
            id="custom",
            name="Custom",
            status="healthy",
            score=88.0,
            metrics={"root": str(root)},
            diagnostics=[],
            checked_at="2026-01-01T00:00:00Z",
        )

    summary = build_project_health_summary(
        tmp_path,
        collectors=[fake_collector],
    )
    assert summary.overall_status == "healthy"
    assert summary.components[0].id == "custom"
    assert summary.components[0].score == 88.0


def test_cli_writes_output_and_exit_zero(tmp_path: Path, capsys):
    out = tmp_path / "health.json"
    code = cli_main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(out),
            "--quiet",
            "--fail-on",
            "never",
        ]
    )
    assert code == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert validate_project_health_summary(payload) == []


def test_cli_fail_on_critical_with_forced_collector(tmp_path: Path):
    def critical_collector(root: Path, thresholds: dict) -> ComponentHealth:
        return ComponentHealth(
            id="boom",
            name="Boom",
            status="critical",
            score=0.0,
            metrics={},
            diagnostics=["forced failure for gate test"],
            checked_at="2026-01-01T00:00:00Z",
        )

    # Build with custom collector and write via API, then use CLI fail-on against real repo is hard.
    # Instead assert STATUS gate helper path through building a critical summary and CLI on empty root.
    summary = build_project_health_summary(tmp_path, collectors=[critical_collector])
    assert summary.overall_status == "critical"

    out = tmp_path / "critical.json"
    write_project_health_summary(out, summary=summary)
    # CLI rebuilds from repo_root collectors, so for empty root it should be critical/degraded
    code = cli_main(
        [
            "--repo-root",
            str(tmp_path),
            "--quiet",
            "--fail-on",
            "critical",
        ]
    )
    assert code == 1
