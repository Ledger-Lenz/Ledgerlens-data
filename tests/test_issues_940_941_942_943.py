"""Tests for issues #940, #941, #942, #943.

#940 — Automated post-deployment rollback trigger
    - Simulated regression triggers automated rollback within the monitoring window
    - Manual override suppresses an unwanted automated rollback
    - Rollback event fully reconstructable from the audit log

#941 — Experiment lineage tracking with CI promotion check
    - Lineage lookup end-to-end for a test model version
    - Promotion of a model with incomplete lineage is blocked
    - lookup_lineage raises KeyError for unknown run_id

#942 — FATF jurisdiction risk-code mapping validation
    - Export using a deliberately outdated mapping version is flagged/blocked
    - Export using the current mapping version succeeds without warning
    - Soft-warn mode (FATF_MAPPING_VERSION_BLOCK=0) does not raise

#943 — Tamper-evident audit summary with cryptographic signing
    - Signature verification succeeds on an untampered export
    - Modified document is rejected by verify_summary_signature
    - Key rotation: new key pair generates a verifiable signature
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Issue #940 — Rollback trigger
# ---------------------------------------------------------------------------


@pytest.fixture()
def artifact_file(tmp_path: Path) -> str:
    p = tmp_path / "rf.joblib"
    p.write_bytes(b"fake-model-bytes")
    return str(p)


@pytest.fixture()
def registry(tmp_path: Path):
    from detection.artifact_lifecycle import ModelArtifactRegistry

    return ModelArtifactRegistry(
        manifest_path=str(tmp_path / "manifest.json"),
        trust_verifier=lambda record: None,
    )


@pytest.fixture()
def monitor(tmp_path: Path, registry):
    from detection.rollback_trigger import PostDeploymentMonitor, RollbackConfig

    cfg = RollbackConfig(
        window_seconds=3600,
        thresholds={"auc_roc": 0.02, "f1": 0.03},
        audit_log_path=str(tmp_path / "rollback_audit.jsonl"),
    )
    return PostDeploymentMonitor(registry=registry, config=cfg)


class TestRollbackTrigger:
    """Issue #940 — automated rollback trigger."""

    def _promote_version(self, registry, artifact_file: str) -> str:
        version = registry.register("rf", artifact_file, metrics={"auc_roc": 0.94, "f1": 0.89})
        registry.validate("rf", version)
        registry.promote("rf", version)
        return version

    def test_regression_triggers_rollback_within_window(
        self, registry, monitor, artifact_file, tmp_path
    ):
        """Acceptance criterion: regression triggers automated rollback in window."""
        from detection.artifact_lifecycle import ArtifactStage

        # Promote v1 as baseline
        v1 = self._promote_version(registry, artifact_file)

        # Register and promote v2 (the "bad" deployment)
        artifact2 = tmp_path / "rf_v2.joblib"
        artifact2.write_bytes(b"model-v2")
        v2 = registry.register("rf", str(artifact2))
        registry.validate("rf", v2)
        registry.promote("rf", v2)

        baseline = {"auc_roc": 0.94, "f1": 0.89}
        monitor.record_baseline("rf", v2, baseline, deploy_time=time.time())

        # Simulate regression beyond threshold
        observed = {"auc_roc": 0.91, "f1": 0.85}  # drops: 0.03 > 0.02; 0.04 > 0.03
        result = monitor.evaluate("rf", v2, observed)

        assert result.rolled_back is True
        assert len(result.regressions) == 2
        assert result.within_window is True
        # v2 should now be ROLLED_BACK
        assert registry._get("rf", v2).stage == ArtifactStage.ROLLED_BACK
        # v1 should be re-activated
        active = registry.get_active("rf")
        assert active.version == v1

    def test_no_rollback_outside_monitoring_window(
        self, registry, monitor, artifact_file, tmp_path
    ):
        """No rollback when evaluation happens after the monitoring window."""
        artifact2 = tmp_path / "rf_v2.joblib"
        artifact2.write_bytes(b"v2")
        v2 = registry.register("rf", str(artifact2))

        # Set deploy_time in the past (beyond window)
        past = time.time() - 7200  # 2 hours ago, window is 1 hour
        monitor.record_baseline("rf", v2, {"auc_roc": 0.94}, deploy_time=past)

        result = monitor.evaluate("rf", v2, {"auc_roc": 0.90})  # would be a regression
        assert result.within_window is False
        assert result.rolled_back is False

    def test_manual_override_suppresses_rollback(
        self, registry, monitor, artifact_file, tmp_path
    ):
        """Acceptance criterion: manual override suppresses unwanted automated rollback."""
        from detection.artifact_lifecycle import ArtifactStage

        artifact2 = tmp_path / "rf_v2.joblib"
        artifact2.write_bytes(b"v2")
        v2 = registry.register("rf", str(artifact2))
        registry.validate("rf", v2)
        registry.promote("rf", v2)

        monitor.record_baseline("rf", v2, {"auc_roc": 0.94}, deploy_time=time.time())

        # Operator sets override before regression evaluation
        monitor.set_override(
            "rf", v2, suppress=True, reason="Expected degradation on holiday traffic"
        )

        result = monitor.evaluate("rf", v2, {"auc_roc": 0.91})
        assert result.rolled_back is False
        assert result.suppressed is True
        assert "holiday traffic" in (result.suppression_reason or "")
        # v2 should still be PROMOTED (no rollback executed)
        assert registry._get("rf", v2).stage == ArtifactStage.PROMOTED

    def test_audit_log_written_for_every_evaluation(
        self, registry, monitor, artifact_file, tmp_path
    ):
        """Acceptance criterion: rollback event fully reconstructable from audit log."""
        v1 = self._promote_version(registry, artifact_file)
        monitor.record_baseline("rf", v1, {"auc_roc": 0.94}, deploy_time=time.time())

        # Trigger a rollback (no parent, so registry.rollback will fail — that's ok,
        # we just verify the audit log entry is written)
        monitor.evaluate("rf", v1, {"auc_roc": 0.91})

        log_path = Path(monitor.audit_log_path)
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(entries) >= 1
        entry = entries[-1]
        assert entry["model_name"] == "rf"
        assert entry["version"] == v1
        assert "regressions" in entry
        assert "rolled_back" in entry

    def test_audit_log_written_for_override(self, registry, monitor, artifact_file, tmp_path):
        """Override set events are also written to the audit log."""
        v1 = self._promote_version(registry, artifact_file)
        monitor.set_override("rf", v1, suppress=True, reason="deliberate")

        log_path = Path(monitor.audit_log_path)
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        override_entries = [e for e in entries if e["event"] == "override_set"]
        assert len(override_entries) == 1
        assert override_entries[0]["suppress"] is True
        assert override_entries[0]["reason"] == "deliberate"

    def test_no_regression_below_threshold(self, registry, monitor, artifact_file):
        """Small drops below threshold do not trigger rollback."""
        v1 = self._promote_version(registry, artifact_file)
        monitor.record_baseline("rf", v1, {"auc_roc": 0.94}, deploy_time=time.time())

        result = monitor.evaluate("rf", v1, {"auc_roc": 0.93})  # drop 0.01 < threshold 0.02
        assert result.rolled_back is False
        assert len(result.regressions) == 0

    def test_rollback_config_defaults(self):
        """Default thresholds and window are sane."""
        from detection.rollback_trigger import DEFAULT_THRESHOLDS, DEFAULT_WINDOW_SECONDS, RollbackConfig

        cfg = RollbackConfig()
        assert cfg.window_seconds == DEFAULT_WINDOW_SECONDS
        assert "auc_roc" in cfg.thresholds
        assert cfg.thresholds == DEFAULT_THRESHOLDS


# ---------------------------------------------------------------------------
# Issue #941 — Experiment lineage tracking
# ---------------------------------------------------------------------------


class TestExperimentLineage:
    """Issue #941 — lineage linking and CI promotion check."""

    @pytest.fixture()
    def tracker(self, tmp_path: Path):
        from mlops.experiment_tracking import JsonlExperimentTracker

        return JsonlExperimentTracker(path=tmp_path / "experiments.jsonl")

    @pytest.fixture()
    def full_run(self, tracker):
        from mlops.experiment_tracking import ExperimentRun

        run = ExperimentRun(
            name="rf_v1",
            params={"n_estimators": 100},
            feature_schema_hash="abc123",
            dataset_sha256="def456",
            git_sha="deadbeef01234567",
            dataset_snapshot_id="snap_20240601T120000_abc12345",
            resolved_hyperparams={"n_estimators": 100, "max_depth": 10, "random_state": 42},
        )
        record = tracker.log_run(run, metrics={"auc_roc": 0.94, "f1": 0.89})
        return record, run

    def test_run_record_contains_lineage_fields(self, full_run):
        """Acceptance criterion: every run records dataset snapshot, git SHA, hyperparams."""
        record, run = full_run
        assert record["dataset_snapshot_id"] == "snap_20240601T120000_abc12345"
        assert record["git_sha"] == "deadbeef01234567"
        assert record["resolved_hyperparams"] == run.resolved_hyperparams
        assert record["feature_schema_hash"] == "abc123"
        assert record["dataset_sha256"] == "def456"

    def test_lookup_lineage_end_to_end(self, tracker, full_run):
        """Acceptance criterion: lineage lookup demonstrated end-to-end for a test version."""
        from mlops.experiment_tracking import lookup_lineage

        record, run = full_run
        lineage = lookup_lineage(tracker, run_id=record["run_id"])
        assert lineage["run_id"] == record["run_id"]
        assert lineage["dataset_snapshot_id"] == "snap_20240601T120000_abc12345"
        assert lineage["git_sha"] == "deadbeef01234567"
        assert lineage["resolved_hyperparams"]["n_estimators"] == 100

    def test_lookup_lineage_unknown_run_raises_key_error(self, tracker):
        """lookup_lineage raises KeyError for an unknown run_id."""
        from mlops.experiment_tracking import lookup_lineage

        with pytest.raises(KeyError, match="No experiment run found"):
            lookup_lineage(tracker, run_id="nonexistent")

    def test_check_lineage_complete_passes_for_full_record(self, full_run):
        """check_lineage_complete returns empty list for a complete record."""
        from mlops.experiment_tracking import check_lineage_complete

        record, _ = full_run
        errors = check_lineage_complete(record)
        assert errors == []

    def test_check_lineage_complete_blocks_promotion_when_incomplete(self):
        """Acceptance criterion: promotion of a model with incomplete lineage is blocked."""
        from mlops.experiment_tracking import check_lineage_complete

        incomplete_record = {
            "run_id": "abc123",
            "name": "rf",
            "feature_schema_hash": "abc",
            "dataset_sha256": "def",
            "git_sha": None,  # missing
            "dataset_snapshot_id": None,  # missing
            "resolved_hyperparams": None,  # missing
        }
        errors = check_lineage_complete(incomplete_record)
        assert "git_sha" in errors
        assert "dataset_snapshot_id" in errors
        assert "resolved_hyperparams" in errors

    def test_check_lineage_blocks_model_with_no_snapshot(self):
        """A run with no dataset_snapshot_id must be blocked from promotion."""
        from mlops.experiment_tracking import check_lineage_complete

        record = {
            "run_id": "abc",
            "name": "rf",
            "feature_schema_hash": "abc",
            "dataset_sha256": "def",
            "git_sha": "deadbeef",
            "dataset_snapshot_id": None,
            "resolved_hyperparams": {"n_estimators": 100},
        }
        errors = check_lineage_complete(record)
        assert "dataset_snapshot_id" in errors
        # Simulate CI gate
        with pytest.raises(SystemExit):
            if errors:
                raise SystemExit(f"Lineage incomplete — promotion blocked: {errors}")

    def test_run_falls_back_to_params_when_resolved_hyperparams_not_given(self, tracker):
        """resolved_hyperparams defaults to params when not explicitly set."""
        from mlops.experiment_tracking import ExperimentRun

        run = ExperimentRun(
            name="rf",
            params={"n_estimators": 50},
            feature_schema_hash="aaa",
            dataset_sha256="bbb",
            git_sha="abc",
            dataset_snapshot_id="snap_001",
        )
        record = tracker.log_run(run, metrics={"auc_roc": 0.90})
        assert record["resolved_hyperparams"] == {"n_estimators": 50}

    def test_multiple_runs_all_retrievable(self, tracker):
        """All logged runs are retrievable; lookup returns the correct one."""
        from mlops.experiment_tracking import ExperimentRun, lookup_lineage

        runs = []
        for i in range(3):
            run = ExperimentRun(
                name=f"rf_v{i}",
                params={"n_estimators": 100 + i},
                feature_schema_hash=f"hash{i}",
                dataset_sha256=f"sha{i}",
                git_sha=f"git{i}",
                dataset_snapshot_id=f"snap_{i:04d}",
            )
            record = tracker.log_run(run, metrics={"auc_roc": 0.9 + i * 0.01})
            runs.append(record)

        assert len(tracker.list_runs()) == 3
        # Look up the second run specifically
        found = lookup_lineage(tracker, run_id=runs[1]["run_id"])
        assert found["name"] == "rf_v1"
        assert found["dataset_snapshot_id"] == "snap_0001"


# ---------------------------------------------------------------------------
# Issue #942 — FATF mapping version validation
# ---------------------------------------------------------------------------


class TestFATFMappingVersionValidation:
    """Issue #942 — jurisdiction risk-code mapping validation."""

    def test_current_version_passes_validation(self):
        """Acceptance criterion: export using the current mapping version succeeds."""
        from reporting.fatf_risk_codes import MAPPING_VERSION, validate_mapping_version

        # Should not raise or warn when version == minimum
        validate_mapping_version(current_version=MAPPING_VERSION, minimum_version=MAPPING_VERSION)

    def test_outdated_version_raises_in_block_mode(self):
        """Acceptance criterion: export using outdated mapping is blocked (block mode)."""
        from reporting.fatf_risk_codes import MappingVersionError, validate_mapping_version

        with pytest.raises(MappingVersionError) as exc_info:
            validate_mapping_version(
                current_version="2019.06",
                minimum_version="2021.10",
                block_on_stale=True,
            )
        assert "2019.06" in str(exc_info.value)
        assert "2021.10" in str(exc_info.value)

    def test_outdated_version_warns_in_soft_mode(self, caplog):
        """Soft mode emits a warning but does not raise."""
        import logging

        from reporting.fatf_risk_codes import validate_mapping_version

        with caplog.at_level(logging.WARNING, logger="reporting.fatf_risk_codes"):
            validate_mapping_version(
                current_version="2019.06",
                minimum_version="2021.10",
                block_on_stale=False,
            )

        assert any("2019.06" in r.message for r in caplog.records)

    def test_env_var_block_mode_respected(self, monkeypatch):
        """FATF_MAPPING_VERSION_BLOCK=0 enables soft mode."""
        from reporting.fatf_risk_codes import validate_mapping_version

        monkeypatch.setenv("FATF_MAPPING_VERSION_BLOCK", "0")
        monkeypatch.setenv("FATF_MIN_MAPPING_VERSION", "2021.10")
        # Should not raise even with old version
        validate_mapping_version(current_version="2019.06")  # soft mode — no raise

    def test_env_var_block_mode_blocks_by_default(self, monkeypatch):
        """Default mode (FATF_MAPPING_VERSION_BLOCK=1) raises on stale version."""
        from reporting.fatf_risk_codes import MappingVersionError, validate_mapping_version

        monkeypatch.setenv("FATF_MAPPING_VERSION_BLOCK", "1")
        monkeypatch.setenv("FATF_MIN_MAPPING_VERSION", "2021.10")
        with pytest.raises(MappingVersionError):
            validate_mapping_version(current_version="2019.06")

    def test_export_ivms101_calls_validate_mapping_version(self):
        """export_ivms101 calls validate_mapping_version before producing output."""
        from reporting.fatf_exporter import export_ivms101
        from reporting.fatf_risk_codes import MappingVersionError

        with patch(
            "reporting.fatf_exporter.validate_mapping_version",
            side_effect=MappingVersionError("2019.06", "2021.10"),
        ):
            with pytest.raises(MappingVersionError):
                export_ivms101({"report_id": "x", "risk_score": 95})

    def test_export_ivms101_succeeds_with_current_mapping(self):
        """Acceptance criterion: export using current mapping succeeds without warning."""
        from reporting.fatf_exporter import export_ivms101

        report = {
            "report_id": "aaaaaaaa-0000-0000-0000-000000000001",
            "generated_at": "2024-06-01T12:00:00+00:00",
            "wallet": "GABC123456789STELLAR",
            "asset_pair": "USDC:GA5ZSEJY/XLM:native",
            "risk_score": 92,
            "score_lower": 82,
            "score_upper": 100,
            "verdict": "wash_trade",
            "top_shap_features": [],
            "benford_analysis": {},
            "trade_evidence": [],
            "model_metadata": {"version": "1.0"},
            "report_sha256": "abc123" * 10,
        }
        # Should succeed without raising
        with patch("reporting.fatf_exporter.validate_mapping_version"):
            doc = export_ivms101(report)
        assert doc["@type"] == "ivms101:IdentityPayload"

    def test_version_tuple_invalid_format_raises(self):
        """_version_tuple raises ValueError for malformed version strings."""
        from reporting.fatf_risk_codes import _version_tuple

        with pytest.raises(ValueError, match="Invalid mapping version"):
            _version_tuple("not-a-version")

    def test_mapping_version_metadata_present(self):
        """MAPPING_VERSION, MAPPING_GUIDANCE_DATE and MAPPING_GUIDANCE_DOCUMENT are set."""
        from reporting.fatf_risk_codes import (
            MAPPING_GUIDANCE_DATE,
            MAPPING_GUIDANCE_DOCUMENT,
            MAPPING_VERSION,
        )

        assert MAPPING_VERSION
        assert MAPPING_GUIDANCE_DATE
        assert MAPPING_GUIDANCE_DOCUMENT
        # Version format: "YYYY.MM"
        parts = MAPPING_VERSION.split(".")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()


# ---------------------------------------------------------------------------
# Issue #943 — Tamper-evident audit summary with cryptographic signing
# ---------------------------------------------------------------------------


class TestAuditSummarySigning:
    """Issue #943 — Ed25519 signing for audit summaries."""

    @pytest.fixture()
    def key_pair(self, tmp_path: Path):
        """Generate a fresh Ed25519 key pair for testing."""
        pytest.importorskip("cryptography", reason="cryptography package required for signing")
        from reporting.audit_summary import generate_signing_key_pair

        priv_path = tmp_path / "audit_signing.pem"
        pub_path = tmp_path / "audit_verify.pem"
        generate_signing_key_pair(priv_path, pub_path)
        return str(priv_path), str(pub_path)

    @pytest.fixture()
    def sample_summary(self):
        from reporting.audit_summary import AuditSummaryBuilder

        report = {
            "report_id": "rpt-001",
            "generated_at": "2025-01-15T12:00:00+00:00",
            "wallet": "GBLT2XJKNNB7DOYP3QOELK4WPU64BXFMFYXKGQP6K5FKZRZE6SYGNM",
            "asset_pair": "USDC:native/XLM:native",
            "risk_score": 82,
            "score_lower": 72,
            "score_upper": 92,
            "verdict": "wash_trade",
            "top_shap_features": [],
            "benford_analysis": {},
            "trade_evidence": [],
            "model_metadata": {"version": "1.0"},
        }
        return AuditSummaryBuilder().build(report)

    def test_sign_and_verify_roundtrip(self, key_pair, sample_summary):
        """Acceptance criterion: verification tool validates an untampered export."""
        from reporting.audit_summary import load_signing_key, load_verify_key, sign_summary, verify_summary_signature

        priv_path, pub_path = key_pair
        private_key = load_signing_key(priv_path)
        public_key = load_verify_key(pub_path)

        signed_doc = sign_summary(sample_summary, private_key)
        assert "ed25519_signature" in signed_doc
        assert len(signed_doc["ed25519_signature"]) == 128  # 64 bytes hex-encoded

        result = verify_summary_signature(signed_doc, public_key)
        assert result is True

    def test_modified_document_is_rejected(self, key_pair, sample_summary):
        """Acceptance criterion: modified document is rejected by verify tool."""
        from reporting.audit_summary import (
            SignatureVerificationError,
            load_signing_key,
            load_verify_key,
            sign_summary,
            verify_summary_signature,
        )

        priv_path, pub_path = key_pair
        private_key = load_signing_key(priv_path)
        public_key = load_verify_key(pub_path)

        signed_doc = sign_summary(sample_summary, private_key)

        # Tamper with a field
        tampered = dict(signed_doc)
        tampered["risk_score"] = 0  # attacker tries to lower the risk score

        with pytest.raises(SignatureVerificationError):
            verify_summary_signature(tampered, public_key)

    def test_missing_signature_raises(self, key_pair, sample_summary):
        """Document without ed25519_signature raises SignatureVerificationError."""
        from reporting.audit_summary import SignatureVerificationError, load_verify_key, verify_summary_signature

        _, pub_path = key_pair
        public_key = load_verify_key(pub_path)

        doc_without_sig = sample_summary.to_dict()  # no signature field
        with pytest.raises(SignatureVerificationError, match="ed25519_signature"):
            verify_summary_signature(doc_without_sig, public_key)

    def test_wrong_key_rejects_signature(self, tmp_path, sample_summary):
        """Signature made with key A is rejected when verified with key B."""
        pytest.importorskip("cryptography", reason="cryptography package required for signing")
        from reporting.audit_summary import (
            SignatureVerificationError,
            generate_signing_key_pair,
            load_signing_key,
            load_verify_key,
            sign_summary,
            verify_summary_signature,
        )

        priv_a = tmp_path / "a_priv.pem"
        pub_a = tmp_path / "a_pub.pem"
        priv_b = tmp_path / "b_priv.pem"
        pub_b = tmp_path / "b_pub.pem"

        generate_signing_key_pair(priv_a, pub_a)
        generate_signing_key_pair(priv_b, pub_b)

        signed_with_a = sign_summary(sample_summary, load_signing_key(str(priv_a)))

        with pytest.raises(SignatureVerificationError):
            verify_summary_signature(signed_with_a, load_verify_key(str(pub_b)))

    def test_key_rotation_new_key_verifies(self, tmp_path, sample_summary):
        """Acceptance criterion: new key pair after rotation generates a verifiable signature."""
        pytest.importorskip("cryptography", reason="cryptography package required for signing")
        from reporting.audit_summary import (
            generate_signing_key_pair,
            load_signing_key,
            load_verify_key,
            sign_summary,
            verify_summary_signature,
        )

        # Original key pair
        priv1 = tmp_path / "priv1.pem"
        pub1 = tmp_path / "pub1.pem"
        generate_signing_key_pair(priv1, pub1)

        # Rotate: generate a new key pair
        priv2 = tmp_path / "priv2.pem"
        pub2 = tmp_path / "pub2.pem"
        generate_signing_key_pair(priv2, pub2)

        # New signatures with rotated key verify with the new public key
        signed_new = sign_summary(sample_summary, load_signing_key(str(priv2)))
        assert verify_summary_signature(signed_new, load_verify_key(str(pub2))) is True

        # Old signatures still verify with old public key (key rotation does not invalidate old sigs)
        signed_old = sign_summary(sample_summary, load_signing_key(str(priv1)))
        assert verify_summary_signature(signed_old, load_verify_key(str(pub1))) is True

    def test_generate_signing_key_pair_creates_files(self, tmp_path):
        """generate_signing_key_pair writes both PEM files."""
        pytest.importorskip("cryptography", reason="cryptography package required for signing")
        from reporting.audit_summary import generate_signing_key_pair

        priv_path = tmp_path / "test_priv.pem"
        pub_path = tmp_path / "test_pub.pem"
        generate_signing_key_pair(priv_path, pub_path)

        assert priv_path.exists()
        assert pub_path.exists()
        # Private key must have restricted permissions
        assert oct(priv_path.stat().st_mode)[-3:] == "600"

    def test_sha256_integrity_still_works_on_signed_doc(self, key_pair, sample_summary):
        """The SHA-256 integrity field is preserved in the signed document."""
        from reporting.audit_summary import load_signing_key, sign_summary

        priv_path, _ = key_pair
        private_key = load_signing_key(priv_path)
        signed_doc = sign_summary(sample_summary, private_key)

        assert signed_doc["summary_sha256"] == sample_summary.summary_sha256
        # The original summary's verify_integrity should still pass
        assert sample_summary.verify_integrity()

    def test_load_signing_key_missing_env_raises(self):
        """load_signing_key raises SigningError when no path is set."""
        pytest.importorskip("cryptography", reason="cryptography package required for signing")
        from reporting.audit_summary import SigningError, load_signing_key

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AUDIT_SIGNING_KEY_PATH", None)
            with pytest.raises(SigningError, match="AUDIT_SIGNING_KEY_PATH"):
                load_signing_key()
