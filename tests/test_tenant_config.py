"""Tests for Pydantic schema validation and drift detection in tenant config (Issue #954)."""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml

from config.tenants_schema import TenantConfigSchema, TenantsFileSchema, validate_tenants_yaml
from config.tenant_config import (
    TenantConfig,
    TenantConfigDriftDetector,
    TenantConfigDriftError,
    load_tenants_config,
    get_tenant_config,
)


# ---------------------------------------------------------------------------
# Schema validation — reject malformed config
# ---------------------------------------------------------------------------


def test_schema_rejects_unknown_field():
    """Pydantic raises ValidationError for an unrecognised field in a tenant entry."""
    from pydantic import ValidationError

    raw = {
        "tenants": {
            "test_tenant": {
                "risk_threshold": 70,
                "benford_min_sample": 100,
                "alert_channels": ["stdout"],
                "asset_pair_whitelist": ["USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"],
                "totally_unknown_field": "should_fail",
            }
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        TenantsFileSchema.model_validate(raw)
    assert "totally_unknown_field" in str(exc_info.value)


def test_schema_rejects_risk_threshold_out_of_range():
    """risk_threshold must be 0–100; values outside that range are rejected."""
    from pydantic import ValidationError

    for bad_value in (-1, 101, 200):
        raw = {
            "tenants": {
                "t": {
                    "risk_threshold": bad_value,
                    "benford_min_sample": 50,
                    "alert_channels": ["stdout"],
                    "asset_pair_whitelist": [],
                }
            }
        }
        with pytest.raises(ValidationError, match="risk_threshold"):
            TenantsFileSchema.model_validate(raw)


def test_schema_rejects_unknown_alert_channel():
    """Unrecognised alert channels are rejected."""
    from pydantic import ValidationError

    raw = {
        "tenants": {
            "t": {
                "risk_threshold": 70,
                "benford_min_sample": 50,
                "alert_channels": ["stdout", "pigeon_post"],
                "asset_pair_whitelist": [],
            }
        }
    }
    with pytest.raises(ValidationError):
        TenantsFileSchema.model_validate(raw)


def test_schema_rejects_malformed_asset_pair():
    """Asset pair entries without ':' are rejected."""
    from pydantic import ValidationError

    raw = {
        "tenants": {
            "t": {
                "risk_threshold": 70,
                "benford_min_sample": 50,
                "alert_channels": ["stdout"],
                "asset_pair_whitelist": ["USDC_NATIVE"],  # missing ':' separator
            }
        }
    }
    with pytest.raises(ValidationError):
        TenantsFileSchema.model_validate(raw)


def test_schema_rejects_invalid_tenant_id():
    """Tenant IDs with special characters (spaces, dots) are rejected."""
    from pydantic import ValidationError

    raw = {
        "tenants": {
            "invalid tenant id!": {
                "risk_threshold": 70,
                "benford_min_sample": 50,
                "alert_channels": ["stdout"],
                "asset_pair_whitelist": [],
            }
        }
    }
    with pytest.raises(ValidationError):
        TenantsFileSchema.model_validate(raw)


def test_schema_accepts_valid_config():
    """A fully valid tenant config passes schema validation without errors."""
    raw = {
        "tenants": {
            "exchange_a": {
                "risk_threshold": 70,
                "benford_min_sample": 100,
                "alert_channels": ["stdout"],
                "asset_pair_whitelist": [
                    "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
                ],
                "threshold_strategy": "static",
                "threshold_config": {},
            },
            "exchange_b": {
                "risk_threshold": 75,
                "benford_min_sample": 50,
                "alert_channels": ["webhook"],
                "asset_pair_whitelist": [
                    "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
                ],
                "threshold_strategy": "statistical",
                "threshold_config": {"recall_floor": 0.85, "target_metric": "f1"},
            },
        }
    }
    schema = TenantsFileSchema.model_validate(raw)
    assert set(schema.tenants.keys()) == {"exchange_a", "exchange_b"}
    assert schema.tenants["exchange_a"].risk_threshold == 70


def test_validate_tenants_yaml_loads_real_file(tmp_path):
    """validate_tenants_yaml correctly loads and validates the sample tenants.yaml."""
    # Write a valid YAML file to a temp path
    content = textwrap.dedent("""\
        tenants:
          tenant_x:
            risk_threshold: 65
            benford_min_sample: 80
            alert_channels: [stdout]
            asset_pair_whitelist:
              - "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    """)
    f = tmp_path / "tenants.yaml"
    f.write_text(content)
    schema = validate_tenants_yaml(str(f))
    assert "tenant_x" in schema.tenants


def test_validate_tenants_yaml_raises_on_malformed_file(tmp_path):
    """validate_tenants_yaml raises ValidationError on malformed YAML content."""
    from pydantic import ValidationError

    content = textwrap.dedent("""\
        tenants:
          bad_tenant:
            risk_threshold: 9999
            benford_min_sample: 50
            alert_channels: [stdout]
            asset_pair_whitelist: []
    """)
    f = tmp_path / "tenants_bad.yaml"
    f.write_text(content)
    with pytest.raises(ValidationError):
        validate_tenants_yaml(str(f))


def test_load_tenants_config_validates_on_load(tmp_path):
    """load_tenants_config raises ValidationError when YAML has bad fields."""
    from pydantic import ValidationError

    bad_yaml = textwrap.dedent("""\
        tenants:
          t:
            risk_threshold: 70
            benford_min_sample: 50
            alert_channels: [stdout]
            asset_pair_whitelist: []
            mystery_key: not_allowed
    """)
    f = tmp_path / "tenants.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_tenants_config(str(f))


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def _make_configs() -> dict[str, TenantConfig]:
    return {
        "exchange_a": TenantConfig(
            risk_threshold=70,
            benford_min_sample=100,
            alert_channels=["stdout"],
            asset_pair_whitelist=["USDC:GA5Z.../XLM:native"],
        ),
        "exchange_b": TenantConfig(
            risk_threshold=75,
            benford_min_sample=50,
            alert_channels=["webhook"],
            asset_pair_whitelist=["USDC:GA5Z.../XLM:native"],
        ),
    }


def test_drift_detector_passes_when_unchanged(tmp_path):
    """No drift when config matches the deployed snapshot exactly."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "snap.json"))
    detector.save_snapshot(configs)

    # Should not raise
    detector.assert_no_drift(configs)


def test_drift_detector_flags_threshold_change(tmp_path):
    """TenantConfigDriftError raised when a tenant's risk_threshold changes."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "snap.json"))
    detector.save_snapshot(configs)

    # Mutate one tenant's threshold
    changed = dict(configs)
    changed["exchange_a"] = TenantConfig(
        risk_threshold=80,  # was 70
        benford_min_sample=100,
        alert_channels=["stdout"],
        asset_pair_whitelist=["USDC:GA5Z.../XLM:native"],
    )

    with pytest.raises(TenantConfigDriftError) as exc_info:
        detector.assert_no_drift(changed)

    err = str(exc_info.value)
    assert "exchange_a" in err
    assert "risk_threshold" in err


def test_drift_detector_flags_added_tenant(tmp_path):
    """TenantConfigDriftError raised when a new tenant appears."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "snap.json"))
    detector.save_snapshot(configs)

    added = dict(configs)
    added["exchange_c"] = TenantConfig(
        risk_threshold=60,
        benford_min_sample=20,
        alert_channels=["stdout"],
        asset_pair_whitelist=[],
    )

    with pytest.raises(TenantConfigDriftError) as exc_info:
        detector.assert_no_drift(added)

    assert "exchange_c" in str(exc_info.value)


def test_drift_detector_flags_removed_tenant(tmp_path):
    """TenantConfigDriftError raised when an existing tenant is removed."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "snap.json"))
    detector.save_snapshot(configs)

    removed = {"exchange_a": configs["exchange_a"]}

    with pytest.raises(TenantConfigDriftError) as exc_info:
        detector.assert_no_drift(removed)

    assert "exchange_b" in str(exc_info.value)


def test_drift_detector_raises_file_not_found_when_no_snapshot(tmp_path):
    """assert_no_drift raises FileNotFoundError if no snapshot exists yet."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "nonexistent.json"))

    with pytest.raises(FileNotFoundError):
        detector.assert_no_drift(configs)


def test_drift_detector_detect_drift_returns_empty_list_when_aligned(tmp_path):
    """detect_drift returns [] when configs match snapshot."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "snap.json"))
    detector.save_snapshot(configs)

    diffs = detector.detect_drift(configs)
    assert diffs == []


def test_drift_detector_detect_drift_returns_list_of_diffs(tmp_path):
    """detect_drift returns a non-empty list describing each change."""
    configs = _make_configs()
    detector = TenantConfigDriftDetector(snapshot_path=str(tmp_path / "snap.json"))
    detector.save_snapshot(configs)

    changed = dict(configs)
    changed["exchange_a"] = TenantConfig(
        risk_threshold=90,
        benford_min_sample=100,
        alert_channels=["stdout"],
        asset_pair_whitelist=["USDC:GA5Z.../XLM:native"],
    )

    diffs = detector.detect_drift(changed)
    assert len(diffs) > 0
    assert any("risk_threshold" in d for d in diffs)


def test_drift_detector_save_and_reload_snapshot(tmp_path):
    """save_snapshot writes valid JSON that load_snapshot can reload."""
    configs = _make_configs()
    snap_path = str(tmp_path / "snap.json")
    detector = TenantConfigDriftDetector(snapshot_path=snap_path)
    written = detector.save_snapshot(configs)
    assert written == snap_path

    loaded = detector.load_snapshot()
    assert set(loaded.keys()) == {"exchange_a", "exchange_b"}
    assert loaded["exchange_a"]["risk_threshold"] == 70
