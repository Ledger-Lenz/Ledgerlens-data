"""Tests for typed deployment-mode configuration fixtures (Issue #543)."""

import pytest

from config import Config
from config.deployment_modes import (
    DEPLOYMENT_MODE_FIXTURES,
    DeploymentMode,
    DeploymentModeFixture,
    DeploymentModeValidationError,
    UnknownDeploymentModeError,
    apply_deployment_mode,
    get_deployment_mode_fixture,
)


def test_every_deployment_mode_has_a_registered_fixture():
    for mode in DeploymentMode:
        fixture = DEPLOYMENT_MODE_FIXTURES[mode]
        assert isinstance(fixture, DeploymentModeFixture)
        assert fixture.mode is mode
        assert fixture.description


@pytest.mark.parametrize("mode", list(DeploymentMode))
def test_get_deployment_mode_fixture_by_enum(mode):
    fixture = get_deployment_mode_fixture(mode)
    assert fixture.mode is mode


@pytest.mark.parametrize("mode", ["local", "testnet", "production"])
def test_get_deployment_mode_fixture_by_string(mode):
    fixture = get_deployment_mode_fixture(mode)
    assert fixture.mode.value == mode


def test_get_deployment_mode_fixture_unknown_mode_raises_typed_error():
    with pytest.raises(UnknownDeploymentModeError) as exc:
        get_deployment_mode_fixture("staging-eu-west")

    assert "staging-eu-west" in str(exc.value)
    assert "local" in str(exc.value)
    assert "testnet" in str(exc.value)
    assert "production" in str(exc.value)


@pytest.mark.parametrize("mode", list(DeploymentMode))
def test_apply_deployment_mode_validates_successfully(mode):
    with apply_deployment_mode(mode) as fixture:
        assert fixture.mode is mode
        # Overrides must actually be applied onto Config while inside the block.
        for name, value in fixture.overrides.items():
            assert getattr(Config, name) == value


def test_apply_deployment_mode_restores_previous_values_on_exit():
    original = Config.STELLAR_NETWORK
    with apply_deployment_mode(DeploymentMode.TESTNET):
        assert Config.STELLAR_NETWORK == "TESTNET"
    assert Config.STELLAR_NETWORK == original


def test_apply_deployment_mode_restores_on_exception():
    original = Config.STELLAR_NETWORK
    with pytest.raises(RuntimeError):
        with apply_deployment_mode(DeploymentMode.LOCAL):
            raise RuntimeError("boom")
    assert Config.STELLAR_NETWORK == original


def test_apply_deployment_mode_unknown_mode_raises_before_mutating_config():
    original = Config.STELLAR_NETWORK
    with pytest.raises(UnknownDeploymentModeError):
        with apply_deployment_mode("does-not-exist"):
            pass  # pragma: no cover - should never be reached
    assert Config.STELLAR_NETWORK == original


def test_apply_deployment_mode_surfaces_validation_failures(monkeypatch):
    broken = DeploymentModeFixture(
        mode=DeploymentMode.LOCAL,
        description="Deliberately invalid fixture for the negative test path.",
        overrides={"WATCHED_ASSET_PAIRS": [], "RISK_SCORE_DB_URL": "", "MODEL_DIR": ""},
        require_onchain=False,
    )
    monkeypatch.setitem(DEPLOYMENT_MODE_FIXTURES, DeploymentMode.LOCAL, broken)
    original = Config.RISK_SCORE_DB_URL

    with pytest.raises(DeploymentModeValidationError) as exc:
        with apply_deployment_mode(DeploymentMode.LOCAL):
            pass  # pragma: no cover - validation fails before the body runs

    assert exc.value.mode is DeploymentMode.LOCAL
    # Overrides applied during the failed validation must still be rolled back.
    assert Config.RISK_SCORE_DB_URL == original


def test_apply_deployment_mode_validate_false_skips_validation(monkeypatch):
    broken = DeploymentModeFixture(
        mode=DeploymentMode.LOCAL,
        description="Deliberately invalid fixture for the negative test path.",
        overrides={"WATCHED_ASSET_PAIRS": [], "RISK_SCORE_DB_URL": "", "MODEL_DIR": ""},
        require_onchain=False,
    )
    monkeypatch.setitem(DEPLOYMENT_MODE_FIXTURES, DeploymentMode.LOCAL, broken)

    with apply_deployment_mode(DeploymentMode.LOCAL, validate=False) as fixture:
        assert fixture is broken
        assert Config.RISK_SCORE_DB_URL == ""


def test_local_deployment_config_fixture(local_deployment_config):
    assert local_deployment_config.mode is DeploymentMode.LOCAL
    assert Config.STREAMING_BACKEND == "sse"
    assert Config.HORIZON_DEV_MODE is True


def test_testnet_deployment_config_fixture(testnet_deployment_config):
    assert testnet_deployment_config.mode is DeploymentMode.TESTNET
    assert Config.STELLAR_NETWORK == "TESTNET"
    assert Config.LEDGERLENS_CONTRACT_ID


def test_production_deployment_config_fixture(production_deployment_config):
    assert production_deployment_config.mode is DeploymentMode.PRODUCTION
    assert Config.STELLAR_NETWORK == "PUBLIC"
    assert Config.HORIZON_DEV_MODE is False


# ---------------------------------------------------------------------------
# Issue #955 — ProductionSafetyChecker tests
# ---------------------------------------------------------------------------


from config.deployment_modes import (  # noqa: E402 — appended block
    PRODUCTION_UNSAFE_FLAGS,
    ProductionSafetyChecker,
    ProductionSafetyError,
    ProductionSafetyViolation,
)


class _SafeConfig:
    """Minimal mock config with all unsafe flags set to safe values."""

    HORIZON_DEV_MODE = False
    DEBUG_MODE = False
    VERBOSE_ERRORS = False
    DISABLE_AUTH = False
    PERMISSIVE_CORS = False
    ALLOW_UNAUTHENTICATED_SCORING = False
    SKIP_MODEL_INTEGRITY_CHECK = False
    LEDGERLENS_OFFLINE = ""


class _UnsafeConfig(_SafeConfig):
    """Same mock config but with HORIZON_DEV_MODE deliberately misconfigured."""

    HORIZON_DEV_MODE = True


class _MultipleUnsafeConfig(_SafeConfig):
    """Config with several unsafe flags enabled simultaneously."""

    DEBUG_MODE = True
    DISABLE_AUTH = True
    PERMISSIVE_CORS = True


def test_production_safety_checker_passes_safe_config():
    """ProductionSafetyChecker.verify_production_safe does not raise on a safe config."""
    checker = ProductionSafetyChecker()
    # Should not raise
    checker.verify_production_safe(_SafeConfig)


def test_production_safety_checker_catches_single_unsafe_flag():
    """ProductionSafetyError is raised when HORIZON_DEV_MODE=True in production."""
    checker = ProductionSafetyChecker()
    with pytest.raises(ProductionSafetyError) as exc_info:
        checker.verify_production_safe(_UnsafeConfig)

    err = exc_info.value
    assert len(err.violations) >= 1
    flag_names = [v.attr for v in err.violations]
    assert "HORIZON_DEV_MODE" in flag_names


def test_production_safety_checker_catches_multiple_unsafe_flags():
    """ProductionSafetyError lists all violated flags, not just the first."""
    checker = ProductionSafetyChecker()
    with pytest.raises(ProductionSafetyError) as exc_info:
        checker.verify_production_safe(_MultipleUnsafeConfig)

    err = exc_info.value
    flag_names = {v.attr for v in err.violations}
    assert "DEBUG_MODE" in flag_names
    assert "DISABLE_AUTH" in flag_names
    assert "PERMISSIVE_CORS" in flag_names


def test_production_safety_checker_check_returns_empty_list_when_safe():
    """check() returns an empty list for a fully safe config."""
    checker = ProductionSafetyChecker()
    violations = checker.check(_SafeConfig)
    assert violations == []


def test_production_safety_checker_check_returns_violations_list():
    """check() returns a non-empty list and does not raise."""
    checker = ProductionSafetyChecker()
    violations = checker.check(_UnsafeConfig)
    assert len(violations) >= 1
    assert all(isinstance(v, ProductionSafetyViolation) for v in violations)


def test_production_safety_error_message_contains_flag_name():
    """ProductionSafetyError message names the offending flag and its value."""
    checker = ProductionSafetyChecker()
    with pytest.raises(ProductionSafetyError) as exc_info:
        checker.verify_production_safe(_UnsafeConfig)

    msg = str(exc_info.value)
    assert "HORIZON_DEV_MODE" in msg
    assert "True" in msg


def test_production_safety_error_message_contains_remediation_guidance():
    """ProductionSafetyError message tells the operator what to do."""
    checker = ProductionSafetyChecker()
    with pytest.raises(ProductionSafetyError) as exc_info:
        checker.verify_production_safe(_UnsafeConfig)

    msg = str(exc_info.value)
    # Should include some guidance text
    assert "production" in msg.lower()


def test_production_unsafe_flags_covers_expected_flags():
    """PRODUCTION_UNSAFE_FLAGS enumerates all known debug/unsafe feature flags."""
    flag_names = {attr for attr, _, _ in PRODUCTION_UNSAFE_FLAGS}
    required = {
        "HORIZON_DEV_MODE",
        "DEBUG_MODE",
        "VERBOSE_ERRORS",
        "DISABLE_AUTH",
        "PERMISSIVE_CORS",
        "SKIP_MODEL_INTEGRITY_CHECK",
        "LEDGERLENS_OFFLINE",
    }
    missing = required - flag_names
    assert not missing, f"PRODUCTION_UNSAFE_FLAGS is missing: {missing}"


def test_production_safety_checker_custom_flags():
    """ProductionSafetyChecker accepts a custom flag list for targeted testing."""

    class _CustomConfig:
        MY_DEBUG_FLAG = True

    custom_flags = [("MY_DEBUG_FLAG", True, "Custom flag must be off in production.")]
    checker = ProductionSafetyChecker(unsafe_flags=custom_flags)

    with pytest.raises(ProductionSafetyError):
        checker.verify_production_safe(_CustomConfig)


def test_production_safety_checker_offline_flag_case_insensitive():
    """LEDGERLENS_OFFLINE='TRUE' (uppercase) is treated as unsafe."""

    class _UpperCaseOfflineConfig(_SafeConfig):
        LEDGERLENS_OFFLINE = "TRUE"

    checker = ProductionSafetyChecker()
    violations = checker.check(_UpperCaseOfflineConfig)
    flag_names = [v.attr for v in violations]
    assert "LEDGERLENS_OFFLINE" in flag_names
