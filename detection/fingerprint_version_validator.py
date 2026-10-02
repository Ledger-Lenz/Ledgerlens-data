"""Bot fingerprint version compatibility validation.

Ensures that bot fingerprints consumed by downstream models are compatible
with the model's expected fingerprint feature schema, preventing silent
score drift when the fingerprinter is updated without retraining consumers.

Usage:
    from detection.fingerprint_version_validator import validate_fingerprint_version

    # In model inference path:
    validate_fingerprint_version(fingerprint, model_expected_version="v1")
"""

from __future__ import annotations

import logging

from ingestion.data_models import BotFingerprint

logger = logging.getLogger(__name__)


class FingerprintVersionMismatchError(Exception):
    """Raised when a fingerprint version doesn't match the model's expected version."""

    pass


def validate_fingerprint_version(
    fingerprint: BotFingerprint,
    expected_version: str,
    *,
    strict: bool = True,
) -> None:
    """Validate that fingerprint version matches the model's expected version.

    Args:
        fingerprint: BotFingerprint instance to validate.
        expected_version: Version string the consuming model was trained on.
        strict: When True (default), raises on mismatch; when False, logs warning only.

    Raises:
        FingerprintVersionMismatchError: When versions mismatch and strict=True.

    Examples:
        >>> fingerprint = BotFingerprint(account_id="GA...", fingerprint_version="v1")
        >>> validate_fingerprint_version(fingerprint, "v1")  # OK
        >>> validate_fingerprint_version(fingerprint, "v2")  # Raises
    """
    actual = getattr(fingerprint, "fingerprint_version", None)

    if actual is None:
        msg = (
            f"BotFingerprint for {fingerprint.account_id} missing fingerprint_version field. "
            f"Model expects {expected_version}. This likely indicates an old fingerprint "
            "generated before version tracking was added."
        )
        if strict:
            raise FingerprintVersionMismatchError(msg)
        logger.warning(msg)
        return

    if actual != expected_version:
        msg = (
            f"Fingerprint version mismatch for {fingerprint.account_id}: "
            f"fingerprint has version '{actual}' but model expects '{expected_version}'. "
            "Retraining or feature alignment required."
        )
        if strict:
            raise FingerprintVersionMismatchError(msg)
        logger.warning(msg)


def validate_fingerprint_batch(
    fingerprints: list[BotFingerprint],
    expected_version: str,
    *,
    strict: bool = True,
) -> list[BotFingerprint]:
    """Validate a batch of fingerprints against expected version.

    Args:
        fingerprints: List of BotFingerprint instances to validate.
        expected_version: Version string the consuming model was trained on.
        strict: When True (default), raises on first mismatch; when False, logs all mismatches.

    Returns:
        List of validated fingerprints (in strict mode) or valid fingerprints only (non-strict).

    Raises:
        FingerprintVersionMismatchError: When any fingerprint mismatches and strict=True.
    """
    valid = []
    for fp in fingerprints:
        try:
            validate_fingerprint_version(fp, expected_version, strict=strict)
            valid.append(fp)
        except FingerprintVersionMismatchError:
            if strict:
                raise
            # In non-strict mode, skip invalid fingerprints
            continue
    return valid


def get_supported_versions() -> list[str]:
    """Return list of supported fingerprint versions.

    Returns:
        List of version strings currently supported by the system.
    """
    return ["v1"]
