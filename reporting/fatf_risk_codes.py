"""FATF virtual-asset risk indicator codes for IVMS101 exports — Issue #942.

Each code maps to a specific typology defined in the FATF Guidance for a
Risk-Based Approach to Virtual Assets and Virtual Asset Service Providers
(October 2021) and subsequent red-flag indicator guidance (2023 update).

Issue #942 — Jurisdiction-specific mapping version validation
-------------------------------------------------------------
The risk-code mapping is periodically revised when FATF issues new or updated
guidance.  A stale mapping could produce a non-compliant report.

This module now tracks a ``MAPPING_VERSION`` string and a
``MAPPING_GUIDANCE_DATE`` that identify which FATF guidance document the
mapping was derived from.  A ``MINIMUM_MAPPING_VERSION`` (configurable via
the ``FATF_MIN_MAPPING_VERSION`` environment variable) specifies the oldest
mapping version that is still acceptable for export.

Before every export, :func:`validate_mapping_version` is called.  Depending
on the ``FATF_MAPPING_VERSION_BLOCK`` environment variable it either:

- Raises :class:`MappingVersionError` (``FATF_MAPPING_VERSION_BLOCK=1``,
  the default — hard block), or
- Emits a warning log and continues (``FATF_MAPPING_VERSION_BLOCK=0`` — soft
  warn).

Updating the mapping
~~~~~~~~~~~~~~~~~~~~
When FATF releases new guidance:

1. Increment ``MAPPING_VERSION`` (e.g. ``"2021.10"`` → ``"2023.10"``).
2. Update ``MAPPING_GUIDANCE_DATE`` and ``MAPPING_GUIDANCE_DOCUMENT``.
3. Add, remove, or update entries in ``RISK_CODES`` and ``_FEATURE_CODE_MAP``
   to reflect the new red-flag indicators.
4. Update ``MINIMUM_MAPPING_VERSION`` in your deployment environment
   (``FATF_MIN_MAPPING_VERSION`` env var) once all instances have been updated.
5. Commit with a message referencing the FATF document and version bump.

Usage::

    from reporting.fatf_risk_codes import map_to_risk_codes, validate_mapping_version
    validate_mapping_version()          # raises/warns if version is stale
    codes = map_to_risk_codes(forensic_report_dict)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping version metadata — update these when FATF guidance changes
# ---------------------------------------------------------------------------

#: Semantic version of this risk-code mapping.  Format: "YYYY.MM".
MAPPING_VERSION: str = "2021.10"

#: Date of the FATF guidance document this mapping was derived from.
MAPPING_GUIDANCE_DATE: str = "October 2021"

#: Human-readable reference to the source document.
MAPPING_GUIDANCE_DOCUMENT: str = (
    "FATF Guidance for a Risk-Based Approach to Virtual Assets and VASPs, October 2021 "
    "(updated with 2023 red-flag indicator supplement)"
)

# ---------------------------------------------------------------------------
# Version comparison helpers
# ---------------------------------------------------------------------------


def _version_tuple(version: str) -> tuple[int, int]:
    """Parse a "YYYY.MM" version string to a comparable (year, month) tuple."""
    try:
        parts = version.strip().split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid mapping version format {version!r}. "
            "Expected 'YYYY.MM' (e.g. '2021.10')."
        ) from exc


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class MappingVersionError(Exception):
    """Raised when the risk-code mapping version is below the minimum required."""

    def __init__(
        self,
        current: str,
        minimum: str,
        guidance_doc: str = MAPPING_GUIDANCE_DOCUMENT,
    ) -> None:
        self.current = current
        self.minimum = minimum
        super().__init__(
            f"Risk-code mapping version {current!r} is below the minimum required version "
            f"{minimum!r}. Using a stale mapping could produce a non-compliant FATF report. "
            f"Update the mapping to reflect: {guidance_doc}. "
            "Set FATF_MIN_MAPPING_VERSION in your environment once all instances are updated. "
            "To allow exports with an older mapping (warn only), "
            "set FATF_MAPPING_VERSION_BLOCK=0."
        )


# ---------------------------------------------------------------------------
# Version validation
# ---------------------------------------------------------------------------


def validate_mapping_version(
    current_version: str = MAPPING_VERSION,
    minimum_version: str | None = None,
    *,
    block_on_stale: bool | None = None,
) -> None:
    """Validate the risk-code mapping version against the configured minimum.

    Parameters
    ----------
    current_version:
        The version of the mapping being used.  Defaults to this module's
        :data:`MAPPING_VERSION`.
    minimum_version:
        The minimum acceptable version.  Defaults to the
        ``FATF_MIN_MAPPING_VERSION`` environment variable, which itself
        defaults to :data:`MAPPING_VERSION` (i.e. only the current version
        is accepted by default).
    block_on_stale:
        If ``True``, raise :class:`MappingVersionError` on a stale version.
        If ``False``, emit a WARNING log instead (soft mode).
        Defaults to ``not (FATF_MAPPING_VERSION_BLOCK == "0")``.

    Raises
    ------
    MappingVersionError
        When the current version is below the minimum and ``block_on_stale``
        is ``True`` (the default).
    """
    if minimum_version is None:
        minimum_version = os.getenv("FATF_MIN_MAPPING_VERSION", MAPPING_VERSION)

    if block_on_stale is None:
        block_on_stale = os.getenv("FATF_MAPPING_VERSION_BLOCK", "1") != "0"

    if _version_tuple(current_version) < _version_tuple(minimum_version):
        if block_on_stale:
            raise MappingVersionError(current=current_version, minimum=minimum_version)
        else:
            logger.warning(
                "FATF risk-code mapping version %s is below the minimum %s. "
                "Report may be non-compliant with current FATF guidance (%s). "
                "Update the mapping table and bump MAPPING_VERSION.",
                current_version,
                minimum_version,
                MAPPING_GUIDANCE_DOCUMENT,
            )


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class RiskCode:
    code: str
    description: str
    severity: Severity
    fatf_reference: str  # FATF guidance paragraph / section reference


# ---------------------------------------------------------------------------
# Code registry
# ---------------------------------------------------------------------------

RISK_CODES: dict[str, RiskCode] = {
    "VA-001": RiskCode(
        code="VA-001",
        description="Structuring: transaction amounts split to avoid reporting thresholds",
        severity=Severity.HIGH,
        fatf_reference="FATF VA Guidance 2021, §5.2 — Red Flag A1",
    ),
    "VA-002": RiskCode(
        code="VA-002",
        description="Wash trading: artificial volume generated between related or controlled wallets",
        severity=Severity.CRITICAL,
        fatf_reference="FATF VA Guidance 2021, §6.1 — Red Flag C3",
    ),
    "VA-003": RiskCode(
        code="VA-003",
        description="Layering: funds moved through multiple intermediate hops to obscure origin",
        severity=Severity.HIGH,
        fatf_reference="FATF VA Guidance 2021, §5.4 — Red Flag B2",
    ),
    "VA-004": RiskCode(
        code="VA-004",
        description="Statistical anomaly: Benford's Law deviation in transaction amount distribution",
        severity=Severity.MEDIUM,
        fatf_reference="FATF VA Guidance 2021, §5.3 — Red Flag A3",
    ),
    "VA-005": RiskCode(
        code="VA-005",
        description="Round-trip cycling: assets returned to originating wallet within a short window",
        severity=Severity.HIGH,
        fatf_reference="FATF VA Guidance 2021, §6.2 — Red Flag C1",
    ),
    "VA-006": RiskCode(
        code="VA-006",
        description="Counterparty concentration: dominant single trading partner indicates coordinated activity",
        severity=Severity.MEDIUM,
        fatf_reference="FATF VA Guidance 2021, §5.5 — Red Flag A5",
    ),
    "VA-007": RiskCode(
        code="VA-007",
        description="Network cluster: wallet co-located with known flagged entities in the funding graph",
        severity=Severity.HIGH,
        fatf_reference="FATF VA Guidance 2021, §7.1 — Red Flag D2",
    ),
    "VA-008": RiskCode(
        code="VA-008",
        description="Velocity anomaly: unusual spike in transaction frequency or total volume",
        severity=Severity.MEDIUM,
        fatf_reference="FATF VA Guidance 2021, §5.1 — Red Flag A2",
    ),
    "VA-009": RiskCode(
        code="VA-009",
        description=(
            "Self-matching: coordinated buy/sell orders between wallets sharing a common funding source"
        ),
        severity=Severity.CRITICAL,
        fatf_reference="FATF VA Guidance 2021, §6.3 — Red Flag C4",
    ),
}

# ---------------------------------------------------------------------------
# Feature → code mapping (SHAP feature name prefix → risk code)
# ---------------------------------------------------------------------------

_FEATURE_CODE_MAP: list[tuple[str, str]] = [
    ("benford_mad", "VA-004"),
    ("round_trip_frequency", "VA-005"),
    ("counterparty_concentration_ratio", "VA-006"),
    ("self_matching_rate", "VA-009"),
    ("velocity", "VA-008"),
    ("cross_pair", "VA-003"),
]

# ---------------------------------------------------------------------------
# Public mapping function
# ---------------------------------------------------------------------------


def map_to_risk_codes(report: dict) -> list[RiskCode]:
    """Derive FATF risk indicator codes from a forensic report dict.

    Maps verdict and top SHAP features to the most relevant FATF typology
    codes.  Deduplication is applied so each code appears at most once.

    Args:
        report: Dict produced by ``ForensicReport.to_dict()``.

    Returns:
        Ordered list of ``RiskCode`` objects, highest-severity first.
    """
    seen: set[str] = set()
    codes: list[RiskCode] = []

    def _add(code_id: str) -> None:
        if code_id not in seen and code_id in RISK_CODES:
            seen.add(code_id)
            codes.append(RISK_CODES[code_id])

    verdict = report.get("verdict", "")

    if verdict == "wash_trade":
        _add("VA-002")
        _add("VA-009")
    elif verdict == "suspicious":
        _add("VA-001")

    shap_features: list[dict] = report.get("top_shap_features", [])
    for entry in shap_features:
        fname: str = entry.get("feature", "")
        contribution = entry.get("contribution", 0)
        if not isinstance(contribution, (int, float)) or contribution <= 0:
            continue
        for prefix, code_id in _FEATURE_CODE_MAP:
            if prefix in fname:
                _add(code_id)
                break

    _severity_order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    codes.sort(key=lambda rc: _severity_order[rc.severity])
    return codes
