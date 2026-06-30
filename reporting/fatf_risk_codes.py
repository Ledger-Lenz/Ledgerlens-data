"""FATF virtual-asset risk indicator codes for IVMS101 exports.

Each code maps to a specific typology defined in the FATF Guidance for a
Risk-Based Approach to Virtual Assets and Virtual Asset Service Providers
(October 2021) and subsequent red-flag indicator guidance (2023 update).

Usage::

    from reporting.fatf_risk_codes import map_to_risk_codes
    codes = map_to_risk_codes(forensic_report_dict)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
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
