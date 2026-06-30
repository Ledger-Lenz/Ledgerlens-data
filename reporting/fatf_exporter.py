"""FATF Travel Rule IVMS101 export for LedgerLens forensic reports.

Maps the proprietary LedgerLens forensic report format to JSON-LD conforming
to the IVMS101 1.0 data standard so exchange compliance teams can submit
flagged wallet reports to regulators and partner VASPs without manual
reformatting.

API::

    # Single report
    ivms_doc = export_ivms101(forensic_report.to_dict())

    # Batch — filters below FATF_EXPORT_THRESHOLD automatically
    ivms_docs = export_batch_ivms101([r.to_dict() for r in reports])

Security:
    Raw wallet addresses are pseudonymised by default.  Pass
    ``reveal_addresses=True`` **only** when the ``FATF_ADMIN_TOKEN``
    environment variable is set to a non-empty value.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jsonschema

from config import config
from reporting.fatf_risk_codes import RiskCode, map_to_risk_codes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "ivms101.json"
_IVMS101_CONTEXT = "https://intervasp.org/ivms101"
_IVMS101_TYPE = "ivms101:IdentityPayload"
_REPORTING_ENTITY = "LedgerLens"
_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class ExportValidationError(Exception):
    """Raised when an IVMS101 export fails JSON Schema validation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_schema() -> dict:
    with _SCHEMA_PATH.open() as fh:
        return json.load(fh)


def _mask_wallet(address: str) -> str:
    """Return a stable pseudonymous token for a wallet address.

    Uses the first 12 hex characters of SHA-256(address) so that:
    - the original address is not exposed, and
    - two records for the same wallet produce the same token (correlation safe).
    """
    digest = hashlib.sha256(address.encode()).hexdigest()
    return f"REDACTED-{digest[:12]}"


def _unavailable(field_name: str | None = None) -> dict:
    """Return the IVMS101 data-unavailable sentinel object.

    Fields with no corresponding detection data must not be omitted from the
    export; instead they are represented as::

        {"value": null, "dataUnavailable": true}

    An optional ``fieldName`` annotation is added when ``field_name`` is given.
    """
    sentinel: dict[str, Any] = {"value": None, "dataUnavailable": True}
    if field_name is not None:
        sentinel["fieldName"] = field_name
    return sentinel


def _check_admin_auth() -> bool:
    """Return True only when FATF_ADMIN_TOKEN env var is set to a non-empty value."""
    return bool(os.getenv("FATF_ADMIN_TOKEN"))


def _format_account_number(wallet: str, reveal: bool) -> str:
    return wallet if reveal else _mask_wallet(wallet)


def _build_natural_person(account_ref: str | dict) -> dict:
    """Build an IVMS101 naturalPerson block.

    Personal identification data (name, address, national ID, DoB) is
    unavailable in LedgerLens detection output, which operates on pseudonymous
    on-chain wallet addresses.  All such fields are emitted as
    data-unavailable sentinels so downstream parsers receive a complete
    object instead of missing keys.
    """
    return {
        "name": _unavailable("name"),
        "geographicAddress": _unavailable("geographicAddress"),
        "nationalIdentification": _unavailable("nationalIdentification"),
        "dateAndPlaceOfBirth": _unavailable("dateAndPlaceOfBirth"),
        "customerIdentification": account_ref,
        "countryOfResidence": _unavailable("countryOfResidence"),
    }


def _build_originator(account_ref: str | dict) -> dict:
    return {
        "accountNumber": [account_ref],
        "originatorPersons": [
            {"naturalPerson": _build_natural_person(account_ref)}
        ],
    }


def _build_beneficiary() -> dict:
    """IVMS101 beneficiary block.

    The beneficiary is unknown at detection time (wash rings involve many
    counterparties, none of whom are a designated beneficiary in the Travel
    Rule sense).  All sub-fields use the data-unavailable sentinel.
    """
    return {
        "beneficiaryPersons": [_unavailable("beneficiaryPersons")],
        "accountNumber": [_unavailable("accountNumber")],
    }


def _build_originating_vasp() -> dict:
    return {
        "originatingVASP": {
            "legalPerson": {
                "name": {
                    "nameIdentifier": [{"legalPersonName": _REPORTING_ENTITY}]
                },
                "geographicAddress": _unavailable("geographicAddress"),
                "nationalIdentification": _unavailable("nationalIdentification"),
                "countryOfRegistration": _unavailable("countryOfRegistration"),
            }
        }
    }


def _build_transaction_reference(report: dict) -> dict:
    raw_score = report.get("risk_score")
    risk_score: Any
    if raw_score is not None:
        risk_score = round(raw_score / 100.0, 4)
    else:
        risk_score = _unavailable("riskScore")

    asset_pair = report.get("asset_pair")
    if not asset_pair:
        asset_pair = _unavailable("assetPair")

    verdict = report.get("verdict")
    if not verdict:
        verdict = _unavailable("verdict")

    score_lower = report.get("score_lower")
    score_upper = report.get("score_upper")
    if score_lower is not None and score_upper is not None:
        confidence_interval: Any = {
            "lower": round(score_lower / 100.0, 4),
            "upper": round(score_upper / 100.0, 4),
        }
    else:
        confidence_interval = _unavailable("scoreConfidenceInterval")

    return {
        "reportId": report.get("report_id", ""),
        "assetPair": asset_pair,
        "riskScore": risk_score,
        "verdict": verdict,
        "scoreConfidenceInterval": confidence_interval,
    }


def _build_risk_indicators(report: dict) -> list[dict]:
    codes: list[RiskCode] = map_to_risk_codes(report)
    return [
        {
            "code": rc.code,
            "description": rc.description,
            "severity": rc.severity.value,
            "evidenceReference": report.get("report_id"),
        }
        for rc in codes
    ]


def _validate(doc: dict) -> None:
    schema = _load_schema()
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ExportValidationError(
            f"IVMS101 export failed schema validation: {exc.message} "
            f"(path: {' -> '.join(str(p) for p in exc.absolute_path)})"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_ivms101(
    forensic_report: dict,
    reveal_addresses: bool = False,
) -> dict:
    """Map a LedgerLens forensic report dict to an IVMS101-compatible JSON-LD document.

    The ``risk_score`` field (integer 0–100) is normalised to the [0, 1]
    range required by the IVMS101 schema.  Wallet addresses are pseudonymised
    by default; set ``reveal_addresses=True`` **and** the ``FATF_ADMIN_TOKEN``
    environment variable to include raw addresses.

    Fields that have no corresponding value in the detection output are
    populated with a data-unavailable sentinel object rather than being
    omitted, ensuring downstream parsers always receive structurally complete
    documents.

    Args:
        forensic_report: Dict produced by ``ForensicReport.to_dict()``.
        reveal_addresses: If True, include raw wallet addresses.  Requires
            ``FATF_ADMIN_TOKEN`` to be set; raises ``PermissionError`` otherwise.

    Returns:
        JSON-LD dict conforming to the IVMS101 LedgerLens 1.0 schema.

    Raises:
        PermissionError: If ``reveal_addresses=True`` without admin auth.
        ExportValidationError: If the assembled document fails schema validation.
    """
    if reveal_addresses and not _check_admin_auth():
        raise PermissionError(
            "reveal_addresses=True requires the FATF_ADMIN_TOKEN environment "
            "variable to be set to a non-empty value."
        )

    wallet: str = forensic_report.get("wallet", "")
    account_ref: Any = (
        _format_account_number(wallet, reveal_addresses)
        if wallet
        else _unavailable("accountNumber")
    )

    sha256_ref = forensic_report.get("report_sha256")
    source_sha256: Any = sha256_ref if sha256_ref else _unavailable("sourceReportSha256")

    doc = {
        "@context": _IVMS101_CONTEXT,
        "@type": _IVMS101_TYPE,
        "payloadMetadata": {
            "reportId": forensic_report.get("report_id", ""),
            "generatedAt": forensic_report.get("generated_at", ""),
            "reportingEntity": _REPORTING_ENTITY,
            "schemaVersion": _SCHEMA_VERSION,
            "sourceReportSha256": source_sha256,
        },
        "originator": _build_originator(account_ref),
        "beneficiary": _build_beneficiary(),
        "originatingVASP": _build_originating_vasp(),
        "beneficiaryVASP": _unavailable("beneficiaryVASP"),
        "riskIndicators": _build_risk_indicators(forensic_report),
        "transactionReference": _build_transaction_reference(forensic_report),
    }

    _validate(doc)
    return doc


def export_batch_ivms101(
    reports: list[dict],
    reveal_addresses: bool = False,
) -> list[dict]:
    """Export multiple forensic report dicts to IVMS101 format.

    Reports whose calibrated risk score falls below ``FATF_EXPORT_THRESHOLD``
    (``risk_score / 100 < threshold``) are silently excluded.  The threshold
    defaults to 0.85 and is configurable via the ``FATF_EXPORT_THRESHOLD``
    environment variable.

    Args:
        reports: List of dicts produced by ``ForensicReport.to_dict()``.
        reveal_addresses: Forwarded to ``export_ivms101``; requires admin auth.

    Returns:
        List of valid IVMS101 dicts, one per qualifying report.
    """
    threshold = config.FATF_EXPORT_THRESHOLD
    results: list[dict] = []

    for report in reports:
        raw_score = report.get("risk_score")
        if raw_score is None:
            continue
        if raw_score / 100.0 < threshold:
            continue
        results.append(export_ivms101(report, reveal_addresses=reveal_addresses))

    return results
