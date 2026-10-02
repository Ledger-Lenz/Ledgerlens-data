"""Reporting export services with typed output contracts.

Existing exporters (`reporting/fatf_exporter.py`, `narrative_builder.py`)
each hand-roll their own output handling. This module adds a small, reusable
export layer that sits in front of them (or any tabular report data):

    - `FieldSpec` / `ReportSchema` declare a typed contract for what a report
      record must contain, so a bad record fails at export time with a
      precise "row N, field X" diagnostic instead of producing a malformed
      file downstream.
    - `ExportResult` is a typed output contract (bytes/text + content_type +
      filename + checksum) every exporter returns, so callers (API handlers,
      CLI commands, batch jobs) can treat all formats uniformly.
    - `JSONExporter`, `NDJSONExporter`, and `CSVExporter` implement the
      `ReportExporter` contract; `EXPORT_REGISTRY` and `export_report()` give
      a single call site that picks the exporter by format name.

Data-minimization profiles (Issue #945)
----------------------------------------
Recipient types require different levels of detail. Rather than one-size-fits-all
exports, every export can be scoped to a named **profile** that specifies
exactly which fields are included/excluded and whether aggregate-only mode is
applied.

Profiles are defined in `EXPORT_PROFILES`. Two built-in profiles ship:

* ``"regulator_full"`` — full detail including raw SHAP values, individual
  trade records, and wallet-level features. Intended for regulators and
  internal auditors with appropriate data-handling agreements.
* ``"partner_aggregate"`` — aggregate-only view, stripping all wallet
  identifiers, individual trade data, and raw SHAP values. Safe for sharing
  with exchange partners and third-party analytics providers.

The default profile (used when no profile is specified) is always the **most
restrictive** one — ``"partner_aggregate"`` — following the principle of
least privilege.

Usage::

    schema = ReportSchema(fields=[
        FieldSpec("wallet", str, required=True),
        FieldSpec("risk_score", (int, float), required=True),
    ])
    # Full detail for a regulator:
    result = export_report(records, schema=schema, fmt="csv",
                           profile="regulator_full")
    # Aggregate-only for a partner (or omit profile= to use the default):
    result = export_report(records, schema=schema, fmt="json",
                           profile="partner_aggregate")
    result.content_type   # "application/json"
    result.checksum       # sha256 of result.content, for integrity checks
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol


class SchemaValidationError(ValueError):
    """Raised when a record fails to satisfy a `ReportSchema`.

    Carries the row index and field name so a caller can locate the exact
    offending record in a large batch export without re-scanning it.
    """

    def __init__(self, row_index: int, field_name: str, reason: str):
        self.row_index = row_index
        self.field_name = field_name
        self.reason = reason
        super().__init__(f"row {row_index}, field {field_name!r}: {reason}")


class UnsupportedFormatError(ValueError):
    """Raised when `export_report` is asked for a format with no registered exporter."""


@dataclass(frozen=True)
class FieldSpec:
    """Declares the expected shape of one field in an exported record."""

    name: str
    expected_type: type | tuple[type, ...]
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ReportSchema:
    """Typed contract a batch of report records must satisfy before export."""

    fields: list[FieldSpec]

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def validate_record(self, record: dict[str, Any], row_index: int) -> None:
        for spec in self.fields:
            present = spec.name in record and record[spec.name] is not None
            if not present:
                if spec.required:
                    raise SchemaValidationError(row_index, spec.name, "required field is missing")
                continue
            if not isinstance(record[spec.name], spec.expected_type):
                raise SchemaValidationError(
                    row_index,
                    spec.name,
                    f"expected {spec.expected_type}, got {type(record[spec.name]).__name__}",
                )

    def validate_batch(self, records: list[dict[str, Any]]) -> None:
        for i, record in enumerate(records):
            self.validate_record(record, i)


@dataclass(frozen=True)
class ExportResult:
    """Typed output contract returned by every `ReportExporter`."""

    content: bytes
    content_type: str
    filename: str
    record_count: int
    checksum: str

    def write(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.content)


def _make_result(
    content: str | bytes, content_type: str, filename: str, record_count: int
) -> ExportResult:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    checksum = hashlib.sha256(payload).hexdigest()
    return ExportResult(
        content=payload,
        content_type=content_type,
        filename=filename,
        record_count=record_count,
        checksum=checksum,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


class ReportExporter(Protocol):
    """Contract every report exporter must satisfy."""

    format_name: str
    content_type: str

    def export(
        self, records: list[dict[str, Any]], schema: ReportSchema | None = None
    ) -> ExportResult: ...


class JSONExporter:
    """Exports records as a single pretty-printed JSON array."""

    format_name = "json"
    content_type = "application/json"

    def export(
        self, records: list[dict[str, Any]], schema: ReportSchema | None = None
    ) -> ExportResult:
        if schema is not None:
            schema.validate_batch(records)
        body = json.dumps(records, indent=2, default=_json_default, sort_keys=True)
        return _make_result(body, self.content_type, "report.json", len(records))


class NDJSONExporter:
    """Exports records as newline-delimited JSON (one object per line).

    Preferred for large/streamed exports where a caller wants to process
    records incrementally rather than parsing one large JSON array.
    """

    format_name = "ndjson"
    content_type = "application/x-ndjson"

    def export(
        self, records: list[dict[str, Any]], schema: ReportSchema | None = None
    ) -> ExportResult:
        if schema is not None:
            schema.validate_batch(records)
        lines = [json.dumps(r, default=_json_default, sort_keys=True) for r in records]
        body = "\n".join(lines) + ("\n" if lines else "")
        return _make_result(body, self.content_type, "report.ndjson", len(records))


class CSVExporter:
    """Exports records as CSV.

    Column order is taken from `schema.field_names()` when a schema is
    given; otherwise from the union of keys across all records, sorted for
    determinism. Missing values are written as empty cells.
    """

    format_name = "csv"
    content_type = "text/csv"

    def export(
        self, records: list[dict[str, Any]], schema: ReportSchema | None = None
    ) -> ExportResult:
        if schema is not None:
            schema.validate_batch(records)
            columns = schema.field_names()
        else:
            columns = sorted({key for record in records for key in record})

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col, "") for col in columns})
        return _make_result(buffer.getvalue(), self.content_type, "report.csv", len(records))


# ---------------------------------------------------------------------------
# Data-minimization profiles (Issue #945)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportProfile:
    """Defines field-level inclusion/exclusion rules for a recipient type.

    Attributes:
        name: Unique profile identifier (e.g. ``"regulator_full"``).
        description: Human-readable description for compliance reviewers.
        excluded_fields: Fields that must **never** appear in exports for this
            profile. Applied before any schema validation.
        aggregate_only: When ``True``, wallet-level identifiers and individual
            trade records are stripped and replaced with aggregate summaries.
        allowed_fields: When non-empty, acts as an allowlist — only these
            fields are retained (takes precedence over ``excluded_fields``).
    """

    name: str
    description: str
    excluded_fields: frozenset[str] = field(default_factory=frozenset)
    aggregate_only: bool = False
    allowed_fields: frozenset[str] = field(default_factory=frozenset)

    def apply(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return *records* filtered to satisfy this profile's rules.

        If ``aggregate_only`` is ``True``, wallet identifiers and raw trade
        fields are stripped before field-level exclusion is applied.  The
        ``allowed_fields`` allowlist (when non-empty) further narrows the
        output to exactly those fields.
        """
        result: list[dict[str, Any]] = []
        for record in records:
            filtered = dict(record)

            # Aggregate-only: drop wallet identifiers and individual-trade data
            if self.aggregate_only:
                filtered = {
                    k: v
                    for k, v in filtered.items()
                    if k not in _AGGREGATE_EXCLUDED_FIELDS
                }

            # Explicit exclusion list
            if self.excluded_fields:
                filtered = {k: v for k, v in filtered.items() if k not in self.excluded_fields}

            # Allowlist (whitelist): keep only explicitly permitted fields
            if self.allowed_fields:
                filtered = {k: v for k, v in filtered.items() if k in self.allowed_fields}

            result.append(filtered)
        return result


#: Fields always stripped in ``aggregate_only`` mode — wallet identifiers,
#: raw trade-level data, and model-internal details that could de-anonymise
#: individual trading activity.
_AGGREGATE_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        # Wallet identifiers
        "wallet",
        "wallet_id",
        "account_id",
        "originator_wallet",
        "beneficiary_wallet",
        # Individual trade records (list/dict fields)
        "trades",
        "raw_trades",
        "anomalous_trades",
        "trade_ids",
        # Raw SHAP values — can be reverse-engineered to infer wallet activity
        "shap_values",
        "shap_contributions",
        "feature_contributions",
        "top_shap_features",
        # Private feature vectors
        "feature_vector",
        "feature_row",
    }
)


# -----------------------------------------------------------------------
# Built-in profiles
# -----------------------------------------------------------------------

#: **Regulator full-detail profile** — no field exclusions beyond what the
#: schema itself requires. Intended for FATF Travel Rule regulators, FinCEN
#: SAR filings, and internal compliance audits operating under a data-handling
#: agreement. Includes raw SHAP values, individual trade records, and all
#: wallet-level identifiers.
PROFILE_REGULATOR_FULL = ExportProfile(
    name="regulator_full",
    description=(
        "Full detail for regulatory bodies (FATF, FinCEN, SEC) and internal "
        "compliance auditors. Includes all wallet identifiers, individual trade "
        "records, and raw SHAP feature attributions. Requires a signed data-"
        "handling agreement before transmission."
    ),
    excluded_fields=frozenset(),
    aggregate_only=False,
)

#: **Partner aggregate-only profile** — strips all wallet identifiers, raw
#: SHAP values, and individual trade records. Safe for exchange partners,
#: third-party analytics providers, and other recipients without a full
#: data-handling agreement. This is the **default** profile applied when no
#: profile is specified, implementing the principle of least privilege.
PROFILE_PARTNER_AGGREGATE = ExportProfile(
    name="partner_aggregate",
    description=(
        "Aggregate-only view for exchange partners and third-party analytics "
        "providers. Wallet identifiers, individual trade records, and raw SHAP "
        "values are stripped. Only aggregated risk metrics and summary statistics "
        "are included. Safe for sharing without a full data-handling agreement."
    ),
    excluded_fields=frozenset(),
    aggregate_only=True,
)

#: Registry of all named export profiles.  Add custom profiles here or via
#: :func:`register_profile` to make them available to :func:`export_report`.
PROFILE_REGISTRY: dict[str, ExportProfile] = {
    PROFILE_REGULATOR_FULL.name: PROFILE_REGULATOR_FULL,
    PROFILE_PARTNER_AGGREGATE.name: PROFILE_PARTNER_AGGREGATE,
}

#: Name of the default (most restrictive) profile applied when the caller
#: does not specify one. Changing this constant changes the project-wide
#: default — always pick the **most restrictive** available profile.
DEFAULT_PROFILE_NAME: str = PROFILE_PARTNER_AGGREGATE.name


class UnknownProfileError(ValueError):
    """Raised when ``export_report`` is asked for a profile that has no entry
    in ``PROFILE_REGISTRY``."""


def register_profile(profile: ExportProfile) -> None:
    """Register a custom :class:`ExportProfile` under its ``name``.

    Allows other modules (e.g. tenant-specific exporters) to add profiles
    without modifying this file.
    """
    PROFILE_REGISTRY[profile.name] = profile


EXPORT_REGISTRY: dict[str, ReportExporter] = {
    "json": JSONExporter(),
    "ndjson": NDJSONExporter(),
    "csv": CSVExporter(),
}


def export_report(
    records: list[dict[str, Any]],
    fmt: str,
    schema: ReportSchema | None = None,
    profile: str | None = None,
) -> ExportResult:
    """Single call site for exporting a batch of records in any registered format.

    Args:
        records: The report data to export.
        fmt: Export format (``"json"``, ``"ndjson"``, or ``"csv"``).
        schema: Optional typed schema; validated before export.
        profile: Named data-minimization profile controlling which fields are
            included. Defaults to ``DEFAULT_PROFILE_NAME`` (``"partner_aggregate"``
            — the most restrictive built-in profile). Pass
            ``"regulator_full"`` for exports to regulatory bodies.

    Raises:
        UnsupportedFormatError: ``fmt`` has no registered exporter. The error
            message lists the available formats.
        UnknownProfileError: ``profile`` is not in ``PROFILE_REGISTRY``.
        SchemaValidationError: ``schema`` is given and a record violates it
            *after* profile filtering; the error names the offending row and
            field.
    """
    # Resolve profile — default to most restrictive
    profile_name = profile if profile is not None else DEFAULT_PROFILE_NAME
    export_profile = PROFILE_REGISTRY.get(profile_name)
    if export_profile is None:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        raise UnknownProfileError(
            f"unknown export profile {profile_name!r}; available profiles: {available}"
        )

    # Apply data-minimization filtering
    filtered_records = export_profile.apply(records)

    exporter = EXPORT_REGISTRY.get(fmt)
    if exporter is None:
        available = ", ".join(sorted(EXPORT_REGISTRY))
        raise UnsupportedFormatError(
            f"unknown export format {fmt!r}; available formats: {available}"
        )
    return exporter.export(filtered_records, schema=schema)


def register_exporter(exporter: ReportExporter) -> None:
    """Registers a custom `ReportExporter` under its `format_name`.

    Lets other modules (e.g. a future PDF or Parquet exporter) plug into
    `export_report()` without modifying this file.
    """
    EXPORT_REGISTRY[exporter.format_name] = exporter
