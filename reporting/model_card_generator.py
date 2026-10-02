"""Automated model card generator for LedgerLens trained models.

Reads the JSON metadata produced at training time and renders a Model Card
(Markdown or HTML) following the Google Model Card specification.

Usage:
    from reporting.model_card_generator import generate_model_card

    generate_model_card(
        model_metadata_path="models/model_metadata.json",
        output_path="models/MODEL_CARD_xgboost_0.2.0.md",
    )
"""

from __future__ import annotations

import json
import os
from typing import Literal

import jsonschema

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "model_metadata.json")

# A subgroup is flagged for reviewer attention when its F1 falls more than this
# fraction below the aggregate F1 (e.g. 0.10 == 10% relative shortfall).
_SUBGROUP_FLAG_THRESHOLD = 0.10

# Minimum subgroup size required before a breakdown is surfaced. This mirrors the
# k-anonymity floor used by the privacy work so that small subgroups cannot be
# used to re-identify individual accounts.
_MIN_SUBGROUP_SIZE = 30


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def _validate_metadata(metadata: dict) -> None:
    schema = _load_schema()
    try:
        jsonschema.validate(instance=metadata, schema=schema)
    except jsonschema.ValidationError as exc:
        raise MetadataValidationError(
            f"Invalid model metadata — missing or invalid field: {exc.json_path} — {exc.message}"
        ) from exc


def _aggregate_f1(perf: dict) -> float | None:
    """Return the mean F1 across all reported asset pairs, or ``None``."""
    f1s = [
        m["f1"]
        for m in perf.values()
        if isinstance(m, dict) and isinstance(m.get("f1"), (int, float))
    ]
    if not f1s:
        return None
    return sum(f1s) / len(f1s)


def _subgroup_breakdowns(metadata: dict) -> list[tuple[str, dict]]:
    """Return ``(dimension, breakdown)`` pairs for privacy-compliant subgroups.

    Only dimensions that are meaningful for this domain and that do not expose
    individual accounts are included: asset pair, account-age bucket, and
    account type. Subgroups smaller than ``_MIN_SUBGROUP_SIZE`` are dropped to
    avoid re-identification risk.
    """
    raw = metadata.get("subgroup_performance", {})
    breakdowns: list[tuple[str, dict]] = []
    for dimension, groups in raw.items():
        if not isinstance(groups, dict):
            continue
        kept = {
            label: metrics
            for label, metrics in groups.items()
            if isinstance(metrics, dict)
            and metrics.get("sample_size", 0) >= _MIN_SUBGROUP_SIZE
        }
        if kept:
            breakdowns.append((dimension, kept))
    return breakdowns


def _render_subgroup_section(metadata: dict) -> list[str]:
    """Render the fairness / subgroup performance section."""
    perf = metadata.get("performance_metrics", {})
    aggregate_f1 = _aggregate_f1(perf)
    breakdowns = _subgroup_breakdowns(metadata)

    lines = ["", "## Subgroup Performance (Fairness)", ""]

    if not breakdowns:
        lines.append("_No privacy-compliant subgroup breakdowns available._")
        return lines

    if aggregate_f1 is not None:
        lines.append(f"Aggregate F1 (mean across asset pairs): **{aggregate_f1:.4f}**")
        lines.append("")

    flagged: list[str] = []
    for dimension, groups in breakdowns:
        lines += [f"### By {dimension}", ""]
        lines += [
            "| Subgroup | Sample Size | Precision | Recall | F1 | Flag |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for label, metrics in groups.items():
            precision = metrics.get("precision", "N/A")
            recall = metrics.get("recall", "N/A")
            f1 = metrics.get("f1", "N/A")
            sample_size = metrics.get("sample_size", "N/A")
            flag = ""
            if (
                aggregate_f1 is not None
                and isinstance(f1, (int, float))
                and aggregate_f1 > 0
                and (aggregate_f1 - f1) / aggregate_f1 > _SUBGROUP_FLAG_THRESHOLD
            ):
                flag = "⚠️ below aggregate"
                flagged.append(f"{dimension} / {label}")
            lines.append(
                f"| {label} | {sample_size} | {precision} | {recall} | {f1} | {flag} |"
            )
        lines.append("")

    if flagged:
        lines.append(
            "**Reviewer attention required** — the following subgroups perform "
            f"more than {_SUBGROUP_FLAG_THRESHOLD:.0%} below the aggregate F1: "
            + ", ".join(flagged)
            + "."
        )
    else:
        lines.append("_No subgroups fall significantly below the aggregate F1._")

    return lines


def _render_markdown(metadata: dict) -> str:
    name = metadata["model_name"]
    version = metadata.get("ledgerlens_version", "unknown")
    trained_at = metadata.get("training_date", metadata.get("trained_at", "unknown"))
    dataset_version = metadata["dataset_version"]
    intended_use = metadata["intended_use"]
    out_of_scope = metadata["out_of_scope_uses"]
    limitations = metadata["known_limitations"]
    hyperparams = metadata.get("hyperparameters", {})
    perf = metadata.get("performance_metrics", {})
    fingerprint = metadata.get("dataset_fingerprint", "N/A")

    lines = [
        f"# Model Card — {name} v{version}",
        "",
        f"**Training date:** {trained_at}  ",
        f"**Dataset version:** {dataset_version}  ",
        f"**Dataset fingerprint (SHA-256):** `{fingerprint}`  ",
        "",
        "## Intended Use",
        "",
        intended_use,
        "",
        "## Out-of-Scope Uses",
        "",
        out_of_scope,
        "",
        "## Known Limitations",
        "",
        limitations,
        "",
        "## Hyperparameters",
        "",
    ]
    if hyperparams:
        lines += ["| Parameter | Value |", "| --- | --- |"]
        for k, v in hyperparams.items():
            lines.append(f"| `{k}` | `{v}` |")
    else:
        lines.append("_Not specified._")

    lines += ["", "## Performance Metrics", ""]

    if perf:
        lines += ["| Asset Pair | Precision | Recall | F1 |", "| --- | --- | --- | --- |"]
        for pair, m in perf.items():
            precision = m.get("precision", "N/A")
            recall = m.get("recall", "N/A")
            f1 = m.get("f1", "N/A")
            lines.append(f"| {pair} | {precision} | {recall} | {f1} |")
    else:
        lines.append("_Not available._")

    lines += _render_subgroup_section(metadata)

    shap_path = metadata.get("shap_importance_chart_path")
    if shap_path:
        lines += ["", "## SHAP Feature Importance", "", f"![SHAP chart]({shap_path})"]

    lines += [
        "",
        "## Data Provenance",
        "",
        f"Training dataset fingerprint (SHA-256): `{fingerprint}`",
        "",
        "_This fingerprint is computed from the actual training Parquet file "
        "at training time and cannot be supplied by the caller._",
    ]

    return "\n".join(lines) + "\n"


def _render_html(metadata: dict) -> str:
    md = _render_markdown(metadata)
    try:
        import markdown as _md

        return _md.markdown(md, extensions=["tables"])
    except ImportError:
        # Minimal fallback: wrap each line in <p>
        escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = "".join(f"<p>{line}</p>\n" for line in escaped.splitlines())
        return f"<!DOCTYPE html><html><body>\n{body}</body></html>\n"


def generate_model_card(
    model_metadata_path: str,
    output_path: str,
    fmt: Literal["markdown", "html"] = "markdown",
) -> str:
    """Read *model_metadata_path*, validate it, and render a model card.

    Args:
        model_metadata_path: Path to the JSON metadata file produced at training time.
        output_path: Destination path for the rendered card.
        fmt: Output format — ``"markdown"`` (default) or ``"html"``.

    Returns:
        The rendered card as a string (also written to *output_path*).

    Raises:
        MetadataValidationError: When required metadata fields are absent.
    """
    with open(model_metadata_path) as f:
        metadata = json.load(f)

    _validate_metadata(metadata)

    if fmt == "html":
        content = _render_html(metadata)
    else:
        content = _render_markdown(metadata)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    return content
