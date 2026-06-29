"""Natural-language narrative summaries for forensic reports.

Converts the numeric scores and feature attributions in a forensic report
into a short, human-readable paragraph for non-technical compliance
officers and regulators, via Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import config
from reporting.feature_labels import label_for

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_MAX_WORDS = 300

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["j2", "html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _top_features(report: dict[str, Any], markdown: bool, n: int = 3) -> list[dict[str, Any]]:
    raw = report.get("top_shap_features") or []
    ranked = sorted(raw, key=lambda f: abs(f.get("contribution", 0) or 0), reverse=True)[:n]

    results = []
    for f in ranked:
        if "feature" not in f:
            continue
        label = label_for(f["feature"])
        results.append({**f, "label": label, "display": f"**{label}**" if markdown else label})
    return results


def _normalize_benford(benford: Any) -> dict[str, float | None] | None:
    """Find the violating window (or flat summary) in a benford_analysis
    field, supporting both `{window_hours: {metrics}}` and a flat
    `{chi_square/chi2, p_value/p}` summary dict. Returns None if absent or
    no window is flagged non-conforming."""
    if not benford or not isinstance(benford, dict):
        return None

    if any(k in benford for k in ("chi_square", "chi2", "p_value", "p")):
        candidate = benford
    else:
        flagged = [m for m in benford.values() if isinstance(m, dict) and m.get("mad_nonconforming")]
        if not flagged:
            return None
        candidate = flagged[0]

    chi_square = candidate.get("chi_square", candidate.get("chi2"))
    p_value = candidate.get("p_value", candidate.get("p"))
    return {"chi_square": chi_square, "p_value": p_value}


def build_narrative(report_dict: dict[str, Any]) -> str:
    """Render a natural-language narrative summary for a forensic report.

    Flush trigger: this is a pure render -- it runs synchronously over a
    single, already-complete `report_dict` rather than buffering anything,
    so there is no separate "flush" step; the whole narrative is produced
    in one call.

    Ordering behaviour: one paragraph is rendered per applicable signal, in
    a fixed priority order -- ring detection, then Benford violation, then
    velocity anomaly -- so a wallet flagged on multiple signals reads as
    "most structurally significant first". If no signal applies, a single
    `low_confidence.j2` paragraph is rendered instead.

    Evidence merging: each paragraph independently references the same
    top-3 SHAP features (by contribution magnitude, plain-English label via
    `reporting.feature_labels.label_for`), so contributing factors are
    repeated rather than deduplicated across paragraphs -- this keeps each
    paragraph self-contained and correct on its own. Optional fields
    missing from `report_dict` (e.g. no SHAP values, no ring/benford/
    velocity signal) are omitted from the text entirely rather than
    rendered as `None`. The combined text is capped at 300 words to fit
    regulatory report page constraints; any excess is truncated at a word
    boundary.

    Args:
        report_dict: forensic report as a dict (e.g.
            `ForensicReport.to_dict()`), optionally including
            `narrative_format` ("plain_text" or "markdown"; defaults to
            `config.REPORT_NARRATIVE_FORMAT`).

    Returns:
        The rendered narrative, <= 300 words.
    """
    fmt = report_dict.get("narrative_format", config.REPORT_NARRATIVE_FORMAT)
    markdown = fmt == "markdown"
    top_features = _top_features(report_dict, markdown)

    paragraphs: list[str] = []

    ring = report_dict.get("ring_detection")
    if ring:
        paragraphs.append(
            _env.get_template("ring_detected.j2")
            .render(report=report_dict, ring=ring, top_features=top_features)
            .strip()
        )

    benford = _normalize_benford(report_dict.get("benford_analysis"))
    if benford:
        paragraphs.append(
            _env.get_template("benford_violation.j2")
            .render(report=report_dict, benford=benford, top_features=top_features)
            .strip()
        )

    velocity = report_dict.get("velocity_anomaly")
    if velocity:
        paragraphs.append(
            _env.get_template("velocity_anomaly.j2")
            .render(report=report_dict, velocity=velocity, top_features=top_features)
            .strip()
        )

    if not paragraphs:
        paragraphs.append(
            _env.get_template("low_confidence.j2")
            .render(report=report_dict, top_features=top_features)
            .strip()
        )

    text = " ".join(paragraphs)
    words = text.split()
    if len(words) > _MAX_WORDS:
        text = " ".join(words[:_MAX_WORDS])
    return text
