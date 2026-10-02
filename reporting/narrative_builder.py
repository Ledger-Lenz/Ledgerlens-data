"""Natural-language narrative summaries for forensic reports.

Converts the numeric scores and feature attributions in a forensic report
into a short, human-readable paragraph for non-technical compliance
officers and regulators, via Jinja2 templates.

Hallucination guardrails
------------------------
Narratives are rendered from Jinja2 templates that interpolate values
directly out of the structured ``report_dict`` -- there is no free-form
generative model in this path, so every factual claim is templated from
structured evidence by construction. To keep that invariant true as
future templates change, :func:`build_narrative` runs a post-generation
validation step (:func:`validate_narrative`) that extracts every factual
claim (amounts, addresses, timestamps, counts, percentages) from the
rendered text and cross-checks it against the structured evidence record.
Any claim not directly supported by the evidence is flagged and the
narrative is rejected (``NarrativeGroundingError``) rather than emitted.
"""

from __future__ import annotations

import re
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


class NarrativeGroundingError(ValueError):
    """Raised when a rendered narrative contains an ungrounded factual claim."""

    def __init__(self, ungrounded: list[str]) -> None:
        self.ungrounded = ungrounded
        super().__init__(
            "narrative contains factual claims not supported by the "
            f"structured evidence: {ungrounded}"
        )


# Factual-claim patterns: numeric amounts, percentages, addresses, and
# ISO-8601 timestamps. These are the claim types that must be traceable to
# the structured evidence record.
_AMOUNT_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_PERCENT_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?%")
_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"
)


def _iter_evidence_values(evidence: Any):
    """Yield every scalar leaf value found in the structured evidence."""
    if isinstance(evidence, dict):
        for value in evidence.values():
            yield from _iter_evidence_values(value)
    elif isinstance(evidence, (list, tuple, set)):
        for value in evidence:
            yield from _iter_evidence_values(value)
    else:
        yield evidence


def _grounded_tokens(evidence: dict[str, Any]) -> set[str]:
    """Build the set of normalized string tokens supported by the evidence.

    Each scalar leaf is normalized (lowercased, thousands separators
    stripped) and added both as a whole token and, for numeric values, in
    common renderings (int/float forms) so templated values match.
    """
    tokens: set[str] = set()
    for value in _iter_evidence_values(evidence):
        if value is None or isinstance(value, bool):
            continue
        text = str(value).strip().lower()
        if not text:
            continue
        tokens.add(text)
        tokens.add(text.replace(",", ""))
        if isinstance(value, (int, float)):
            tokens.add(str(value))
            tokens.add(f"{value:g}")
            tokens.add(f"{value:,.0f}")
    return tokens


def _extract_claims(text: str) -> list[str]:
    """Extract factual claims (amounts, percentages, addresses, timestamps)."""
    claims: list[str] = []
    for pattern in (_TIMESTAMP_RE, _ADDRESS_RE, _PERCENT_RE, _AMOUNT_RE):
        claims.extend(match.group(0) for match in pattern.finditer(text))
    return claims


def validate_narrative(
    narrative: str, evidence: dict[str, Any]
) -> list[str]:
    """Cross-check every factual claim in ``narrative`` against ``evidence``.

    Returns the list of ungrounded claims (empty when the narrative is
    fully grounded). A claim is grounded when its normalized form appears
    among the scalar values of the structured evidence record.
    """
    grounded = _grounded_tokens(evidence)
    ungrounded: list[str] = []
    for claim in _extract_claims(narrative):
        normalized = claim.strip().lower().replace(",", "")
        if normalized in grounded or claim.strip().lower() in grounded:
            continue
        ungrounded.append(claim)
    return ungrounded


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
        flagged = [
            m for m in benford.values() if isinstance(m, dict) and m.get("mad_nonconforming")
        ]
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

    Hallucination guardrail: after rendering, the narrative is validated
    against `report_dict` via :func:`validate_narrative`. Any factual claim
    (amount, address, timestamp, count, percentage) not directly supported
    by the structured evidence raises :class:`NarrativeGroundingError`
    instead of being emitted.

    Args:
        report_dict: forensic report as a dict (e.g.
            `ForensicReport.to_dict()`), optionally including
            `narrative_format` ("plain_text" or "markdown"; defaults to
            `config.REPORT_NARRATIVE_FORMAT`).

    Returns:
        The rendered narrative, <= 300 words.

    Raises:
        NarrativeGroundingError: if the narrative contains a factual claim
            not supported by the structured evidence record.
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

    ungrounded = validate_narrative(text, report_dict)
    if ungrounded:
        raise NarrativeGroundingError(ungrounded)
    return text
