"""LLM-powered regulatory narrative generator for forensic reports (issue #216).

Transforms a :class:`detection.forensic_report.ForensicReport` into a
plain-language draft narrative suitable for filing with financial intelligence
units (FIUs), exchanges, or for inclusion in SAR/FATF Travel Rule packages.

Supports three LLM backends, selected by the ``NARRATIVE_LLM_BACKEND`` env var:
- ``openai``   — OpenAI Chat Completions API (default; requires ``OPENAI_API_KEY``)
- ``anthropic``— Anthropic Messages API (requires ``ANTHROPIC_API_KEY``)
- ``stub``     — Returns a template-filled stub (no API key needed; for tests/CI)

Usage::

    from detection.narrative_generator import NarrativeGenerator
    from detection.forensic_report import ForensicReport

    gen = NarrativeGenerator()
    narrative = gen.generate(report)
    print(narrative)

CLI::

    python -m detection.narrative_generator --report reports/forensic/report.json

Hallucination guardrails (issue #944)
-------------------------------------
Generative backends can emit factual claims (amounts, addresses, timestamps,
risk scores, trade counts) that are not supported by the structured evidence
record.  Every narrative produced by :meth:`NarrativeGenerator.generate` is
passed through :func:`validate_narrative`, which extracts numeric/address/
timestamp claims and cross-checks them against the structured report.  Any
claim that cannot be grounded in the evidence is reported as a
:class:`NarrativeViolation`.  By default ungrounded narratives raise
:class:`NarrativeGroundingError`; callers may opt into flag-only behaviour via
``NarrativeGenerator(validate=False)`` or ``generate(..., strict=False)``.

When extending the narrative path, keep factual claims templated directly from
structured fields (see ``_build_user_prompt``) and add a regression case to
``tests/test_narrative_guardrails.py`` covering any new claim type.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime

from utils.logging import get_logger

logger = get_logger(__name__)

_BACKEND_ENV = "NARRATIVE_LLM_BACKEND"
_OPENAI_MODEL_ENV = "NARRATIVE_OPENAI_MODEL"
_ANTHROPIC_MODEL_ENV = "NARRATIVE_ANTHROPIC_MODEL"
_MAX_TOKENS_ENV = "NARRATIVE_MAX_TOKENS"

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
_DEFAULT_MAX_TOKENS = 1024

_SYSTEM_PROMPT = (
    "You are a financial compliance analyst specialising in DeFi market manipulation. "
    "Your task is to write a concise, professional regulatory narrative from structured "
    "on-chain forensic data. The narrative should be suitable for submission to a "
    "financial intelligence unit (FIU) or exchange compliance team. "
    "Write in clear prose. Do not invent facts not present in the data provided."
)


# ----------------------------------------------------------------------
# Hallucination guardrails (issue #944)
# ----------------------------------------------------------------------

#: Numeric tokens (optionally signed, with thousands separators / decimals).
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
#: EVM-style hex addresses.
_ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{6,}")
#: ISO-8601-ish timestamps (date, optional time, optional Z/offset).
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)

#: Small integers that are structural prose (paragraph counts, list indices)
#: rather than factual claims, and are therefore exempt from grounding.
_STRUCTURAL_NUMBERS = {1, 2, 3, 4, 5}


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _collect_grounded_numbers(report_dict: dict) -> set[float]:
    """Collect every numeric value present in the structured evidence record."""
    grounded: set[float] = set()

    def _walk(value) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            grounded.add(float(value))
        elif isinstance(value, str):
            for token in _NUMBER_RE.findall(value):
                num = _to_float(token)
                if num is not None:
                    grounded.add(num)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item)

    _walk(report_dict)
    # Derived values that legitimately appear in narratives.
    n_trades = len(report_dict.get("trade_evidence", []) or [])
    grounded.add(float(n_trades))
    return grounded


def _collect_grounded_addresses(report_dict: dict) -> set[str]:
    """Collect every address-like string present in the structured record."""
    addresses: set[str] = set()

    def _walk(value) -> None:
        if isinstance(value, str):
            for match in _ADDRESS_RE.findall(value):
                addresses.add(match.lower())
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item)

    _walk(report_dict)
    return addresses


def _collect_grounded_timestamps(report_dict: dict) -> set[str]:
    """Collect every timestamp-like string present in the structured record."""
    timestamps: set[str] = set()

    def _walk(value) -> None:
        if isinstance(value, str):
            for match in _TIMESTAMP_RE.findall(value):
                timestamps.add(match)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item)

    _walk(report_dict)
    return timestamps


@dataclass
class NarrativeViolation:
    """A single factual claim in a narrative that is not grounded in evidence."""

    kind: str  # "number" | "address" | "timestamp"
    claim: str
    context: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.kind}] ungrounded claim {self.claim!r} in: {self.context!r}"


class NarrativeGroundingError(ValueError):
    """Raised when a generated narrative contains ungrounded factual claims."""

    def __init__(self, violations: list[NarrativeViolation]) -> None:
        self.violations = violations
        detail = "; ".join(str(v) for v in violations)
        super().__init__(f"Narrative contains {len(violations)} ungrounded claim(s): {detail}")


@dataclass
class NarrativeValidationResult:
    """Outcome of validating a narrative against its structured evidence."""

    grounded: bool
    violations: list[NarrativeViolation] = field(default_factory=list)


def validate_narrative(narrative: str, report_dict: dict) -> NarrativeValidationResult:
    """Cross-check factual claims in *narrative* against *report_dict*.

    Extracts numeric, address, and timestamp claims from the narrative and
    verifies each is present in the structured evidence record.  Returns a
    :class:`NarrativeValidationResult`; ungrounded claims are listed in
    ``violations`` and ``grounded`` is ``False`` when any are found.
    """
    grounded_numbers = _collect_grounded_numbers(report_dict)
    grounded_addresses = _collect_grounded_addresses(report_dict)
    grounded_timestamps = _collect_grounded_timestamps(report_dict)

    violations: list[NarrativeViolation] = []

    for match in _ADDRESS_RE.finditer(narrative):
        claim = match.group(0)
        if claim.lower() not in grounded_addresses:
            violations.append(
                NarrativeViolation("address", claim, _context(narrative, match.start()))
            )

    for match in _TIMESTAMP_RE.finditer(narrative):
        claim = match.group(0)
        if claim not in grounded_timestamps:
            violations.append(
                NarrativeViolation("timestamp", claim, _context(narrative, match.start()))
            )

    for match in _NUMBER_RE.finditer(narrative):
        token = match.group(0)
        num = _to_float(token)
        if num is None:
            continue
        if num in _STRUCTURAL_NUMBERS and float(num).is_integer():
            continue
        if num not in grounded_numbers:
            violations.append(
                NarrativeViolation("number", token, _context(narrative, match.start()))
            )

    return NarrativeValidationResult(grounded=not violations, violations=violations)


def _context(text: str, index: int, window: int = 40) -> str:
    """Return a short snippet of *text* around *index* for diagnostics."""
    start = max(0, index - window)
    end = min(len(text), index + window)
    return text[start:end].strip()


def _build_user_prompt(report_dict: dict) -> str:
    """Build the user-turn prompt from a forensic report dict."""
    wallet = report_dict.get("wallet", "unknown")
    pair = report_dict.get("asset_pair", "unknown")
    score = report_dict.get("risk_score", 0)
    score_lower = report_dict.get("score_lower", 0)
    score_upper = report_dict.get("score_upper", 100)
    verdict = report_dict.get("verdict", "unknown")
    generated_at = report_dict.get("generated_at", datetime.now(UTC).isoformat())

    # Top SHAP features (up to 5)
    shap_lines = []
    for feat in report_dict.get("top_shap_features", [])[:5]:
        name = feat.get("feature", "")
        val = feat.get("shap_value", feat.get("value", ""))
        desc = feat.get("description", name)
        shap_lines.append(f"  - {desc} (SHAP contribution: {val})")
    shap_block = "\n".join(shap_lines) if shap_lines else "  - (no SHAP data)"

    # Benford analysis summary (first window available)
    benford = report_dict.get("benford_analysis", {})
    benford_summary = "(no Benford data)"
    if benford:
        first_window = next(iter(benford.values()), {})
        chi2 = first_window.get("chi_square", "n/a")
        mad = first_window.get("mad", "n/a")
        nonconform = first_window.get("mad_nonconforming", False)
        benford_summary = (
            f"chi-square={chi2}, MAD={mad}, " f"non-conforming={'yes' if nonconform else 'no'}"
        )

    # Trade evidence count
    n_trades = len(report_dict.get("trade_evidence", []))

    prompt = textwrap.dedent(f"""
        Please write a regulatory narrative for the following LedgerLens forensic finding.

        --- FINDINGS ---
        Report date     : {generated_at}
        Wallet          : {wallet}
        Asset pair      : {pair}
        Risk score      : {score} / 100  (95% CI: {score_lower}–{score_upper})
        Verdict         : {verdict}
        Anomalous trades: {n_trades} selected for evidence

        Top risk factors (SHAP attributions):
        {shap_block}

        Benford's Law analysis (shortest window):
        {benford_summary}
        --- END FINDINGS ---

        Write a narrative of 3–5 paragraphs covering:
        1. Summary of the suspicious activity and the wallet involved.
        2. Key quantitative indicators (risk score, Benford metrics, top SHAP features).
        3. Nature of the evidence (number of anomalous trades, asset pair).
        4. Recommended next steps for a compliance officer.
        Do not include section headings. Write in plain prose.
        Only state facts (amounts, addresses, timestamps, scores) that appear in the
        FINDINGS block above. Do not introduce any other numbers or addresses.
    """).strip()
    return prompt


class NarrativeGenerator:
    """Generate plain-language regulatory narratives from ForensicReport objects.

    The backend is chosen at construction time from the ``NARRATIVE_LLM_BACKEND``
    environment variable (``openai``, ``anthropic``, or ``stub``).

    Generated narratives are validated against the structured evidence record
    (issue #944).  Set ``validate=False`` to disable the guardrail, or pass
    ``strict=False`` to :meth:`generate` to flag rather than raise.
    """

    def __init__(self, backend: str | None = None, validate: bool = True) -> None:
        self._backend = (backend or os.getenv(_BACKEND_ENV, "openai")).lower()
        self._max_tokens = int(os.getenv(_MAX_TOKENS_ENV, str(_DEFAULT_MAX_TOKENS)))
        self._validate = validate
        logger.info(
            "NarrativeGenerator initialised with backend=%s validate=%s",
            self._backend,
            self._validate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, report, strict: bool = True) -> str:
        """Generate a narrative for *report*.

        Args:
            report: A :class:`detection.forensic_report.ForensicReport` instance
                    **or** a plain dict with the same keys.
            strict: When ``True`` (default) an ungrounded narrative raises
                    :class:`NarrativeGroundingError`.  When ``False`` the
                    narrative is returned but violations are logged.

        Returns:
            Narrative string (plain text, no markdown headings).

        Raises:
            NarrativeGroundingError: If validation is enabled, *strict* is
                ``True``, and the narrative contains ungrounded claims.
        """
        report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        user_prompt = _build_user_prompt(report_dict)

        if self._backend == "openai":
            narrative = self._call_openai(user_prompt)
        elif self._backend == "anthropic":
            narrative = self._call_anthropic(user_prompt)
        else:
            narrative = self._stub(report_dict)

        return self._guard(narrative, report_dict, strict=strict)

    def _guard(self, narrative: str, report_dict: dict, strict: bool) -> str:
        """Validate *narrative* against *report_dict* (issue #944 guardrail)."""
        if not self._validate:
            return narrative
        result = validate_narrative(narrative, report_dict)
        if result.grounded:
            return narrative
        for violation in result.violations:
            logger.warning("Narrative grounding violation: %s", violation)
        if strict:
            raise NarrativeGroundingError(result.violations)
        return narrative

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _call_openai(self, user_prompt: str) -> str:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package required for NARRATIVE_LLM_BACKEND=openai. " "pip install openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY environment variable is not set.")

        model = os.getenv(_OPENAI_MODEL_ENV, _DEFAULT_OPENAI_MODEL)
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, user_prompt: str) -> str:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for NARRATIVE_LLM_BACKEND=anthropic. "
                "pip install anthropic"
            ) from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY environment variable is not set.")

        model = os.getenv(_ANTHROPIC_MODEL_ENV, _DEFAULT_ANTHROPIC_MODEL)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()

    @staticmethod
    def _stub(report_dict: dict) -> str:
        """Return a template-filled stub — no API required (for tests/CI)."""
        wallet = report_dict.get("wallet", "unknown")
        pair = report_dict.get("asset_pair", "unknown")
        score = report_dict.get("risk_score", 0)
        verdict = report_dict.get("verdict", "unknown")
        n_trades = len(report_dict.get("trade_evidence", []))
        return (
            f"The wallet {wallet} trading the {pair} pair was flagged with a risk "
            f"score of {score} out of 100 and a verdict of {verdict}. "
            f"A total of {n_trades} anomalous trades were selected as evidence. "
            "This activity is consistent with potential market manipulation and "
            "warrants further review by a compliance officer."
        )


def _load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(description="Generate a regulatory narrative.")
    parser.add_argument("--report", required=True, help="Path to forensic report JSON.")
    parser.add_argument("--backend", default=None, help="LLM backend override.")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable hallucination guardrails (issue #944).",
    )
    args = parser.parse_args(argv)

    report_dict = _load_report(args.report)
    generator = NarrativeGenerator(backend=args.backend, validate=not args.no_validate)
    print(generator.generate(report_dict))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
