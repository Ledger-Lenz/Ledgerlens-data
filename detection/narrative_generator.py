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
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timezone

from config import config
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


def _build_user_prompt(report_dict: dict) -> str:
    """Build the user-turn prompt from a forensic report dict."""
    wallet = report_dict.get("wallet", "unknown")
    pair = report_dict.get("asset_pair", "unknown")
    score = report_dict.get("risk_score", 0)
    score_lower = report_dict.get("score_lower", 0)
    score_upper = report_dict.get("score_upper", 100)
    verdict = report_dict.get("verdict", "unknown")
    generated_at = report_dict.get("generated_at", datetime.now(timezone.utc).isoformat())

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
            f"chi-square={chi2}, MAD={mad}, "
            f"non-conforming={'yes' if nonconform else 'no'}"
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
    """).strip()
    return prompt


class NarrativeGenerator:
    """Generate plain-language regulatory narratives from ForensicReport objects.

    The backend is chosen at construction time from the ``NARRATIVE_LLM_BACKEND``
    environment variable (``openai``, ``anthropic``, or ``stub``).
    """

    def __init__(self, backend: str | None = None) -> None:
        self._backend = (backend or os.getenv(_BACKEND_ENV, "openai")).lower()
        self._max_tokens = int(os.getenv(_MAX_TOKENS_ENV, str(_DEFAULT_MAX_TOKENS)))
        logger.info("NarrativeGenerator initialised with backend=%s", self._backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, report) -> str:
        """Generate a narrative for *report*.

        Args:
            report: A :class:`detection.forensic_report.ForensicReport` instance
                    **or** a plain dict with the same keys.

        Returns:
            Narrative string (plain text, no markdown headings).
        """
        report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        user_prompt = _build_user_prompt(report_dict)

        if self._backend == "openai":
            return self._call_openai(user_prompt)
        if self._backend == "anthropic":
            return self._call_anthropic(user_prompt)
        return self._stub(report_dict)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _call_openai(self, user_prompt: str) -> str:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package required for NARRATIVE_LLM_BACKEND=openai. "
                "pip install openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

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
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

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
            f"[STUB NARRATIVE] Wallet {wallet} trading pair {pair} received a "
            f"LedgerLens risk score of {score}/100 (verdict: {verdict}). "
            f"{n_trades} anomalous trade(s) were identified as supporting evidence. "
            "This stub narrative was generated without an LLM backend. "
            "Set NARRATIVE_LLM_BACKEND=openai or anthropic and supply the "
            "corresponding API key to generate a real narrative."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a regulatory narrative from a LedgerLens forensic report JSON."
    )
    parser.add_argument(
        "--report", required=True, help="Path to forensic report JSON file."
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="LLM backend: openai | anthropic | stub (overrides NARRATIVE_LLM_BACKEND).",
    )
    parser.add_argument(
        "--output", default=None, help="Write narrative to this file instead of stdout."
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as fh:
        report_dict = json.load(fh)

    gen = NarrativeGenerator(backend=args.backend)
    narrative = gen.generate(report_dict)

    if args.output:
        from detection.forensic_report import write_report_secure
        write_report_secure(args.output, narrative)
        print(f"Narrative written to {args.output}")
    else:
        print(narrative)


if __name__ == "__main__":
    _cli()
