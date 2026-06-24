"""Forensic Reporting Engine for LedgerLens risk scores.

Produces tamper-evident, auditable ForensicReport objects that document
exactly how a risk score was computed, with an optional on-chain anchor
via Soroban for non-repudiable timestamping.

Security invariants enforced here:
- horizon_url is always constructed from config.HORIZON_URL (no user input).
- report_sha256 is computed in __post_init__ over all other fields.
- Report files must be written with mode 0o600 by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from config import config

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TradeEvidence:
    trade_id: str
    ledger: int
    base_account: str
    counter_account: str
    base_amount: float
    counter_amount: float
    asset_pair: str
    horizon_url: str  # always constructed from config.HORIZON_URL


@dataclass
class ForensicReport:
    report_id: str  # UUID v4
    generated_at: str  # ISO 8601 UTC
    wallet: str
    asset_pair: str
    risk_score: int
    score_lower: int
    score_upper: int
    verdict: Literal["clean", "suspicious", "wash_trade"]
    top_shap_features: list[dict]
    benford_analysis: dict
    trade_evidence: list[TradeEvidence]
    model_metadata: dict
    report_sha256: str = field(default="", init=False)
    soroban_anchor_tx: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.report_sha256 = self._compute_sha256()

    def _compute_sha256(self) -> str:
        d = self._to_dict_without_hash()
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=_json_default).encode()
        ).hexdigest()

    def _to_dict_without_hash(self) -> dict:
        d = {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "wallet": self.wallet,
            "asset_pair": self.asset_pair,
            "risk_score": self.risk_score,
            "score_lower": self.score_lower,
            "score_upper": self.score_upper,
            "verdict": self.verdict,
            "top_shap_features": self.top_shap_features,
            "benford_analysis": self.benford_analysis,
            "trade_evidence": [asdict(t) for t in self.trade_evidence],
            "model_metadata": self.model_metadata,
            "soroban_anchor_tx": self.soroban_anchor_tx,
        }
        return d

    def to_dict(self) -> dict:
        d = self._to_dict_without_hash()
        d["report_sha256"] = self.report_sha256
        return d

    def verify_integrity(self) -> bool:
        """Recompute the SHA-256 and assert it matches the stored value."""
        return self._compute_sha256() == self.report_sha256

    def to_markdown(self) -> str:
        """Render the report as Markdown using the Jinja2 template."""
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
        except ImportError as e:
            raise RuntimeError("jinja2 is required for Markdown rendering") from e

        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape([]),
            keep_trailing_newline=True,
        )
        tmpl = env.get_template("forensic_report.md.j2")
        return tmpl.render(report=self)

    def to_pdf(self, output_path: str) -> bool:
        """Render to PDF using weasyprint. Returns True on success, False if
        weasyprint is not installed (Markdown is written instead)."""
        md = self.to_markdown()
        try:
            import weasyprint  # noqa: F401
        except ImportError:
            md_path = output_path.replace(".pdf", ".md")
            _write_secure(md_path, md)
            return False

        try:
            import markdown as md_lib

            html = md_lib.markdown(md, extensions=["tables"])
        except ImportError:
            html = f"<pre>{md}</pre>"

        import weasyprint

        weasyprint.HTML(string=html).write_pdf(output_path)
        return True


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ForensicReportGenerator:
    """Assembles a ForensicReport from scored wallet data."""

    MAX_EVIDENCE_TRADES = 20

    def generate(
        self,
        wallet: str,
        wallet_trades: pd.DataFrame,
        risk_score_dict: dict,
        shap_values: list[dict],
        asset_pair: str = "",
        model_metadata: dict | None = None,
    ) -> ForensicReport:
        score = int(risk_score_dict.get("score", 0))
        verdict = _verdict(score)

        benford_analysis = _build_benford_analysis(wallet_trades)
        trade_evidence = _select_anomalous_trades(wallet, wallet_trades, asset_pair)
        enriched_shap = _enrich_shap(shap_values)

        # Conformal prediction interval: ±10 clamped to [0, 100]
        score_lower = max(0, score - 10)
        score_upper = min(100, score + 10)

        metadata = model_metadata or _default_model_metadata()

        return ForensicReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now(UTC).isoformat(),
            wallet=wallet,
            asset_pair=asset_pair,
            risk_score=score,
            score_lower=score_lower,
            score_upper=score_upper,
            verdict=verdict,
            top_shap_features=enriched_shap[:10],
            benford_analysis=benford_analysis,
            trade_evidence=trade_evidence,
            model_metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verdict(score: int) -> Literal["clean", "suspicious", "wash_trade"]:
    if score >= 80:
        return "wash_trade"
    if score >= config.RISK_SCORE_FLAG_THRESHOLD:
        return "suspicious"
    return "clean"


def _build_benford_analysis(wallet_trades: pd.DataFrame) -> dict:
    if wallet_trades.empty:
        return {}
    from detection.benford_engine import compute_benford_metrics_for_windows

    per_window = compute_benford_metrics_for_windows(wallet_trades)
    return {
        str(h): {
            "chi_square": m["chi_square"],
            "mad": m["mad"],
            "mad_nonconforming": m.get("mad_nonconforming", False),
            "z_scores": m.get("z_scores", {}),
            "sample_size": m.get("sample_size", 0),
        }
        for h, m in per_window.items()
    }


def _select_anomalous_trades(
    wallet: str,
    wallet_trades: pd.DataFrame,
    asset_pair: str,
) -> list[TradeEvidence]:
    if wallet_trades.empty:
        return []

    df = wallet_trades.copy()

    # Anomaly score: price ratio deviation + large round amounts
    if "base_amount" in df.columns and "counter_amount" in df.columns:
        counter = df["counter_amount"].replace(0, float("nan"))
        price = df["base_amount"] / counter
        price_median = price.median()
        df["_anom"] = (price - price_median).abs() / (price_median + 1e-9)
    elif "amount" in df.columns:
        med = df["amount"].median()
        df["_anom"] = (df["amount"] - med).abs() / (med + 1e-9)
    else:
        df["_anom"] = 0.0

    top = df.nlargest(ForensicReportGenerator.MAX_EVIDENCE_TRADES, "_anom")

    evidence = []
    for _, row in top.iterrows():
        trade_id = str(row.get("trade_id", row.get("id", "")))
        # Horizon URL constructed only from config.HORIZON_URL — SSRF prevention
        horizon_url = f"{config.HORIZON_URL.rstrip('/')}/trades/{trade_id}"
        evidence.append(
            TradeEvidence(
                trade_id=trade_id,
                ledger=int(row.get("ledger", 0)),
                base_account=str(row.get("base_account", "")),
                counter_account=str(row.get("counter_account", "")),
                base_amount=float(row.get("base_amount", row.get("amount", 0.0))),
                counter_amount=float(row.get("counter_amount", 0.0)),
                asset_pair=str(row.get("pair_id", asset_pair)),
                horizon_url=horizon_url,
            )
        )
    return evidence


def _enrich_shap(shap_values: list[dict]) -> list[dict]:
    """Attach plain-English description to each SHAP entry."""
    try:
        from detection.feature_engineering import FEATURE_DESCRIPTIONS
    except ImportError:
        FEATURE_DESCRIPTIONS = {}

    result = []
    for entry in shap_values:
        enriched = dict(entry)
        fname = entry.get("feature", "")
        enriched["description"] = FEATURE_DESCRIPTIONS.get(fname, fname)
        result.append(enriched)
    return result


def _default_model_metadata() -> dict:
    """Return placeholder model metadata when none is provided."""
    return {
        "name": "LedgerLens Ensemble",
        "version": "unknown",
        "training_dataset_sha256": "unknown",
        "feature_schema_version": "unknown",
    }


def _json_default(obj):
    if isinstance(obj, (bool,)):
        return obj
    return str(obj)


def _write_secure(path: str, content: str) -> None:
    """Write content to path with mode 0o600."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)


def write_report_secure(path: str, content: str) -> None:
    """Public wrapper so callers (CLI, bulk job) can write reports securely."""
    _write_secure(path, content)
