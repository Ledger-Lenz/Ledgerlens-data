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

import csv
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# CounterfactualResult is imported lazily to avoid circular imports;
# we use TYPE_CHECKING for the type annotation only.
from typing import TYPE_CHECKING, Literal

import networkx as nx
import pandas as pd

from config import config
from detection.causal_attribution import CounterfactualAttributor
from detection.causal_forensic_report import (
    CausalAttribution,
    PropagationContributor,
    PropagationPath,
)
from detection.model_inference import RiskScorer

if TYPE_CHECKING:
    from detection.counterfactual_explainer import CounterfactualResult
    from ingestion.asset_metadata_fetcher import AssetMetadataRecord

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


FEATURE_DESCRIPTIONS = {
    "benford_mad_1h": "Benford's Law Mean Absolute Deviation over a 1-hour window.",
    "benford_mad_4h": "Benford's Law Mean Absolute Deviation over a 4-hour window.",
    "benford_mad_24h": "Benford's Law Mean Absolute Deviation over a 24-hour window.",
    "benford_mad_168h": "Benford's Law Mean Absolute Deviation over a 168-hour (7d) window.",
    "benford_mad_720h": "Benford's Law Mean Absolute Deviation over a 720-hour (30d) window.",
    "counterparty_concentration_ratio": "Fraction of total volume traded with the single most frequent counterparty.",
    "round_trip_frequency": "Frequency of round-trip trades returning assets to the originating wallet within N ledgers.",
    "self_matching_rate": "Fraction of trades that match buy/sell orders between wallets with shared funding sources.",
}

# CSV export columns (issue #97) — one row per SHAP feature, or a single
# placeholder row (empty feature/value/contribution) when there are none.
CSV_COLUMNS = ["wallet", "asset_pair", "risk_score", "feature", "shap_value", "shap_contribution"]


def write_csv_report(out_path: str, report: ForensicReport) -> None:
    """Write *report* as a CSV (one row per SHAP feature) with mode 0o600."""
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(out_path, flags, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(report.to_csv_rows())
    _apply_secure_mode(out_path)


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
    causal_attribution: CausalAttribution | None = None
    propagation_path: PropagationPath | None = None
    counterfactual_result: CounterfactualResult | None = None
    # AssetMetadataRecord.to_dict(): supply plus trust tier and staleness (issue #917)
    asset_metadata: dict | None = None

    def __post_init__(self) -> None:
        self.report_sha256 = self._compute_sha256()

    @property
    def shap_explanations(self) -> list[dict]:
        """Alias used by :mod:`detection.audit_trail`."""
        return self.top_shap_features

    def _compute_sha256(self) -> str:
        d = self._to_dict_without_hash()
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=_json_default).encode()
        ).hexdigest()

    def _to_dict_without_hash(self) -> dict:
        d: dict = {
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
        if self.causal_attribution is not None:
            d["causal_attribution"] = asdict(self.causal_attribution)
        if self.propagation_path is not None:
            d["propagation_path"] = asdict(self.propagation_path)
        if self.counterfactual_result is not None:
            d["counterfactual_result"] = self.counterfactual_result.to_dict()
        if self.asset_metadata is not None:
            d["asset_metadata"] = self.asset_metadata
        return d

    @property
    def asset_metadata_degraded(self) -> bool:
        """True when the asset metadata used is not fresh primary-source data."""
        if self.asset_metadata is None:
            return False
        return self.asset_metadata.get("trust_tier") != "primary" or bool(
            self.asset_metadata.get("is_stale")
        )

    def to_dict(self) -> dict:
        from utils.version_stamp import stamp_artifact

        d = self._to_dict_without_hash()
        d["report_sha256"] = self.report_sha256
        return stamp_artifact(d, include_git=False)

    def verify_integrity(self) -> bool:
        """Recompute the SHA-256 and assert it matches the stored value."""
        return self._compute_sha256() == self.report_sha256

    def to_csv_rows(self) -> list[dict]:
        """One CSV row per SHAP feature (issue #97).

        Emits a single placeholder row (empty feature/value/contribution)
        when there are no SHAP explanations, so the CSV always has >= 1 row.
        """
        if not self.top_shap_features:
            return [
                {
                    "wallet": self.wallet,
                    "asset_pair": self.asset_pair,
                    "risk_score": self.risk_score,
                    "feature": "",
                    "shap_value": "",
                    "shap_contribution": "",
                }
            ]
        return [
            {
                "wallet": self.wallet,
                "asset_pair": self.asset_pair,
                "risk_score": self.risk_score,
                "feature": s.get("feature", ""),
                "shap_value": s.get("value", ""),
                "shap_contribution": s.get("contribution", ""),
            }
            for s in self.top_shap_features
        ]

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
        asset_pair: str = "",
        *,
        risk_score_dict: dict | None = None,
        shap_values: list[dict] | None = None,
        model_metadata: dict | None = None,
        feature_row: pd.Series | None = None,
        activity=None,
        orderbook_events: pd.DataFrame | None = None,
        funding_graph: nx.DiGraph | None = None,
        all_pairs_df: pd.DataFrame | None = None,
        causal: bool = False,
        top_n: int = 5,
        base_scores: dict[str, float] | None = None,
        co_trade_graph: nx.Graph | None = None,
        propagation_alpha: float = 0.15,
        provenance: dict[str, list[str]] | None = None,
        asset_metadata: AssetMetadataRecord | None = None,
    ) -> ForensicReport:
        if risk_score_dict is None and feature_row is not None:
            risk_score_dict = self._scorer.score(feature_row)
        if shap_values is None and feature_row is not None:
            try:
                shap_values = self._explainer.explain_ensemble(
                    feature_row, self._scorer.models, top_n=top_n
                )
            except Exception:  # noqa: BLE001
                shap_values = []

        score_dict = risk_score_dict or {"score": 0}
        score = int(score_dict.get("score", 0))
        verdict = _verdict(score)

        benford_analysis = _build_benford_analysis(wallet_trades)
        trade_evidence = _select_anomalous_trades(wallet, wallet_trades, asset_pair)
        enriched_shap = _enrich_shap(shap_values or [], provenance=provenance)

        score_lower = max(0, score - 10)
        score_upper = min(100, score + 10)
        metadata = model_metadata or _default_model_metadata()

        causal_attribution = None
        if causal and feature_row is not None:
            causal_attribution = _build_causal_attribution(
                self._scorer,
                wallet,
                feature_row,
                wallet_trades,
                activity=activity,
                orderbook_events=orderbook_events,
                funding_graph=funding_graph,
                all_pairs_df=all_pairs_df,
            )

        propagation_path: PropagationPath | None = None
        if base_scores is not None and funding_graph is not None:
            propagation_path = _build_propagation_path(
                wallet,
                base_scores,
                funding_graph,
                co_trade_graph=co_trade_graph,
                propagation_alpha=propagation_alpha,
                top_n=top_n,
            )

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
            causal_attribution=causal_attribution,
            propagation_path=propagation_path,
            asset_metadata=asset_metadata.to_dict() if asset_metadata is not None else None,
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


def _enrich_shap(
    shap_values: list[dict],
    provenance: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Attach plain-English description to each SHAP entry.

    When *provenance* is provided (a feature_name → [trade_id, ...] mapping),
    the top-5 entries also receive an ``evidence_links`` list of Horizon
    explorer URLs, one per contributing trade ID.
    """
    try:
        from detection.feature_engineering import FEATURE_DESCRIPTIONS
    except ImportError:
        FEATURE_DESCRIPTIONS = {}

    result = []
    for i, entry in enumerate(shap_values):
        enriched = dict(entry)
        fname = entry.get("feature", "")
        enriched["description"] = FEATURE_DESCRIPTIONS.get(fname, fname)
        if provenance is not None and i < 5:
            trade_ids = provenance.get(fname, [])
            horizon_base = config.HORIZON_URL.rstrip("/")
            enriched["evidence_links"] = [f"{horizon_base}/trades/{tid}" for tid in trade_ids]
        result.append(enriched)
    return result


def _default_model_metadata() -> dict:
    return {
        "name": "LedgerLens Ensemble",
        "version": "unknown",
        "training_dataset_sha256": "unknown",
        "feature_schema_version": "unknown",
    }


def _build_causal_attribution(
    scorer: RiskScorer,
    wallet: str,
    feature_row: pd.Series,
    wallet_trades: pd.DataFrame,
    *,
    activity=None,
    orderbook_events: pd.DataFrame | None = None,
    funding_graph: nx.DiGraph | None = None,
    all_pairs_df: pd.DataFrame | None = None,
) -> CausalAttribution:
    attributor = CounterfactualAttributor(scorer)
    minimal_set = (
        attributor.minimal_exonerating_set(
            wallet,
            wallet_trades,
            activity=activity,
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df,
        )
        or []
    )
    counterfactual = attributor.counterfactual_score(
        wallet,
        wallet_trades,
        minimal_set,
        activity=activity,
        orderbook_events=orderbook_events,
        funding_graph=funding_graph,
        all_pairs_df=all_pairs_df,
    )
    scm = attributor.build_scm(
        wallet,
        wallet_trades,
        activities=[activity] if activity is not None else None,
        orderbook_events=orderbook_events,
        funding_graph=funding_graph,
        all_pairs_df=all_pairs_df,
    )
    intervention_score = counterfactual["counterfactual_score"]
    intervention_key = next(
        (name for name in feature_row.index if name == "benford_chi_square_24h"),
        next(
            (name for name in feature_row.index if name.startswith("benford_chi_square_")),
            None,
        ),
    )
    if intervention_key is not None:
        intervention_result = attributor.interventional_score(wallet, scm, {intervention_key: 0.0})
        intervention_score = intervention_result["score"]

    return CausalAttribution(
        minimal_exonerating_trades=minimal_set,
        counterfactual_score=counterfactual["counterfactual_score"],
        root_cause_wallet=attributor.root_cause_wallet(
            wallet,
            wallet_trades,
            funding_graph,
            activity=activity,
            orderbook_events=orderbook_events,
            all_pairs_df=all_pairs_df,
        ),
        causal_chain=attributor.causal_chain(wallet, funding_graph),
        interventional_score_if_no_wash=intervention_score,
    )


def _build_propagation_path(
    wallet: str,
    base_scores: dict[str, float],
    funding_graph: nx.DiGraph,
    *,
    co_trade_graph: nx.Graph | None = None,
    propagation_alpha: float = 0.15,
    top_n: int = 5,
) -> PropagationPath | None:
    from detection.risk_propagation import propagate_risk_scores, propagation_attribution

    propagated_scores = propagate_risk_scores(
        base_scores,
        funding_graph,
        co_trade_graph=co_trade_graph,
        alpha=propagation_alpha,
    )
    wallet_propagated = propagated_scores.get(wallet, 0.0)
    if wallet_propagated <= 0.0:
        return None

    raw_contributors = propagation_attribution(
        wallet,
        base_scores,
        funding_graph,
        co_trade_graph=co_trade_graph,
        alpha=propagation_alpha,
        top_n=top_n,
    )
    return PropagationPath(
        propagated_risk=round(wallet_propagated, 4),
        contributors=[
            PropagationContributor(
                source_wallet=c["source_wallet"],
                base_score=c["base_score"],
                ppr_weight=c["ppr_weight"],
                contribution=c["contribution"],
                fraction=c["fraction"],
            )
            for c in raw_contributors
        ],
    )


def _json_default(obj):
    if isinstance(obj, bool):
        return obj
    return str(obj)


def _write_secure(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    _apply_secure_mode(str(p))


def _apply_secure_mode(path: str) -> None:
    """Attempt to apply 0o600 permissions; warn on platforms where it is a no-op."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError) as exc:
        logger.warning(
            "Could not set 0o600 permissions on %s: %s. "
            "The permission guarantee does not apply on this platform.",
            path,
            exc,
        )


def write_report_secure(path: str, content: str) -> None:
    """Public wrapper so callers (CLI, bulk job) can write reports securely."""
    _write_secure(path, content)
