"""Forensic report structures for risk scoring and causal attribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from detection.causal_attribution import CounterfactualAttributor
from detection.model_inference import RiskScorer
from detection.shap_explainer import ShapExplainer


@dataclass(slots=True)
class CausalAttribution:
    minimal_exonerating_trades: list[str]
    counterfactual_score: int
    root_cause_wallet: str | None
    causal_chain: list[dict]
    interventional_score_if_no_wash: int


@dataclass(slots=True)
class ForensicReport:
    wallet: str
    asset_pair: str
    risk_score: dict
    shap_explanations: list[dict] = field(default_factory=list)
    causal_attribution: CausalAttribution | None = None


class ForensicReportGenerator:
    """Build a structured report for a scored wallet."""

    def __init__(self, scorer: RiskScorer | None = None, explainer: ShapExplainer | None = None):
        self._scorer = scorer or RiskScorer()
        self._explainer = explainer or ShapExplainer()

    def generate(
        self,
        wallet: str,
        asset_pair: str,
        feature_row: pd.Series,
        wallet_trades: pd.DataFrame,
        activity=None,
        orderbook_events: pd.DataFrame | None = None,
        funding_graph: nx.DiGraph | None = None,
        all_pairs_df: pd.DataFrame | None = None,
        causal: bool = False,
        top_n: int = 5,
    ) -> ForensicReport:
        risk_score = self._scorer.score(feature_row)
        shap_explanations = []
        try:
            shap_explanations = self._explainer.explain_ensemble(
                feature_row, self._scorer.models, top_n=top_n
            )
        except Exception:  # noqa: BLE001
            shap_explanations = []

        causal_attribution = None
        if causal:
            attributor = CounterfactualAttributor(self._scorer)
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
                intervention_result = attributor.interventional_score(
                    wallet, scm, {intervention_key: 0.0}
                )
                intervention_score = intervention_result["score"]

            causal_attribution = CausalAttribution(
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

        return ForensicReport(
            wallet=wallet,
            asset_pair=asset_pair,
            risk_score=risk_score,
            shap_explanations=shap_explanations,
            causal_attribution=causal_attribution,
        )
