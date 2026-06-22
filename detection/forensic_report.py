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
class PropagationContributor:
    """A single wallet that contributed to the target wallet's propagated score."""

    source_wallet: str
    base_score: float
    ppr_weight: float
    contribution: float
    fraction: float  # share of total propagated score from this source


@dataclass(slots=True)
class PropagationPath:
    """Propagation attribution section of a :class:`ForensicReport`."""

    propagated_risk: float
    contributors: list[PropagationContributor] = field(default_factory=list)


@dataclass(slots=True)
class ForensicReport:
    wallet: str
    asset_pair: str
    risk_score: dict
    shap_explanations: list[dict] = field(default_factory=list)
    causal_attribution: CausalAttribution | None = None
    propagation_path: PropagationPath | None = None


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
        # Propagation inputs — both required for propagation_path to be populated
        base_scores: dict[str, float] | None = None,
        co_trade_graph: nx.Graph | None = None,
        propagation_alpha: float = 0.15,
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

        # ------------------------------------------------------------------
        # Propagation path — only computed when base_scores and a graph are
        # supplied, and only when the wallet has a non-zero propagated score.
        # ------------------------------------------------------------------
        propagation_path: PropagationPath | None = None
        if base_scores is not None and funding_graph is not None:
            from detection.risk_propagation import (
                propagate_risk_scores,
                propagation_attribution,
            )

            propagated_scores = propagate_risk_scores(
                base_scores,
                funding_graph,
                co_trade_graph=co_trade_graph,
                alpha=propagation_alpha,
            )
            wallet_propagated = propagated_scores.get(wallet, 0.0)

            if wallet_propagated > 0.0:
                raw_contributors = propagation_attribution(
                    wallet,
                    base_scores,
                    funding_graph,
                    co_trade_graph=co_trade_graph,
                    alpha=propagation_alpha,
                    top_n=top_n,
                )
                propagation_path = PropagationPath(
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

        return ForensicReport(
            wallet=wallet,
            asset_pair=asset_pair,
            risk_score=risk_score,
            shap_explanations=shap_explanations,
            causal_attribution=causal_attribution,
            propagation_path=propagation_path,
        )
