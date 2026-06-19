"""Counterfactual and causal attribution helpers for forensic analysis.

This module keeps the implementation deterministic and local to the existing
feature pipeline. It reuses feature engineering and the ensemble scorer, then
layers lightweight causal heuristics on top so investigators can ask concrete
what-if questions without maintaining a second scoring stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from config import config
from detection.feature_engineering import build_feature_vector
from detection.model_inference import RiskScorer

CAUSAL_DISCLAIMER = (
    "The presence of a minimal exonerating set does not indicate the wallet is innocent; "
    "it indicates which specific trades are most anomalous."
)


def _jsonable(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _other_party(trade_row: pd.Series, wallet: str) -> str | None:
    if trade_row.get("base_account") == wallet:
        return trade_row.get("counter_account")
    if trade_row.get("counter_account") == wallet:
        return trade_row.get("base_account")
    return None


def _shared_funding_similarity(wallet_a: str, wallet_b: str, graph: nx.DiGraph | None) -> float:
    if graph is None or wallet_a not in graph or wallet_b not in graph:
        return 0.0

    ancestors_a = nx.ancestors(graph, wallet_a)
    ancestors_b = nx.ancestors(graph, wallet_b)
    if not ancestors_a or not ancestors_b:
        return 0.0

    union = ancestors_a | ancestors_b
    if not union:
        return 0.0

    return float(len(ancestors_a & ancestors_b) / len(union))


def _feature_dependency_graph(feature_names: list[str]) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(feature_names)

    for hours in config.BENFORD_WINDOWS_HOURS:
        chi = f"benford_chi_square_{hours}h"
        mad = f"benford_mad_{hours}h"
        z_max = f"benford_z_max_{hours}h"
        if chi in graph:
            graph.add_edge(chi, mad)
            graph.add_edge(chi, z_max)

    if "counterparty_concentration_ratio" in graph and "round_trip_frequency" in graph:
        graph.add_edge("counterparty_concentration_ratio", "round_trip_frequency")
    if "round_trip_frequency" in graph and "net_roundtrip_ratio" in graph:
        graph.add_edge("round_trip_frequency", "net_roundtrip_ratio")
    if "benford_chi_square_24h" in graph and "net_roundtrip_ratio" in graph:
        graph.add_edge("benford_chi_square_24h", "net_roundtrip_ratio")
    if "round_trip_frequency" in graph and "self_matching_rate" in graph:
        graph.add_edge("round_trip_frequency", "self_matching_rate")
    if "intra_minute_clustering" in graph and "volume_spike_frequency" in graph:
        graph.add_edge("intra_minute_clustering", "volume_spike_frequency")
    if "funding_source_similarity" in graph and "network_centrality" in graph:
        graph.add_edge("funding_source_similarity", "network_centrality")
    if "cross_pair_trade_synchrony" in graph:
        for child in (
            "net_asset_flow_deviation",
            "cross_pair_volume_correlation",
            "pair_diversity_score",
        ):
            if child in graph:
                graph.add_edge("cross_pair_trade_synchrony", child)
    if "benford_mad_24h" in graph and "cross_pair_mad_std" in graph:
        graph.add_edge("benford_mad_24h", "cross_pair_mad_std")

    return graph


def _recompute_feature(name: str, values: dict[str, float]) -> float:
    if name.startswith("benford_mad_"):
        chi_name = name.replace("benford_mad_", "benford_chi_square_")
        chi_value = float(values.get(chi_name, values.get(name, 0.0)))
        current = float(values.get(name, 0.0))
        if chi_value <= 0:
            return 0.0
        return float(min(current, chi_value / 1000.0))

    if name.startswith("benford_z_max_"):
        chi_name = name.replace("benford_z_max_", "benford_chi_square_")
        chi_value = float(values.get(chi_name, values.get(name, 0.0)))
        current = float(values.get(name, 0.0))
        if chi_value <= 0:
            return 0.0
        return float(min(current, chi_value / 10.0))

    if name == "round_trip_frequency":
        concentration = float(values.get("counterparty_concentration_ratio", 0.0))
        return float(min(float(values.get(name, 0.0)), concentration))

    if name == "self_matching_rate":
        return float(values.get("round_trip_frequency", values.get(name, 0.0)))

    if name == "net_roundtrip_ratio":
        round_trip_frequency = float(values.get("round_trip_frequency", 0.0))
        return float(min(float(values.get(name, 0.0)), round_trip_frequency * 0.5))

    if name == "volume_spike_frequency":
        clustering = float(values.get("intra_minute_clustering", 0.0))
        return float(min(float(values.get(name, 0.0)), clustering))

    if name == "network_centrality":
        similarity = float(values.get("funding_source_similarity", 0.0))
        return float(min(float(values.get(name, 0.0)), similarity))

    if name == "net_asset_flow_deviation":
        synchrony = float(values.get("cross_pair_trade_synchrony", 0.0))
        return float(max(0.0, min(float(values.get(name, 0.0)), 1.0 - synchrony)))

    if name == "cross_pair_volume_correlation":
        synchrony = float(values.get("cross_pair_trade_synchrony", 0.0))
        return float(max(-1.0, min(float(values.get(name, 0.0)), 1.0 - synchrony)))

    if name == "pair_diversity_score":
        synchrony = float(values.get("cross_pair_trade_synchrony", 0.0))
        return float(max(0.0, min(float(values.get(name, 0.0)), 1.0 - synchrony)))

    if name == "cross_pair_mad_std":
        mad_values = [float(v) for key, v in values.items() if key.startswith("benford_mad_")]
        if not mad_values:
            return float(values.get(name, 0.0))
        return float(min(float(values.get(name, 0.0)), sum(mad_values) / len(mad_values)))

    return float(values.get(name, 0.0))


@dataclass(slots=True)
class CausalAttributionResult:
    minimal_exonerating_trades: list[str]
    counterfactual_score: int
    root_cause_wallet: str | None
    causal_chain: list[dict]
    interventional_score_if_no_wash: int


class CounterfactualAttributor:
    """Compute counterfactual risk scores and lightweight causal attributions."""

    def __init__(self, scorer: RiskScorer | None = None):
        self._scorer = scorer or RiskScorer()

    def _score_feature_row(self, feature_row: pd.Series) -> dict:
        return self._scorer.score(feature_row)

    def _feature_row(
        self,
        wallet: str,
        wallet_trades: pd.DataFrame,
        activity=None,
        orderbook_events: pd.DataFrame | None = None,
        funding_graph: nx.DiGraph | None = None,
        all_pairs_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        feature_vector = build_feature_vector(
            wallet,
            wallet_trades,
            activity=activity,
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df,
        )
        return pd.Series(feature_vector)

    def counterfactual_score(
        self,
        wallet: str,
        wallet_trades: pd.DataFrame,
        remove_trade_ids: list[str],
        activity=None,
        orderbook_events: pd.DataFrame | None = None,
        funding_graph: nx.DiGraph | None = None,
        all_pairs_df: pd.DataFrame | None = None,
    ) -> dict:
        original_row = self._feature_row(
            wallet,
            wallet_trades,
            activity=activity,
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df,
        )
        original_score = self._score_feature_row(original_row)

        if wallet_trades.empty:
            return {
                "original_score": int(original_score["score"]),
                "counterfactual_score": 0,
                "score_delta": int(original_score["score"]),
                "features_changed": {},
            }

        filtered_trades = wallet_trades[
            ~wallet_trades["trade_id"].astype(str).isin(remove_trade_ids)
        ].copy()
        counterfactual_row = self._feature_row(
            wallet,
            filtered_trades,
            activity=activity,
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df,
        )
        counterfactual_score = self._score_feature_row(counterfactual_row)["score"]
        counterfactual_score = min(int(original_score["score"]), int(counterfactual_score))

        features_changed: dict[str, dict[str, object]] = {}
        for feature_name in original_row.index:
            original_value = _jsonable(original_row[feature_name])
            counterfactual_value = _jsonable(counterfactual_row.get(feature_name))
            if original_value != counterfactual_value:
                delta = None
                if isinstance(original_value, (int, float)) and isinstance(
                    counterfactual_value, (int, float)
                ):
                    delta = float(counterfactual_value) - float(original_value)
                features_changed[feature_name] = {
                    "original": original_value,
                    "counterfactual": counterfactual_value,
                    "delta": delta,
                }

        return {
            "original_score": int(original_score["score"]),
            "counterfactual_score": int(counterfactual_score),
            "score_delta": int(original_score["score"] - counterfactual_score),
            "features_changed": features_changed,
        }

    def minimal_exonerating_set(
        self,
        wallet: str,
        wallet_trades: pd.DataFrame,
        threshold: int = 50,
        activity=None,
        orderbook_events: pd.DataFrame | None = None,
        funding_graph: nx.DiGraph | None = None,
        all_pairs_df: pd.DataFrame | None = None,
    ) -> list[str] | None:
        if wallet_trades.empty:
            return []

        remaining = wallet_trades.copy()
        removed_trade_ids: list[str] = []
        current_score = self.counterfactual_score(
            wallet,
            remaining,
            removed_trade_ids,
            activity=activity,
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df,
        )["counterfactual_score"]

        if current_score < threshold:
            return []

        max_steps = min(20, len(remaining))
        for _ in range(max_steps):
            best_choice: tuple[int, str] | None = None

            for trade_id in sorted(remaining["trade_id"].astype(str).unique()):
                candidate_removed = removed_trade_ids + [trade_id]
                candidate_score = self.counterfactual_score(
                    wallet,
                    remaining,
                    candidate_removed,
                    activity=activity,
                    orderbook_events=orderbook_events,
                    funding_graph=funding_graph,
                    all_pairs_df=all_pairs_df,
                )["counterfactual_score"]

                if best_choice is None or candidate_score < best_choice[0]:
                    best_choice = (candidate_score, trade_id)

            if best_choice is None or best_choice[0] >= current_score:
                return None

            current_score, chosen_trade = best_choice
            removed_trade_ids.append(chosen_trade)
            remaining = remaining[remaining["trade_id"].astype(str) != chosen_trade].copy()

            if current_score < threshold:
                return removed_trade_ids

        return None

    def root_cause_wallet(
        self,
        wallet: str,
        wallet_trades: pd.DataFrame,
        graph: nx.DiGraph | None,
        activity=None,
        orderbook_events: pd.DataFrame | None = None,
        all_pairs_df: pd.DataFrame | None = None,
    ) -> str | None:
        if wallet_trades.empty:
            return None

        original_score = self.counterfactual_score(
            wallet,
            wallet_trades,
            [],
            activity=activity,
            orderbook_events=orderbook_events,
            funding_graph=graph,
            all_pairs_df=all_pairs_df,
        )["counterfactual_score"]

        candidates: list[tuple[int, float, int, str]] = []
        other_wallets = sorted(
            {
                other
                for _, trade_row in wallet_trades.iterrows()
                if (other := _other_party(trade_row, wallet)) is not None
            }
        )
        for candidate_wallet in other_wallets:
            remove_trade_ids = [
                str(row.trade_id)
                for row in wallet_trades.itertuples(index=False)
                if _other_party(pd.Series(row._asdict()), wallet) == candidate_wallet
            ]
            candidate_score = self.counterfactual_score(
                wallet,
                wallet_trades,
                remove_trade_ids,
                activity=activity,
                orderbook_events=orderbook_events,
                funding_graph=graph,
                all_pairs_df=all_pairs_df,
            )["counterfactual_score"]
            similarity = _shared_funding_similarity(wallet, candidate_wallet, graph)
            score_reduction = original_score - candidate_score
            candidates.append(
                (score_reduction, similarity, len(remove_trade_ids), candidate_wallet)
            )

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        return candidates[0][3]

    def causal_chain(self, wallet: str, graph: nx.DiGraph | None, max_hops: int = 3) -> list[dict]:
        if graph is None or wallet not in graph:
            return [{"wallet": wallet, "role": "primary", "hop": 0}]

        chain = [{"wallet": wallet, "role": "primary", "hop": 0}]
        current_wallet = wallet
        visited = {wallet}

        for hop in range(1, max_hops + 1):
            parents = sorted(graph.predecessors(current_wallet))
            if not parents:
                chain[-1]["role"] = "source"
                break

            next_wallet = parents[0]
            if next_wallet in visited:
                chain.append({"wallet": next_wallet, "role": "source", "hop": hop})
                break

            role = "source" if hop == max_hops else "intermediary"
            chain.append({"wallet": next_wallet, "role": role, "hop": hop})
            visited.add(next_wallet)
            current_wallet = next_wallet

            if role == "source":
                break

        return chain

    def build_scm(
        self,
        wallet: str,
        wallet_trades: pd.DataFrame,
        activities: list | None = None,
        orderbook_events: pd.DataFrame | None = None,
        funding_graph: nx.DiGraph | None = None,
        all_pairs_df: pd.DataFrame | None = None,
    ) -> dict:
        feature_row = self._feature_row(
            wallet,
            wallet_trades,
            activity=activities[0] if activities else None,
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df,
        )

        scm_graph = _feature_dependency_graph(
            [column for column in feature_row.index if column != "wallet"]
        )
        return {
            "wallet": wallet,
            "graph": scm_graph,
            "feature_values": feature_row.to_dict(),
            "wallet_trades": wallet_trades.copy(),
            "activities": list(activities or []),
            "orderbook_events": None if orderbook_events is None else orderbook_events.copy(),
            "funding_graph": funding_graph.copy() if funding_graph is not None else None,
            "all_pairs_df": None if all_pairs_df is None else all_pairs_df.copy(),
        }

    def interventional_score(self, wallet: str, scm: dict, intervention: dict) -> dict:
        values = dict(scm.get("feature_values", {}))
        values.pop("wallet", None)
        graph: nx.DiGraph = scm.get("graph", nx.DiGraph())

        changed: set[str] = set()
        for feature_name, feature_value in intervention.items():
            values[feature_name] = feature_value
            changed.add(feature_name)

        for node in nx.topological_sort(graph):
            if node in intervention:
                continue

            parents = list(graph.predecessors(node))
            if not parents:
                continue

            if any(parent in changed for parent in parents):
                original_value = values.get(node)
                recomputed = _recompute_feature(node, values)
                values[node] = recomputed
                if recomputed != original_value:
                    changed.add(node)

        if any(name.startswith("benford_chi_square_") for name in intervention):
            original_value = values.get("net_roundtrip_ratio")
            recomputed = _recompute_feature("net_roundtrip_ratio", values)
            values["net_roundtrip_ratio"] = recomputed
            if recomputed != original_value:
                changed.add("net_roundtrip_ratio")

        feature_row = pd.Series({"wallet": wallet, **values})
        score = self._score_feature_row(feature_row)

        return {
            "score": int(score["score"]),
            "benford_flag": bool(score["benford_flag"]),
            "ml_flag": bool(score["ml_flag"]),
            "confidence": int(score["confidence"]),
            "features_changed": {name: _jsonable(values.get(name)) for name in sorted(changed)},
        }
