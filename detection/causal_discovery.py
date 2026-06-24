"""Causal discovery module using the PC algorithm to identify causal features of wash trading."""

import json
import os
import networkx as nx
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz


class WashTradeCausalDiscovery:
    def __init__(self):
        self.dag = nx.DiGraph()

    def fit(self, feature_df: pd.DataFrame, alpha: float = 0.05) -> nx.DiGraph:
        """Fit the PC causal discovery algorithm on the features and label.

        Returns a networkx DiGraph.
        """
        # Ensure all columns are numeric
        numeric_cols = [col for col in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[col])]
        df_filtered = feature_df[numeric_cols].dropna()

        # Run PC causal discovery
        cg = pc(df_filtered.values, alpha=alpha, indep_test=fisherz, node_names=list(df_filtered.columns))
        
        self.dag = self._to_networkx(cg.G)
        return self.dag

    def _to_networkx(self, cg_graph) -> nx.DiGraph:
        """Convert causal-learn GeneralGraph to networkx DiGraph."""
        g = nx.DiGraph()
        
        # Add all nodes
        for node in cg_graph.get_nodes():
            g.add_node(node.get_name())
            
        # Add edges
        for edge in cg_graph.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            ep1 = edge.get_endpoint1().name  # 'TAIL' or 'ARROW'
            ep2 = edge.get_endpoint2().name  # 'TAIL' or 'ARROW'
            
            if ep1 == 'TAIL' and ep2 == 'ARROW':
                g.add_edge(u, v)
            elif ep1 == 'ARROW' and ep2 == 'TAIL':
                g.add_edge(v, u)
            elif ep1 == 'TAIL' and ep2 == 'TAIL':
                # Undirected edge: orient consistently using node name order to guarantee acyclicity
                if u < v:
                    g.add_edge(u, v)
                else:
                    g.add_edge(v, u)
            elif ep1 == 'ARROW' and ep2 == 'ARROW':
                # Bidirectional edge: orient consistently using node name order to guarantee acyclicity
                if u < v:
                    g.add_edge(u, v)
                else:
                    g.add_edge(v, u)
                    
        return g

    def causal_features(self, label_name: str = "label") -> list[str]:
        """Features with direct causal edge to the wash-trade label."""
        if not self.dag or label_name not in self.dag:
            return []
        
        features = []
        for u in self.dag.nodes:
            if u == label_name:
                continue
            # Connect to/from label
            if self.dag.has_edge(u, label_name) or self.dag.has_edge(label_name, u):
                features.append(u)
        return sorted(list(set(features)))

    def save_dag(self, path: str) -> None:
        """Save the causal DAG structure as a JSON file."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = {
            "nodes": list(self.dag.nodes),
            "edges": list(self.dag.edges)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
