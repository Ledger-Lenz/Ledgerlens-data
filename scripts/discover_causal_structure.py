"""Script to discover causal structure in LedgerLens data and generate visualizations and metrics."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from config import config
from detection.causal_discovery import WashTradeCausalDiscovery
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover causal structure from LedgerLens data")
    parser.add_argument("--data-path", default="data/synthetic_dataset.parquet", help="Path to features parquet file")
    parser.add_argument("--dag-output", default="models/causal_dag.json", help="Path to save the JSON DAG structure")
    parser.add_argument("--img-output", default="reports/causal_dag.png", help="Path to save the DAG visualization image")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for independence tests")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load or generate data
    if not os.path.exists(args.data_path):
        print(f"Data file '{args.data_path}' not found. Generating a synthetic dataset of 500 wallets...")
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        df = generate_synthetic_dataset(n_wallets=500, seed=args.seed)
        df.to_parquet(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)

    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Remove non-numeric string columns (like 'wallet')
    cols_to_use = [c for c in df.columns if c != "wallet" and pd.api.types.is_numeric_dtype(df[c])]
    df_fit = df[cols_to_use]

    # Run Causal Discovery
    print("Running PC causal discovery algorithm...")
    discoverer = WashTradeCausalDiscovery()
    dag = discoverer.fit(df_fit, alpha=args.alpha)

    # Extract causal features
    causal_feats = discoverer.causal_features(label_name="label")
    print(f"\nDiscovered DAG has {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges.")
    print(f"Features directly causally related to 'label': {causal_feats}")

    # Save JSON DAG
    discoverer.save_dag(args.dag_output)
    print(f"Causal DAG JSON saved to '{args.dag_output}'.")

    # Generate Visualization
    os.makedirs(os.path.dirname(args.img_output), exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Position nodes using networkx layout
    pos = nx.spring_layout(dag, seed=args.seed, k=1.5)
    
    # Highlight 'label' node
    node_colors = []
    for node in dag.nodes:
        if node == "label":
            node_colors.append("salmon")
        elif node in causal_feats:
            node_colors.append("lightblue")
        else:
            node_colors.append("lightgrey")

    nx.draw_networkx_nodes(dag, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_labels(dag, pos, font_size=8, font_family="sans-serif")
    nx.draw_networkx_edges(dag, pos, arrowstyle="->", arrowsize=15, edge_color="grey")
    
    plt.title("Causal DAG for Wash Trading Discovery (LedgerLens)", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.img_output, dpi=300)
    plt.close()
    print(f"Causal DAG visualization image saved to '{args.img_output}'.")

    # Metrics evaluation: causal features vs all features
    print("\nEvaluating classifier precision using Causal Features vs All Features...")
    
    # Split features and label
    X = df_fit.drop(columns=["label"])
    y = df_fit["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # 1. Model with all features
    model_all = RandomForestClassifier(random_state=args.seed)
    model_all.fit(X_train, y_train)
    preds_all = model_all.predict(X_test)
    prec_all = precision_score(y_test, preds_all, zero_division=0)

    # 2. Model with causal features only
    if causal_feats:
        X_train_c = X_train[causal_feats]
        X_test_c = X_test[causal_feats]
        
        model_causal = RandomForestClassifier(random_state=args.seed)
        model_causal.fit(X_train_c, y_train)
        preds_causal = model_causal.predict(X_test_c)
        prec_causal = precision_score(y_test, preds_causal, zero_division=0)
    else:
        prec_causal = 0.0

    print(f"Precision using ALL features:    {prec_all:.4f}")
    print(f"Precision using CAUSAL features: {prec_causal:.4f}")
    
    diff = prec_causal - prec_all
    print(f"Precision difference:            {diff:+.4f}")


if __name__ == "__main__":
    main()
