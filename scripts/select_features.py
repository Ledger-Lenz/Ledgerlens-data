"""Recursive Feature Elimination with Cross-Validation (RFECV) for LedgerLens.

Runs RFECV on the training dataset using RandomForestClassifier as the
estimator, optimising AUC-ROC with 5-fold stratified cross-validation.

Outputs:
  - Ranked feature importance list (printed)
  - Minimal feature subset within 1% AUC of the full-feature model
  - Persists selected subset to models/selected_features.json

Usage::

    python -m scripts.select_features \\
        --data-path data/synthetic_dataset.parquet \\
        --model-dir ./models \\
        --output models/selected_features.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from detection.model_training import FEATURE_COLUMNS_EXCLUDE


def run_rfecv(
    data_path: str = "data/synthetic_dataset.parquet",
    model_dir: str = "./models",
    output: str = "models/selected_features.json",
) -> dict:
    """Run RFECV and persist the selected feature subset.

    Returns a dict with keys:
        selected_features, full_auc, subset_auc, n_features_full, n_features_selected
    """
    df = pd.read_parquet(data_path)
    label_col = "label"
    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE and c != label_col]

    X = df[feature_cols].fillna(0.0).astype(float)
    y = df[label_col].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Full-feature AUC baseline
    full_auc_scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
    full_auc = float(full_auc_scores.mean())
    print(f"Full-feature AUC (5-fold CV): {full_auc:.4f} ± {full_auc_scores.std():.4f}")

    # RFECV — step=1, AUC scoring
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        min_features_to_select=1,
    )
    rfecv.fit(X, y)

    # Feature importances from the fitted estimator
    importances = rfecv.estimator_.feature_importances_
    selected_mask = rfecv.support_
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]

    # Print ranked feature importances
    importance_pairs = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )
    print(f"\nTop 20 feature importances (from RFECV estimator):")
    for feat, imp in importance_pairs[:20]:
        marker = "✓" if feat in selected_features else " "
        print(f"  {marker} {feat:<50s} {imp:.4f}")

    # Minimal subset within 1% AUC of full set
    target_auc = full_auc - 0.01
    subset_auc_scores = cross_val_score(estimator, X[selected_features], y, cv=cv, scoring="roc_auc")
    subset_auc = float(subset_auc_scores.mean())
    print(f"\nSelected {len(selected_features)}/{len(feature_cols)} features")
    print(f"Subset AUC: {subset_auc:.4f} (threshold: {target_auc:.4f})")

    # If RFECV subset doesn't meet target, grow the subset greedily by importance
    if subset_auc < target_auc:
        print("RFECV subset below threshold — growing by importance rank...")
        sorted_features = [f for f, _ in importance_pairs]
        for i in range(len(selected_features), len(sorted_features)):
            candidate = sorted_features[:i + 1]
            auc = float(cross_val_score(estimator, X[candidate], y, cv=cv, scoring="roc_auc").mean())
            if auc >= target_auc:
                selected_features = candidate
                subset_auc = auc
                break

    result = {
        "selected_features": selected_features,
        "full_auc": round(full_auc, 6),
        "subset_auc": round(subset_auc, 6),
        "n_features_full": len(feature_cols),
        "n_features_selected": len(selected_features),
        "eliminated_features": [f for f in feature_cols if f not in selected_features],
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved selected features to {output}")
    print(f"Eliminated {result['n_features_full'] - result['n_features_selected']} features")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RFECV feature selection for LedgerLens")
    parser.add_argument("--data-path", default="data/synthetic_dataset.parquet")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--output", default="models/selected_features.json")
    args = parser.parse_args()

    run_rfecv(data_path=args.data_path, model_dir=args.model_dir, output=args.output)


if __name__ == "__main__":
    main()
