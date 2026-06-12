"""Train the LedgerLens ensemble classifiers (RF, XGBoost, LightGBM).

TODO (not yet implemented):
  - Load labelled wash-trade dataset (see roadmap: "Open dataset release")
  - Apply SMOTE to address class imbalance
  - Hyperparameter search per model
  - Evaluate with AUC-ROC, PR-AUC, F1 and log to a metrics report
  - Persist trained models to `config.MODEL_DIR`
"""

import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import config

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}

FEATURE_COLUMNS_EXCLUDE = {"wallet", "label"}


def load_training_data(path: str) -> pd.DataFrame:
    """Load a labelled feature matrix (output of `build_feature_matrix` plus
    a `label` column: 1 = wash trading, 0 = legitimate)."""
    return pd.read_parquet(path)


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    return df[feature_cols], df["label"]


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train all models in `MODEL_REGISTRY` and return fitted estimators
    plus evaluation metrics.

    Returns:
        {
          "random_forest": {"model": ..., "metrics": {...}},
          "xgboost": {...},
          "lightgbm": {...},
        }
    """
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    results = {}
    for name, model_cls in MODEL_REGISTRY.items():
        model = model_cls(random_state=random_state)
        model.fit(X_train_res, y_train_res)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, probs)

        results[name] = {
            "model": model,
            "metrics": {
                "auc_roc": float(roc_auc_score(y_test, probs)),
                "pr_auc": float(auc(recall, precision)),
                "f1": float(f1_score(y_test, preds)),
            },
        }

    return results


def save_models(results: dict, model_dir: str | None = None) -> None:
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    for name, result in results.items():
        joblib.dump(result["model"], os.path.join(model_dir, f"{name}.joblib"))


if __name__ == "__main__":
    raise SystemExit(
        "model_training.py is a library module — wire it up to a labelled "
        "dataset path before running as a script."
    )
