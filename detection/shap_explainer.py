"""SHAP-based interpretability for risk scores.

Wraps each trained ensemble model with a SHAP explainer so that every
risk score can be accompanied by a per-feature attribution, surfaced via
the API for auditors and end-users.

TODO (not yet implemented):
  - Cache explainers per model (TreeExplainer construction is not free)
  - Aggregate per-model SHAP values into a single ensemble explanation
  - Serialize top-N feature contributions for the API response shape
"""

import pandas as pd
import shap

from detection.model_training import FEATURE_COLUMNS_EXCLUDE


class ShapExplainer:
    """Produces SHAP value explanations for a trained model."""

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, feature_row: pd.Series, top_n: int = 5) -> list[dict]:
        """Return the top `top_n` features driving this wallet's score.

        Each entry: {"feature": str, "contribution": float, "value": float}
        """
        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T

        shap_values = self.explainer.shap_values(X)
        # Binary classifiers may return a list [class_0, class_1]
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contributions = sorted(
            zip(feature_cols, values, X.iloc[0].values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        return [
            {"feature": name, "contribution": float(value), "value": float(raw)}
            for name, value, raw in contributions
        ]
