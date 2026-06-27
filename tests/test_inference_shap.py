import json
import os

import numpy as np
import pandas as pd
import pytest

from detection.model_inference import RiskScorer
from detection.model_training import save_models, save_training_artifacts, train_models
from detection.shap_explainer import ShapExplainer
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=60, seed=2)
    output = train_models(df, test_size=0.3, random_state=2)
    results = output["results"]
    model_dir = str(tmp_path_factory.mktemp("models"))
    save_models(results, model_dir)
    save_training_artifacts(output, "data/synthetic.parquet", model_dir)
    return output, model_dir, df


def test_risk_scorer_score_returns_contract_shape(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)

    required = {"score", "benford_flag", "ml_flag", "confidence"}
    assert required.issubset(set(result))
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100
    assert isinstance(result["benford_flag"], bool)
    assert isinstance(result["ml_flag"], bool)


def test_risk_scorer_score_matrix(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    features = df.drop(columns=["label"])
    scored = scorer.score_matrix(features)

    assert "wallet" in scored.columns
    assert {"score", "benford_flag", "ml_flag", "confidence"}.issubset(set(scored.columns))
    assert len(scored) == len(features)


def test_risk_scorer_raises_without_models(tmp_path):
    scorer = RiskScorer(model_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        scorer.score(
            generate_synthetic_dataset(n_wallets=2, seed=3).drop(columns=["label"]).iloc[0]
        )


def test_risk_scorer_exposes_metadata(trained_models):
    output, model_dir, _ = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    assert scorer.metadata is not None
    assert isinstance(scorer.metadata["trained_at"], str)
    assert len(scorer.metadata["trained_at"]) > 0
    assert scorer.metadata["feature_columns"] == output["feature_columns"]


def test_risk_scorer_raises_on_schema_mismatch(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    # Manually corrupt the metadata hash
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["feature_schema_hash"] = "sha256:wronghash"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # Re-instantiate to load bad metadata
    scorer = RiskScorer(model_dir=model_dir)
    row = df.drop(columns=["label"]).iloc[0]

    with pytest.raises(RuntimeError) as excinfo:
        scorer.score(row)
    assert "schema" in str(excinfo.value).lower()


def test_risk_scorer_metadata_none_without_metadata_file(trained_models, tmp_path):
    output, _, df = trained_models
    # Copy models to a new dir without metadata
    new_dir = str(tmp_path)
    save_models(output["results"], new_dir)

    scorer = RiskScorer(model_dir=new_dir)
    assert scorer.metadata is None

    # Scoring should still work
    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert "score" in result


def test_metadata_backward_compat_no_raise_without_file(trained_models, tmp_path):
    output, _, df = trained_models
    new_dir = str(tmp_path)
    save_models(output["results"], new_dir)

    # Should not raise during init or score
    scorer = RiskScorer(model_dir=new_dir)
    row = df.drop(columns=["label"]).iloc[0]
    scorer.score(row)


def test_shap_explainer_explain(trained_models):
    output, _, df = trained_models
    results = output["results"]
    model = results["random_forest"]["model"]
    explainer = ShapExplainer(model)

    row = df.drop(columns=["label"]).iloc[0]
    explanation = explainer.explain(row, top_n=3)

    assert len(explanation) == 3
    for entry in explanation:
        assert set(entry) == {"feature", "contribution", "value"}


def test_shap_explainer_explain_ensemble(trained_models):
    output, _, df = trained_models
    results = output["results"]
    models = {name: result["model"] for name, result in results.items()}
    explainer = ShapExplainer()

    row = df.drop(columns=["label"]).iloc[0]
    explanation = explainer.explain_ensemble(row, models, top_n=3)

    assert len(explanation) == 3
    for entry in explanation:
        assert set(entry) == {"feature", "contribution", "value"}


# ---------------------------------------------------------------------------
# SHAP interaction value tests (Issue #267)
# ---------------------------------------------------------------------------

def _make_single_feature_model():
    """Train a single depth-1 decision tree that only splits on f0.

    With max_depth=1 and n_estimators=1, the tree makes exactly one split on the
    most informative feature (f0). f1 and f2 are never used, so all pairwise
    SHAP interaction values involving them are identically 0.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, 3))
    y = (X[:, 0] > 0.5).astype(int)  # label depends only on f0
    clf = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0)
    clf.fit(X, y)
    return clf, pd.DataFrame(X[:20], columns=["f0", "f1", "f2"])


def test_interaction_values_zero_for_non_informative_pairs(monkeypatch):
    """For a model linear in f0 only, interactions for (f1,f2) must be < 0.001."""
    import config as cfg_module

    monkeypatch.setattr(cfg_module.config, "SHAP_INTERACTIONS_ENABLED", True)

    model, X = _make_single_feature_model()
    explainer = ShapExplainer(model)
    interactions = explainer.compute_interaction_values(model, X, top_n=3)

    # Find the (f1, f2) interaction — it must be near zero
    f1_f2 = next(
        (ix for ix in interactions if set([ix["feature_a"], ix["feature_b"]]) == {"f1", "f2"}),
        None,
    )
    # If it's not in top_n, it's even smaller — that also passes
    if f1_f2 is not None:
        assert f1_f2["interaction"] < 0.001, (
            f"Expected (f1, f2) interaction < 0.001, got {f1_f2['interaction']}"
        )


def test_format_top_interactions_produces_five_strings():
    """format_top_interactions must return exactly 5 correctly formatted strings."""
    from detection.shap_explainer import format_top_interactions
    import re

    raw = [
        {"feature_a": f"feat_{i}", "feature_b": f"feat_{i+1}", "interaction": float(i) * 0.1}
        for i in range(5)
    ]
    result = format_top_interactions(raw)

    assert len(result) == 5
    pattern = re.compile(r"^.+ x .+ contributes [0-9]+\.[0-9]+ points to the score$")
    for s in result:
        assert pattern.match(s), f"String {s!r} does not match expected format"
