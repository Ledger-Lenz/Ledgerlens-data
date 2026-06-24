import json
import os

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
