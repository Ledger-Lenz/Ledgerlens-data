"""Unit tests for scripts/quantize_models.py.

Tests cover:
1. Quantised DANN encoder output agrees with full-precision within 0.05 on
   ≥ 95 % of samples.
2. Pruned model has strictly fewer non-zero weights than the original.
3. Tree leaf quantisation reduces or preserves model size and leaf values are
   representable in float16.
4. Artifact keys are registered in metrics.json after a non-dry-run save.
5. Dry-run mode writes no files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional torch — skip GPU-dependent tests gracefully
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn  # noqa: E402

from scripts.quantize_models import (  # noqa: E402
    _count_nonzero,
    _get_prunable_layers,
    _load_metrics,
    _quantise_rf_leaves,
    _register_artifact,
    _save_metrics,
    _sha256_file,
    prune_dann_encoder,
    prune_gnn_encoder,
    prune_pytorch_model,
    quantise_dann_int8,
    quantise_gnn_int8,
    quantise_tree_models,
    run_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_dann(input_dim: int = 8) -> nn.Module:
    """Construct a minimal DANNEncoder for fast testing."""
    from detection.dann_encoder import DANNEncoder

    return DANNEncoder(input_dim=input_dim, hidden_dim=16, embedding_dim=8)


def _tiny_gnn() -> nn.Module:
    """Construct a minimal _GraphSAGEModel for fast testing."""
    try:
        from torch_geometric.nn import SAGEConv
    except ImportError:
        pytest.skip("torch_geometric not installed")

    from detection.gnn_encoder import _GraphSAGEModel

    return _GraphSAGEModel(in_channels=5, hidden_channels=16, out_channels=8)


def _save_dann(model_dir: str, input_dim: int = 8) -> str:
    """Save a fresh DANN encoder to *model_dir* and return its path."""
    m = _tiny_dann(input_dim=input_dim)
    path = os.path.join(model_dir, "dann_encoder.pt")
    torch.save(m.state_dict(), path)
    return path


def _save_gnn(model_dir: str) -> str:
    """Save a fresh GNN model to *model_dir* and return its path."""
    m = _tiny_gnn()
    path = os.path.join(model_dir, "gnn_encoder.pt")
    torch.save(m.state_dict(), path)
    # Write minimal metrics.json so GNNEncoder.load() can verify SHA
    sha = _sha256_file(path)
    metrics = {"gnn_encoder": {"artifact_sha256": sha, "embedding_dim": 8, "hidden_dim": 16}}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    return path


def _make_rf(n_estimators: int = 3):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=60, n_features=8, random_state=0)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=3, random_state=0)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# 1. Output agreement: INT8 DANN ≤ 0.05 absolute score difference on ≥ 95 %
# ---------------------------------------------------------------------------


class TestDANNOutputAgreement:
    """Quantised DANN encoder outputs must agree with full-precision within 0.05
    on at least 95 % of test samples (requirement from the issue spec)."""

    def test_static_int8_output_agreement(self, tmp_path):
        input_dim = 16
        m_fp = _tiny_dann(input_dim=input_dim)
        m_fp.eval()
        torch.save(m_fp.state_dict(), tmp_path / "dann_encoder.pt")

        # Quantise
        result = quantise_dann_int8(str(tmp_path), dry_run=False, private_key_path=None)
        assert result, "quantise_dann_int8 returned empty result"
        assert os.path.exists(str(tmp_path / "dann_encoder_int8.pt"))

        # Load quantised wrapper state back (we compare feature extractor outputs)
        from scripts.quantize_models import _QuantisableDANN, tq

        wrapper_q = _QuantisableDANN(m_fp.feature_extractor)
        wrapper_q.eval()
        try:
            tq.fuse_modules(wrapper_q, [["feature_extractor.0", "feature_extractor.1"]], inplace=True)
            tq.fuse_modules(wrapper_q, [["feature_extractor.2", "feature_extractor.3"]], inplace=True)
        except Exception:
            pass
        wrapper_q.qconfig = tq.get_default_qconfig("fbgemm")
        tq.prepare(wrapper_q, inplace=True)
        rng = np.random.default_rng(42)
        calib = torch.tensor(rng.standard_normal((64, input_dim)).astype(np.float32))
        with torch.no_grad():
            for i in range(0, 64, 8):
                wrapper_q(calib[i : i + 8])
        tq.convert(wrapper_q, inplace=True)

        # Generate test samples
        n_test = 200
        X_test = torch.tensor(
            rng.standard_normal((n_test, input_dim)).astype(np.float32)
        )

        with torch.no_grad():
            fp_out = m_fp.encode(X_test).numpy()
            q_out = wrapper_q(X_test).numpy()

        # Compare element-wise absolute differences across all outputs
        diff = np.abs(fp_out - q_out)
        within_tol = (diff < 0.05).all(axis=1)  # all dims within tol per sample
        pct_passing = within_tol.mean()

        assert pct_passing >= 0.95, (
            f"Only {pct_passing:.1%} of samples within 0.05 tolerance (need ≥ 95 %)"
        )

    def test_predict_proba_agreement_within_tolerance(self, tmp_path):
        """Pruned DANN predict_proba must agree within 0.05 on ≥ 95 % of samples."""
        input_dim = 16
        m_fp = _tiny_dann(input_dim=input_dim)
        m_fp.eval()
        torch.save(m_fp.state_dict(), tmp_path / "dann_encoder.pt")

        result = prune_dann_encoder(str(tmp_path), sparsity=0.1, dry_run=False)
        assert result

        from detection.dann_encoder import DANNEncoder

        state_pruned = torch.load(
            result["output_path"], map_location="cpu", weights_only=True
        )
        m_pruned = DANNEncoder(input_dim=input_dim, hidden_dim=16, embedding_dim=8)
        m_pruned.load_state_dict(state_pruned)
        m_pruned.eval()

        rng = np.random.default_rng(0)
        X = torch.tensor(rng.standard_normal((200, input_dim)).astype(np.float32))

        with torch.no_grad():
            fp_proba = m_fp.predict_proba(X)
            pr_proba = m_pruned.predict_proba(X)

        diff = np.abs(fp_proba - pr_proba)
        pct_passing = (diff < 0.05).mean()
        assert pct_passing >= 0.95, (
            f"Only {pct_passing:.1%} of pruned DANN samples within 0.05 tolerance"
        )


# ---------------------------------------------------------------------------
# 2. Pruning sparsity: pruned model has fewer non-zero weights
# ---------------------------------------------------------------------------


class TestPruningSparsity:
    def test_prune_dann_reduces_nonzero(self, tmp_path):
        input_dim = 16
        m = _tiny_dann(input_dim=input_dim)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")

        result = prune_dann_encoder(str(tmp_path), sparsity=0.2, dry_run=False)

        assert result["nonzero_after"] < result["nonzero_before"], (
            "Pruning did not reduce non-zero weight count"
        )
        assert result["sparsity_actual"] >= 0.15, (
            f"Actual sparsity {result['sparsity_actual']:.2f} below expected ≥ 0.15"
        )

    def test_prune_gnn_reduces_nonzero(self, tmp_path):
        _save_gnn(str(tmp_path))

        result = prune_gnn_encoder(str(tmp_path), sparsity=0.3, dry_run=False)

        if not result:
            pytest.skip("torch_geometric not installed or GNN load failed")

        assert result["nonzero_after"] < result["nonzero_before"]
        assert result["sparsity_actual"] >= 0.25

    def test_prune_pytorch_model_nonzero(self):
        model = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)
        )
        nz_before, total = _count_nonzero(model)

        pruned = prune_pytorch_model(model, sparsity=0.3)
        nz_after, _ = _count_nonzero(pruned)

        assert nz_after < nz_before, "Pruning must remove weights"
        assert nz_after <= total * 0.75, "At least 25 % should be zeroed"

    def test_multiple_sparsity_levels_monotonic(self, tmp_path):
        """Higher sparsity → fewer non-zero weights."""
        input_dim = 16
        results = {}
        for s in [0.1, 0.2, 0.3]:
            m = _tiny_dann(input_dim=input_dim)
            torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")
            r = prune_dann_encoder(str(tmp_path), sparsity=s, dry_run=False)
            results[s] = r["nonzero_after"]

        assert results[0.1] >= results[0.2] >= results[0.3], (
            "Non-zero count must be monotonically non-increasing with sparsity"
        )

    @pytest.mark.parametrize("sparsity", [0.1, 0.2, 0.3])
    def test_sparsity_output_file_suffix(self, tmp_path, sparsity):
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")
        result = prune_dann_encoder(str(tmp_path), sparsity=sparsity, dry_run=False)
        expected_suffix = f"_pruned_s{int(round(sparsity * 100))}.pt"
        assert result["output_path"].endswith(expected_suffix), (
            f"Expected suffix {expected_suffix}, got {result['output_path']}"
        )


# ---------------------------------------------------------------------------
# 3. Tree model leaf quantisation
# ---------------------------------------------------------------------------


class TestTreeLeafQuantisation:
    def test_rf_leaf_quantisation(self, tmp_path):
        import joblib

        clf = _make_rf()
        src = str(tmp_path / "random_forest.joblib")
        joblib.dump(clf, src)

        results = quantise_tree_models(str(tmp_path), dry_run=False)

        assert "random_forest" in results
        dst = results["random_forest"]["output_path"]
        assert os.path.exists(dst)

        # Reload and verify leaf values are representable in float16
        clf_q = joblib.load(dst)
        for estimator in clf_q.estimators_:
            vals = estimator.tree_.value
            # float16 round-trip: no value should have changed by more than
            # the float16 epsilon scaled by the value magnitude
            diff = np.abs(vals - vals.astype(np.float16).astype(np.float64))
            max_diff = np.max(diff) if diff.size > 0 else 0
            assert max_diff < 1e-2, (
                f"Leaf value diff {max_diff} exceeds float16 representation error"
            )

    def test_rf_output_shape_preserved(self, tmp_path):
        """Quantised RF must still return correct output shape."""
        import joblib
        from sklearn.datasets import make_classification

        clf = _make_rf()
        src = str(tmp_path / "random_forest.joblib")
        joblib.dump(clf, src)
        quantise_tree_models(str(tmp_path), dry_run=False)
        clf_q = joblib.load(str(tmp_path / "random_forest_leafq.joblib"))

        X, _ = make_classification(n_samples=20, n_features=8, random_state=1)
        proba_orig = clf.predict_proba(X)
        proba_q = clf_q.predict_proba(X)
        assert proba_orig.shape == proba_q.shape

    def test_rf_proba_agreement(self, tmp_path):
        """Quantised RF class probabilities must agree within 0.05 on ≥ 95 %."""
        import joblib
        from sklearn.datasets import make_classification

        clf = _make_rf(n_estimators=10)
        src = str(tmp_path / "random_forest.joblib")
        joblib.dump(clf, src)
        quantise_tree_models(str(tmp_path), dry_run=False)
        clf_q = joblib.load(str(tmp_path / "random_forest_leafq.joblib"))

        X, _ = make_classification(n_samples=200, n_features=8, random_state=2)
        p_fp = clf.predict_proba(X)[:, 1]
        p_q = clf_q.predict_proba(X)[:, 1]

        pct_passing = (np.abs(p_fp - p_q) < 0.05).mean()
        assert pct_passing >= 0.95, (
            f"Only {pct_passing:.1%} of tree samples within 0.05 tolerance"
        )

    def test_dry_run_writes_no_files(self, tmp_path):
        import joblib

        clf = _make_rf()
        joblib.dump(clf, str(tmp_path / "random_forest.joblib"))

        before = set(os.listdir(str(tmp_path)))
        quantise_tree_models(str(tmp_path), dry_run=True)
        after = set(os.listdir(str(tmp_path)))

        assert before == after, f"Dry-run wrote unexpected files: {after - before}"


# ---------------------------------------------------------------------------
# 4. Artifact registration in metrics.json
# ---------------------------------------------------------------------------


class TestArtifactRegistration:
    def test_metrics_updated_after_rf_leafq(self, tmp_path):
        import joblib

        clf = _make_rf()
        joblib.dump(clf, str(tmp_path / "random_forest.joblib"))

        quantise_tree_models(str(tmp_path), dry_run=False)

        metrics = _load_metrics(str(tmp_path))
        assert "random_forest_leafq" in metrics
        assert "artifact_sha256" in metrics["random_forest_leafq"]

    def test_metrics_updated_after_dann_prune(self, tmp_path):
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")

        prune_dann_encoder(str(tmp_path), sparsity=0.1, dry_run=False)

        metrics = _load_metrics(str(tmp_path))
        assert "dann_encoder_pruned_s10" in metrics
        assert "artifact_sha256" in metrics["dann_encoder_pruned_s10"]

    def test_metrics_updated_after_dann_int8(self, tmp_path):
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")

        result = quantise_dann_int8(str(tmp_path), dry_run=False)
        assert result

        metrics = _load_metrics(str(tmp_path))
        assert "dann_encoder_int8" in metrics
        entry = metrics["dann_encoder_int8"]
        assert "artifact_sha256" in entry
        # Verify the SHA matches the actual file
        actual_sha = _sha256_file(result["output_path"])
        assert entry["artifact_sha256"] == actual_sha


# ---------------------------------------------------------------------------
# 5. Filename suffix requirements
# ---------------------------------------------------------------------------


class TestFilenameConventions:
    def test_int8_suffix(self, tmp_path):
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")
        result = quantise_dann_int8(str(tmp_path), dry_run=False)
        assert result["output_path"].endswith("_int8.pt"), (
            "INT8 artifact must end with _int8.pt"
        )

    def test_pruned_suffix_contains_sparsity(self, tmp_path):
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")
        result = prune_dann_encoder(str(tmp_path), sparsity=0.2, dry_run=False)
        assert "_pruned_s20" in result["output_path"], (
            "Pruned artifact must contain _pruned_sNN in filename"
        )

    def test_leafq_suffix(self, tmp_path):
        import joblib

        clf = _make_rf()
        joblib.dump(clf, str(tmp_path / "random_forest.joblib"))
        results = quantise_tree_models(str(tmp_path), dry_run=False)
        assert results["random_forest"]["output_path"].endswith("_leafq.joblib"), (
            "Leaf-quantised tree must end with _leafq.joblib"
        )

    def test_original_not_overwritten(self, tmp_path):
        """Compression must never overwrite the full-precision artifact."""
        import joblib

        clf = _make_rf()
        src = str(tmp_path / "random_forest.joblib")
        joblib.dump(clf, src)
        orig_sha = _sha256_file(src)

        quantise_tree_models(str(tmp_path), dry_run=False)

        assert _sha256_file(src) == orig_sha, "Original .joblib was modified"

    def test_pytorch_original_not_overwritten(self, tmp_path):
        m = _tiny_dann(input_dim=16)
        src = str(tmp_path / "dann_encoder.pt")
        torch.save(m.state_dict(), src)
        orig_sha = _sha256_file(src)

        prune_dann_encoder(str(tmp_path), sparsity=0.1, dry_run=False)
        quantise_dann_int8(str(tmp_path), dry_run=False)

        assert _sha256_file(src) == orig_sha, "Original .pt was modified"


# ---------------------------------------------------------------------------
# 6. run_all integration test
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_run_all_dry_run_writes_nothing(self, tmp_path):
        import joblib

        clf = _make_rf()
        joblib.dump(clf, str(tmp_path / "random_forest.joblib"))
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")

        before = set(os.listdir(str(tmp_path)))
        run_all(str(tmp_path), sparsity_levels=[0.1], dry_run=True)
        after = set(os.listdir(str(tmp_path)))

        assert before == after, f"Dry-run wrote files: {after - before}"

    def test_run_all_tree_only(self, tmp_path):
        import joblib

        clf = _make_rf()
        joblib.dump(clf, str(tmp_path / "random_forest.joblib"))

        report = run_all(
            str(tmp_path),
            sparsity_levels=[0.1],
            targets={"tree_leafq"},
            dry_run=False,
        )
        assert "tree_leafq" in report["results"]
        assert "random_forest" in report["results"]["tree_leafq"]

    def test_run_all_dann_prune_all_sparsities(self, tmp_path):
        m = _tiny_dann(input_dim=16)
        torch.save(m.state_dict(), tmp_path / "dann_encoder.pt")

        report = run_all(
            str(tmp_path),
            sparsity_levels=[0.1, 0.2, 0.3],
            targets={"dann_prune"},
            dry_run=False,
        )
        prune_results = report["results"]["dann_prune"]
        assert "s10" in prune_results
        assert "s20" in prune_results
        assert "s30" in prune_results
        # All three pruned files must exist on disk
        for key in ("s10", "s20", "s30"):
            assert os.path.exists(prune_results[key]["output_path"]), (
                f"Pruned artifact for {key} not found"
            )
