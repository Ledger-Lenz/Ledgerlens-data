"""Unit tests for validate_feature_ranges and SHAP feature dictionary coverage.

Tests:
1. validate_feature_ranges returns zero violations for every row in the
   synthetic dataset (training data must not contain out-of-range values).
2. FEATURE_RANGES keys are all documented in data/feature_dictionary.md.
3. feature_dict_url returns correctly-formed URLs for known features and
   None for unknown ones.
4. validate_feature_ranges correctly detects deliberate out-of-range injections.
5. ShapExplainer.explain() / explain_ensemble() include a dict_url field.

Imports use importlib.util to load detection.feature_engineering and
detection.shap_explainer directly, bypassing detection/__init__.py which
has a transitive import of utils.logging → config that requires a full
environment setup.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Environment bootstrap (mirrors tests/conftest.py)
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "models"))
os.environ.setdefault("RISK_SCORE_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("WATCHED_ASSET_PAIRS", "USDC:native,XLM:native")
os.environ.setdefault("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
os.environ.setdefault("MIN_TRADES_FOR_SCORING", "20")

# ---------------------------------------------------------------------------
# Load modules via importlib to avoid detection/__init__.py import chain
# ---------------------------------------------------------------------------

def _load(name: str, rel_path: str):
    """Load a module by file path without going through package __init__."""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Provide a minimal config stub so benford_engine and feature_engineering load
_cfg_stub = types.SimpleNamespace(
    MIN_TRADES_FOR_SCORING=5,
    BENFORD_WINDOWS_HOURS=[1, 4, 24, 168, 720],
    BENFORD_CI_ENABLED=False,
    GNN_EMBEDDING_DIM=32,
    ASSET_BENFORD_WINDOWS={},
    CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS=30,
    SHAP_INTERACTIONS_ENABLED=False,
    DP_EPSILON=1.0,
    DP_DELTA=1e-5,
    DP_RENYI_QUERY_THRESHOLD=100,
    DP_RENYI_NOISE_MULTIPLIER=3.0,
    DP_DEFAULT_SENSITIVITY=0.05,
    SHAP_SENSITIVITY_PATH="models/shap_sensitivity.json",
)
_cfg_module = types.ModuleType("config")
_cfg_module.config = _cfg_stub
sys.modules.setdefault("config", _cfg_module)

# Load benford_engine first (feature_engineering depends on it)
_load("benford_engine", "detection/benford_engine.py")
sys.modules.setdefault("detection.benford_engine", sys.modules["benford_engine"])

# Minimal stubs for heavy transitive dependencies
for _stub_name in [
    "detection.streaming_benford",
    "detection.wallet_graph",
    "detection.ts_decomposition",
    "detection.cross_venue_features",
    "detection.gnn_encoder",
    "ingestion.data_models",
    "networkx",
]:
    if _stub_name not in sys.modules:
        _m = types.ModuleType(_stub_name)
        sys.modules[_stub_name] = _m

# Minimal NetworkX stub (feature_engineering type-hints use nx.DiGraph)
_nx = sys.modules["networkx"]
_nx.DiGraph = type("DiGraph", (), {})  # type: ignore[attr-defined]

# Minimal wallet_graph stubs
_wg = sys.modules["detection.wallet_graph"]
_wg.NO_RING = -1  # type: ignore[attr-defined]
_wg.compute_wallet_graph_metrics = lambda w, g: {"funding_source_similarity": 0.0, "network_centrality": 0.0}  # type: ignore[attr-defined]
_wg.detect_wash_trading_rings = lambda g, **kw: {}  # type: ignore[attr-defined]
_wg.build_ring_statistics = lambda cm, g: {}  # type: ignore[attr-defined]

# Minimal streaming_benford stub
_sb = sys.modules["detection.streaming_benford"]
_sb.StreamingBenfordSketch = type("StreamingBenfordSketch", (), {})  # type: ignore[attr-defined]

# ingestion.data_models stub
_dm = sys.modules["ingestion.data_models"]
_dm.AccountActivity = type("AccountActivity", (), {})  # type: ignore[attr-defined]

# Load feature_engineering
fe = _load("detection.feature_engineering", "detection/feature_engineering.py")

FEATURE_RANGES = fe.FEATURE_RANGES
_FEATURE_ANCHORS = fe._FEATURE_ANCHORS
FEATURE_DICT_BASE_URL = fe.FEATURE_DICT_BASE_URL
feature_dict_url = fe.feature_dict_url
validate_feature_ranges = fe.validate_feature_ranges

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SYNTHETIC_DATASET = REPO_ROOT / "data" / "synthetic_dataset.parquet"
FEATURE_DICT = REPO_ROOT / "data" / "feature_dictionary.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dict_feature_names() -> set[str]:
    """Extract feature names documented in data/feature_dictionary.md."""
    text = FEATURE_DICT.read_text(encoding="utf-8")
    names: set[str] = set()
    # Pattern: ### N.M · `feature_name`
    for match in re.finditer(r"###.*?`([a-zA-Z_0-9{}]+)`", text):
        raw = match.group(1)
        if "{h}" in raw:
            for h in [1, 4, 24, 168, 720]:
                names.add(raw.replace("{h}", str(h)))
        else:
            names.add(raw)
    # GNN dimensions: gnn_0 … gnn_31
    if "`gnn_0`" in text or "gnn_0" in names:
        for i in range(32):
            names.add(f"gnn_{i}")
    return names


# ---------------------------------------------------------------------------
# Test 1: synthetic dataset → zero range violations
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not SYNTHETIC_DATASET.exists(),
    reason="Synthetic dataset not found; run scripts/generate_synthetic_dataset.py",
)
def test_validate_feature_ranges_on_synthetic_dataset():
    """Every row of the synthetic dataset must pass range validation."""
    df = pd.read_parquet(SYNTHETIC_DATASET)
    all_violations: list[str] = []
    for _, row in df.iterrows():
        violations = validate_feature_ranges(row.to_dict())
        all_violations.extend(violations)

    assert all_violations == [], (
        f"Found {len(all_violations)} out-of-range value(s) in synthetic dataset:\n"
        + "\n".join(f"  {v}" for v in all_violations[:20])
        + ("\n  …" if len(all_violations) > 20 else "")
    )


# ---------------------------------------------------------------------------
# Test 2: FEATURE_RANGES keys are documented in the dictionary
# ---------------------------------------------------------------------------

def test_shap_feature_names_subset_of_dictionary():
    dict_names = _load_dict_feature_names()
    for feature in FEATURE_RANGES:
        assert feature in dict_names or feature in _FEATURE_ANCHORS, (
            f"Feature '{feature}' is in FEATURE_RANGES but not in feature_dictionary.md."
        )


def test_feature_anchor_map_covers_feature_ranges():
    for feature in FEATURE_RANGES:
        assert feature in _FEATURE_ANCHORS, (
            f"Feature '{feature}' is in FEATURE_RANGES but missing from _FEATURE_ANCHORS."
        )


# ---------------------------------------------------------------------------
# Test 3: feature_dict_url
# ---------------------------------------------------------------------------

def test_feature_dict_url_known_features():
    known = [
        "benford_mad_24h",
        "counterparty_concentration_ratio",
        "account_age_days",
        "cross_pair_trade_synchrony",
        "inter_arrival_cv",
        "gnn_0",
        "gnn_31",
        "solana_linked_wash_score",
        "bridge_round_trip_ratio",
    ]
    for feature in known:
        url = feature_dict_url(feature)
        assert url is not None, f"feature_dict_url('{feature}') returned None"
        assert url.startswith(FEATURE_DICT_BASE_URL)
        assert "#" in url, f"URL for '{feature}' has no anchor: {url}"


def test_feature_dict_url_unknown_feature():
    assert feature_dict_url("wallet") is None
    assert feature_dict_url("label") is None
    assert feature_dict_url("totally_unknown_xyz") is None


def test_feature_dict_url_all_windows():
    for h in [1, 4, 24, 168, 720]:
        for prefix in ["benford_chi_square", "benford_mad", "benford_z_max",
                       "benford_residual_chi_square", "benford_residual_mad"]:
            url = feature_dict_url(f"{prefix}_{h}h")
            assert url is not None, f"feature_dict_url('{prefix}_{h}h') returned None"


# ---------------------------------------------------------------------------
# Test 4: deliberate violation detection
# ---------------------------------------------------------------------------

def test_validate_detects_out_of_range_low():
    violations = validate_feature_ranges({"counterparty_concentration_ratio": -0.1})
    assert len(violations) == 1
    assert "counterparty_concentration_ratio" in violations[0]


def test_validate_detects_out_of_range_high():
    violations = validate_feature_ranges({"benford_mad_24h": 1.5})
    assert len(violations) == 1
    assert "benford_mad_24h" in violations[0]


def test_validate_passes_valid_values():
    valid = {
        "counterparty_concentration_ratio": 0.75,
        "benford_mad_24h": 0.02,
        "benford_chi_square_24h": 45.0,
        "account_age_days": 365.0,
        "cross_pair_volume_correlation": -0.3,
        "entropy_of_amounts": 4.2,
    }
    assert validate_feature_ranges(valid) == []


def test_validate_nan_skipped():
    violations = validate_feature_ranges({
        "counterparty_concentration_ratio": float("nan"),
        "benford_mad_24h": float("nan"),
    })
    assert violations == []


def test_validate_unknown_features_ignored():
    violations = validate_feature_ranges({
        "wallet": "GSYNTH000001",
        "label": 1,
        "gnn_0": 0.0,
        "unknown_future_feature": 9999.9,
    })
    assert violations == []


def test_validate_raise_on_violation():
    with pytest.raises(ValueError, match="Feature range violations"):
        validate_feature_ranges({"self_matching_rate": 2.0}, raise_on_violation=True)


def test_validate_multiple_violations():
    bad = {
        "counterparty_concentration_ratio": -0.5,
        "benford_mad_1h": 2.0,
        "cross_pair_volume_correlation": 1.5,
    }
    violations = validate_feature_ranges(bad)
    assert len(violations) == 3


# ---------------------------------------------------------------------------
# Test 5: SHAP explain() / explain_ensemble() include dict_url
# ---------------------------------------------------------------------------

def _load_shap_explainer():
    """Load ShapExplainer with all dependencies stubbed out."""
    # Stub detection.model_training (don't try to exec it — heavy chain)
    _mt = types.ModuleType("detection.model_training")
    _mt.FEATURE_COLUMNS_EXCLUDE = set()  # type: ignore[attr-defined]
    sys.modules["detection.model_training"] = _mt

    # Stub shap with TreeExplainer
    _shap = types.ModuleType("shap")
    _shap.TreeExplainer = type("TreeExplainer", (), {})  # type: ignore[attr-defined]
    sys.modules["shap"] = _shap

    # Stub differential_privacy
    _dp = types.ModuleType("detection.differential_privacy")
    _dp.feature_sensitivity = lambda sens, feat: 0.05  # type: ignore[attr-defined]
    _dp.gaussian_sigma = lambda sensitivity, eps, delta: 0.1  # type: ignore[attr-defined]
    _dp.load_shap_sensitivity = lambda: {}  # type: ignore[attr-defined]
    _dp.renyi_noise_multiplier = lambda count: 1.0  # type: ignore[attr-defined]
    sys.modules["detection.differential_privacy"] = _dp

    se_mod = _load("detection.shap_explainer", "detection/shap_explainer.py")
    return se_mod.ShapExplainer


def test_shap_explain_includes_dict_url():
    """explain() must include a dict_url key in every returned entry."""
    ShapExplainer = _load_shap_explainer()

    feature_cols = [
        "benford_mad_24h", "counterparty_concentration_ratio",
        "account_age_days", "entropy_of_amounts", "inter_arrival_cv",
    ]
    row = pd.Series({col: 0.5 for col in feature_cols})
    mock_explainer = types.SimpleNamespace(
        shap_values=lambda X: np.array([[0.1, 0.2, 0.3, 0.05, 0.15]])
    )
    se = ShapExplainer()
    se._get_explainer = lambda model: mock_explainer  # type: ignore[method-assign]

    results = se.explain(row, top_n=3, model=object())
    assert len(results) > 0
    for entry in results:
        assert "dict_url" in entry, f"dict_url missing from: {entry}"
        assert entry["dict_url"] is None or entry["dict_url"].startswith("https://")


def test_shap_explain_ensemble_includes_dict_url():
    """explain_ensemble() must also include dict_url in each entry."""
    ShapExplainer = _load_shap_explainer()

    feature_cols = ["benford_mad_24h", "counterparty_concentration_ratio", "account_age_days"]
    row = pd.Series({col: 0.5 for col in feature_cols})
    mock_explainer = types.SimpleNamespace(
        shap_values=lambda X: np.array([[0.1, 0.2, 0.3]])
    )
    se = ShapExplainer()
    se._get_explainer = lambda model: mock_explainer  # type: ignore[method-assign]

    results = se.explain_ensemble(row, models={"rf": object(), "xgb": object()}, top_n=3)
    assert len(results) > 0
    for entry in results:
        assert "dict_url" in entry


def test_shap_explain_includes_dict_url():
    """explain() must include a dict_url key in every returned entry."""
    ShapExplainer = _load_shap_explainer()

    feature_cols = [
        "benford_mad_24h", "counterparty_concentration_ratio",
        "account_age_days", "entropy_of_amounts", "inter_arrival_cv",
    ]
    row = pd.Series({col: 0.5 for col in feature_cols})
    mock_explainer = types.SimpleNamespace(
        shap_values=lambda X: np.array([[0.1, 0.2, 0.3, 0.05, 0.15]])
    )
    se = ShapExplainer()
    se._get_explainer = lambda model: mock_explainer  # type: ignore[method-assign]

    results = se.explain(row, top_n=3, model=object())
    assert len(results) > 0
    for entry in results:
        assert "dict_url" in entry, f"dict_url missing from: {entry}"
        assert entry["dict_url"] is None or entry["dict_url"].startswith("https://")


def test_shap_explain_ensemble_includes_dict_url():
    """explain_ensemble() must also include dict_url in each entry."""
    ShapExplainer = _load_shap_explainer()

    feature_cols = ["benford_mad_24h", "counterparty_concentration_ratio", "account_age_days"]
    row = pd.Series({col: 0.5 for col in feature_cols})
    mock_explainer = types.SimpleNamespace(
        shap_values=lambda X: np.array([[0.1, 0.2, 0.3]])
    )
    se = ShapExplainer()
    se._get_explainer = lambda model: mock_explainer  # type: ignore[method-assign]

    results = se.explain_ensemble(row, models={"rf": object(), "xgb": object()}, top_n=3)
    assert len(results) > 0
    for entry in results:
        assert "dict_url" in entry
