import json
import os
import time
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from config import config
from detection.list_override import ListOverride
from detection.model_inference import RiskScorer
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer


@pytest.fixture
def temp_lists(tmp_path):
    allowlist_file = tmp_path / "allowlist.json"
    denylist_file = tmp_path / "denylist.json"

    allowlist_file.write_text(json.dumps(["GALLOW123"]))
    denylist_file.write_text(json.dumps(["GDENY123"]))

    return allowlist_file, denylist_file


def test_list_override_basic(temp_lists):
    allowpath, denypath = temp_lists
    override = ListOverride(allowlist_path=str(allowpath), denylist_path=str(denypath))

    # Listed wallets
    assert override.check("GALLOW123") == 0
    assert override.check("GDENY123") == 100

    # Unlisted wallet
    assert override.check("GUNKNOWN") is None


def test_list_override_hot_reload(temp_lists, monkeypatch):
    allowpath, denypath = temp_lists
    override = ListOverride(allowlist_path=str(allowpath), denylist_path=str(denypath))

    monkeypatch.setattr(config, "LIST_RELOAD_INTERVAL_SECONDS", 1)

    assert override.check("GALLOW123") == 0
    assert override.check("GNEW") is None

    # Update allowlist file on disk
    allowpath.write_text(json.dumps(["GALLOW123", "GNEW"]))

    # Before interval has elapsed, it should not reload
    assert override.check("GNEW") is None

    # Simulate passing of reload interval
    override._last_loaded = time.time() - 2

    # Now it should reload
    assert override.check("GNEW") == 0


def test_risk_scorer_override_integration(temp_lists):
    allowpath, denypath = temp_lists

    # Create dummy scorer without running full init
    scorer = RiskScorer.__new__(RiskScorer)
    scorer.model_dir = "dummy_dir"
    scorer.list_override = ListOverride(allowlist_path=str(allowpath), denylist_path=str(denypath))
    scorer.models = {"dummy": MagicMock()}

    # Score an allowlisted wallet
    row_allow = pd.Series({"wallet": "GALLOW123", "feature_a": 1.0})
    result_allow = scorer.score(row_allow)
    assert result_allow["score"] == 0
    assert result_allow["ml_flag"] is False
    assert result_allow["confidence"] == 100

    # Score a denylisted wallet
    row_deny = pd.Series({"wallet": "GDENY123", "feature_a": 1.0})
    result_deny = scorer.score(row_deny)
    assert result_deny["score"] == 100
    assert result_deny["ml_flag"] is True
    assert result_deny["confidence"] == 100


def test_streaming_scorer_override_integration(temp_lists):
    allowpath, denypath = temp_lists

    # Create streaming scorer
    scorer = StreamingScorer.__new__(StreamingScorer)
    scorer.min_trades = 20
    scorer._risk_scorer = RiskScorer.__new__(RiskScorer)
    scorer._risk_scorer.list_override = ListOverride(allowlist_path=str(allowpath), denylist_path=str(denypath))

    # A wallet with 0 trades in the buffer should still score if overridden
    buf = FeatureBuffer()
    res_allow = scorer.score_wallet("GALLOW123", buf)
    assert res_allow is not None
    assert res_allow["score"] == 0

    res_deny = scorer.score_wallet("GDENY123", buf)
    assert res_deny is not None
    assert res_deny["score"] == 100

    # An unlisted wallet with 0 trades should return None
    assert scorer.score_wallet("GUNKNOWN", buf) is None


@patch("scripts.score_wallet.load_trades")
@patch("scripts.score_wallet.RiskScorer")
def test_score_wallet_cli_bypasses_ingest(mock_scorer_cls, mock_load_trades, temp_lists, capsys):
    allowpath, denypath = temp_lists

    # Set up mock RiskScorer
    mock_scorer = MagicMock()
    mock_scorer.list_override = ListOverride(allowlist_path=str(allowpath), denylist_path=str(denypath))
    mock_scorer_cls.return_value = mock_scorer

    # Allowlisted wallet
    test_wallet = "GALLOW12356789012356789012356789012356789012356789012"  # Must look like Stellar public key
    allowpath.write_text(json.dumps([test_wallet]))
    mock_scorer.list_override._reload()

    with patch("sys.argv", ["score_wallet.py", "--wallet", test_wallet, "--pair", "USDC:G..."]):
        from scripts.score_wallet import main as cli_main
        cli_main()

    # Verify load_trades was NOT called
    mock_load_trades.assert_not_called()

    # Check output
    out, _ = capsys.readouterr()
    assert f"Wallet:   {test_wallet}" in out
    assert "Score:    0  [OK]" in out
