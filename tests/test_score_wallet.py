import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.score_wallet import main


@pytest.fixture
def mock_scorer():
    with patch("scripts.score_wallet.RiskScorer") as mock:
        scorer_instance = mock.return_value
        scorer_instance.score.return_value = {
            "score": 83,
            "benford_flag": True,
            "ml_flag": True,
            "confidence": 76,
        }
        scorer_instance.models = {"random_forest": MagicMock()}
        yield scorer_instance


@pytest.fixture
def mock_ingestion():
    with (
        patch("scripts.score_wallet.load_trades") as m_trades,
        patch("scripts.score_wallet.load_orderbook_events") as m_events,
    ):
        m_trades.return_value = iter([])
        m_events.return_value = iter([])
        yield m_trades, m_events


@pytest.fixture
def mock_explainer():
    with patch("scripts.score_wallet.ShapExplainer") as mock:
        explainer_instance = mock.return_value
        explainer_instance.explain_ensemble.return_value = [
            {"feature": "benford_mad_24h", "contribution": 0.34, "value": 0.047},
            {"feature": "counterparty_concentration_ratio", "contribution": 0.29, "value": 0.98},
            {"feature": "round_trip_frequency", "contribution": 0.21, "value": 0.41},
            {"feature": "benford_chi_square_168h", "contribution": 0.18, "value": 45.2},
            {"feature": "account_age_days", "contribution": -0.12, "value": 3.0},
        ]
        yield explainer_instance


def test_score_wallet_outputs_score_and_shap(capsys, mock_scorer, mock_ingestion, mock_explainer):
    test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
    with patch("sys.argv", ["score_wallet.py", "--wallet", test_wallet, "--pair", "USDC:G..."]):
        main()

    out, _ = capsys.readouterr()
    assert "Score:    83" in out
    assert "Benford:  True" in out
    assert "Top 5 SHAP" in out
    assert "benford_mad_24h" in out


def test_score_wallet_json_output_is_valid_json(
    capsys, mock_scorer, mock_ingestion, mock_explainer
):
    test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
    with patch(
        "sys.argv", ["score_wallet.py", "--wallet", test_wallet, "--pair", "USDC:G...", "--json"]
    ):
        main()

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["wallet"] == test_wallet
    assert data["score"] == 83
    assert len(data["shap_explanations"]) == 5


def test_score_wallet_flagged_label(capsys, mock_scorer, mock_ingestion, mock_explainer):
    mock_scorer.score.return_value["score"] = 85
    test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
    with patch("sys.argv", ["score_wallet.py", "--wallet", test_wallet, "--pair", "USDC:G..."]):
        main()

    out, _ = capsys.readouterr()
    assert "[FLAGGED]" in out


def test_score_wallet_ok_label(capsys, mock_scorer, mock_ingestion, mock_explainer):
    mock_scorer.score.return_value["score"] = 30
    test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
    with patch("sys.argv", ["score_wallet.py", "--wallet", test_wallet, "--pair", "USDC:G..."]):
        main()

    out, _ = capsys.readouterr()
    assert "[OK]" in out


def test_score_wallet_invalid_wallet_id_exits_1(capsys):
    with patch("sys.argv", ["score_wallet.py", "--wallet", "BADID", "--pair", "USDC:G..."]):
        with pytest.raises(SystemExit) as excinfo:
            main()
    assert excinfo.value.code == 1
    out, _ = capsys.readouterr()
    assert "Invalid wallet ID format" in out


def test_score_wallet_missing_models_exits_1(capsys, mock_ingestion):
    with patch(
        "scripts.score_wallet.RiskScorer", side_effect=RuntimeError("No trained models found")
    ):
        test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
        with patch("sys.argv", ["score_wallet.py", "--wallet", test_wallet, "--pair", "USDC:G..."]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        _, err = capsys.readouterr()
        assert "model_training.py" in err


def test_score_wallet_causal_json_output_includes_causal_section(
    capsys, mock_scorer, mock_ingestion, mock_explainer
):
    test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
    with patch("scripts.score_wallet.CounterfactualAttributor") as mock_attributor:
        attributor_instance = mock_attributor.return_value
        attributor_instance.counterfactual_score.return_value = {
            "original_score": 83,
            "counterfactual_score": 41,
            "score_delta": 42,
            "features_changed": {"round_trip_frequency": {"original": 0.6, "counterfactual": 0.0}},
        }
        with patch(
            "sys.argv",
            [
                "score_wallet.py",
                "--wallet",
                test_wallet,
                "--pair",
                "USDC:G...",
                "--json",
                "--causal",
            ],
        ):
            main()

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["wallet"] == test_wallet
    assert "causal_attribution" in data
    assert data["causal_attribution"]["counterfactual_score"] == 41


def test_score_wallet_what_if_remove_invalid_trade_raises_value_error(
    mock_scorer, mock_ingestion, mock_explainer
):
    test_wallet = "GABC1234567890123456789012345678901234567890123456789012"
    with patch(
        "sys.argv",
        [
            "score_wallet.py",
            "--wallet",
            test_wallet,
            "--pair",
            "USDC:G...",
            "--what-if-remove",
            "not-a-trade",
        ],
    ):
        with pytest.raises(ValueError):
            main()
