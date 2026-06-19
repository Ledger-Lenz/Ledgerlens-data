"""Tests for streaming.streaming_scorer.StreamingScorer (Issue #12)."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from ingestion.data_models import Asset, Trade
from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"


def _make_trade(
    base_account: str = WALLET_A,
    counter_account: str = WALLET_B,
    base_amount: float = 100.0,
    trade_id: str = "t1",
) -> Trade:
    return Trade(
        trade_id=trade_id,
        ledger_close_time=datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.UTC),
        base_account=base_account,
        counter_account=counter_account,
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=base_amount,
        counter_amount=50.0,
        price=2.0,
    )


@pytest.fixture(scope="module")
def trained_scorer(tmp_path_factory):
    """Return a StreamingScorer backed by real trained models (no metadata file
    so the schema-hash check is skipped — the buffer only populates the core
    feature set, not the cross-pair features that require all_pairs_df)."""
    from detection.model_training import save_models, train_models

    df = generate_synthetic_dataset(n_wallets=60, seed=42)
    output = train_models(df, test_size=0.3, random_state=42)
    model_dir = str(tmp_path_factory.mktemp("models"))
    save_models(output["results"], model_dir)
    # Deliberately omit save_training_artifacts so no model_metadata.json is
    # written; RiskScorer then skips the feature-schema-hash validation.
    return StreamingScorer(model_dir=model_dir)


# ---------------------------------------------------------------------------
# 1. Returns None when wallet has fewer than min_trades
# ---------------------------------------------------------------------------


def test_score_wallet_returns_none_below_min_trades():
    buf = FeatureBuffer()
    scorer = StreamingScorer.__new__(StreamingScorer)
    scorer.min_trades = 20
    scorer._risk_scorer = MagicMock()

    # Add only 5 trades — below the threshold
    for i in range(5):
        buf.update(_make_trade(base_amount=float(i + 1), trade_id=f"t{i}"))

    result = scorer.score_wallet(WALLET_A, buf)
    assert result is None
    scorer._risk_scorer.score.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Returns None for a wallet not present in the buffer
# ---------------------------------------------------------------------------


def test_score_wallet_returns_none_for_unknown_wallet():
    buf = FeatureBuffer()
    scorer = StreamingScorer.__new__(StreamingScorer)
    scorer.min_trades = 1
    scorer._risk_scorer = MagicMock()

    result = scorer.score_wallet("GUNKNOWN_WALLET_XYZ", buf)
    assert result is None


# ---------------------------------------------------------------------------
# 3. Returns None and logs a warning when RiskScorer.score raises
# ---------------------------------------------------------------------------


def test_score_wallet_returns_none_on_scorer_exception():
    buf = FeatureBuffer()
    for i in range(25):
        buf.update(_make_trade(base_amount=float(i + 1) * 7.3, trade_id=f"t{i}"))

    scorer = StreamingScorer.__new__(StreamingScorer)
    scorer.min_trades = 20
    mock_risk_scorer = MagicMock()
    mock_risk_scorer.score.side_effect = RuntimeError("model error")
    scorer._risk_scorer = mock_risk_scorer

    result = scorer.score_wallet(WALLET_A, buf)
    assert result is None


# ---------------------------------------------------------------------------
# 4. Returns a valid risk-score dict when conditions are met
# ---------------------------------------------------------------------------


def test_score_wallet_returns_risk_score_dict(trained_scorer):
    buf = FeatureBuffer()
    for i in range(25):
        buf.update(
            _make_trade(
                base_amount=float(i + 1) * 13.7,
                trade_id=f"t{i}",
            )
        )
    trained_scorer.min_trades = 20

    result = trained_scorer.score_wallet(WALLET_A, buf)

    assert result is not None
    assert set(result) == {"score", "benford_flag", "ml_flag", "confidence"}
    assert 0 <= result["score"] <= 100
    assert isinstance(result["benford_flag"], bool)
    assert isinstance(result["ml_flag"], bool)


# ---------------------------------------------------------------------------
# 5. Uses config.MIN_TRADES_FOR_SCORING as the default minimum
# ---------------------------------------------------------------------------


def test_scorer_uses_config_min_trades_for_scoring(tmp_path):
    with patch("streaming.streaming_scorer.config") as mock_cfg:
        mock_cfg.MIN_TRADES_FOR_SCORING = 42
        mock_cfg.MODEL_DIR = str(tmp_path)
        # RiskScorer.__init__ will fail to load models from tmp_path, but we
        # only need to verify that min_trades picks up the config value.
        with patch("streaming.streaming_scorer.RiskScorer"):
            s = StreamingScorer(model_dir=str(tmp_path))
    assert s.min_trades == 42
