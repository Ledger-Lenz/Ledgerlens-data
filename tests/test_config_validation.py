"""Tests for `run_pipeline.py`'s `--dry-run` flag.

`--dry-run` must run every pipeline stage (ingestion, feature engineering,
scoring) while suppressing all writes: no `RiskScoreStore.upsert` and no
`LedgerLensContractClient.submit_score`. Flagged wallets are still logged.
"""

import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd

import run_pipeline
from config import Config


def _run_dry_run(argv: list[str]) -> MagicMock:
    """Run `run_pipeline.main()` with the given args and all I/O stubbed out.

    Ingestion, feature engineering, and scoring are replaced with fakes so the
    pipeline produces a flagged wallet without touching Horizon, the DB, or the
    network. Returns the MagicMock standing in for `RiskScorer`.
    """
    trades = pd.DataFrame({"base_account": ["GA"], "counter_account": ["GB"]})
    feature_matrix = pd.DataFrame({"wallet": ["GA", "GB"], "benford_mad_1h": [0.02, 0.0]})
    scored = pd.DataFrame(
        {
            "wallet": ["GA", "GB"],
            "score": [85, 10],
            "benford_flag": [True, False],
            "ml_flag": [True, False],
            "confidence": [90, 30],
        }
    )

    fake_scorer = MagicMock()
    fake_scorer.score_matrix.return_value = scored

    with ExitStack() as stack:
        stack.enter_context(patch("sys.argv", ["run_pipeline.py", *argv]))
        stack.enter_context(
            patch.object(
                Config,
                "WATCHED_ASSET_PAIRS",
                [("USDC", "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")],
            )
        )
        stack.enter_context(
            patch.object(run_pipeline, "load_pair_to_dataframe", return_value=trades)
        )
        stack.enter_context(
            patch.object(run_pipeline, "build_feature_matrix", return_value=feature_matrix)
        )
        stack.enter_context(patch("detection.model_inference.RiskScorer", return_value=fake_scorer))
        run_pipeline.main()

    return fake_scorer


def test_dry_run_skips_persist():
    with patch("run_pipeline.RiskScoreStore.upsert") as upsert:
        _run_dry_run(["--dry-run", "--no-orderbook"])
    upsert.assert_not_called()


def test_dry_run_skips_submit_onchain():
    with patch(
        "integrations.contract_client.LedgerLensContractClient.submit_score"
    ) as submit_score:
        _run_dry_run(["--dry-run", "--submit-onchain", "--no-orderbook"])
    submit_score.assert_not_called()


def test_dry_run_still_prints_flagged(caplog):
    with caplog.at_level(logging.INFO):
        _run_dry_run(["--dry-run", "--no-orderbook"])
    # The flagged wallet (score 85 >= threshold) is logged in the summary line.
    assert "Flagged wallets" in caplog.text
    assert "GA" in caplog.text


def test_dry_run_log_banner(caplog):
    with caplog.at_level(logging.INFO):
        _run_dry_run(["--dry-run", "--no-orderbook"])
    assert "[DRY RUN] No data will be written." in caplog.text


def test_validate_passes_with_required_vars_set(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("XLM", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")

    Config.validate()
