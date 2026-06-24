"""Integration test: run_pipeline.main() with --submit-onchain against Testnet.

Run with:
    LEDGERLENS_INTEGRATION_TESTS=1 pytest tests/integration/test_pipeline_submit_onchain.py -v
"""

import importlib
import os
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config as cfg_module
import run_pipeline

pytestmark = pytest.mark.integration

_RETRIES = 3
_RETRY_DELAY = 5


@pytest.mark.timeout(120)
def test_full_pipeline_submit_onchain(live_client, contract_id, submitter_secret, rpc_url):
    """run_pipeline.main() --submit-onchain stores at least one score on-chain."""

    wallet = "GPIPELINETESTWALLETLEDGERLENS00000000000000000000000000001"
    asset_pair = "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVV/XLM:native"

    trades = pd.DataFrame(
        {
            "base_account": [wallet],
            "counter_account": ["GCOUNTERPARTY000000000000000000000000000000000000000000001"],
            "base_amount": ["100.0"],
            "counter_amount": ["50.0"],
            "price": ["0.5"],
            "timestamp": ["2024-01-01T00:00:00Z"],
        }
    )

    feature_matrix = pd.DataFrame(
        {"wallet": [wallet], "benford_mad_1h": [0.02], "benford_chi_square_1h": [5.0]}
    )

    scored = pd.DataFrame(
        {
            "wallet": [wallet],
            "score": [85],
            "benford_flag": [True],
            "ml_flag": [True],
            "confidence": [80],
        }
    )

    fake_scorer = MagicMock()
    fake_scorer.score_matrix.return_value = scored

    env_overrides = {
        "LEDGERLENS_CONTRACT_ID": contract_id,
        "LEDGERLENS_SUBMITTER_SECRET": submitter_secret,
        "SOROBAN_RPC_URL": rpc_url,
        "WATCHED_ASSET_PAIRS": "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVV",
        "RISK_SCORE_FLAG_THRESHOLD": "70",
    }

    with ExitStack() as stack:
        for key, val in env_overrides.items():
            stack.enter_context(patch.dict(os.environ, {key: val}))
        stack.enter_context(
            patch(
                "sys.argv",
                [
                    "run_pipeline.py",
                    "--submit-onchain",
                    "--no-orderbook",
                    "--no-persist",
                ],
            )
        )
        stack.enter_context(
            patch.object(run_pipeline, "load_watched_pairs_to_dataframe", return_value=trades)
        )
        stack.enter_context(
            patch.object(run_pipeline, "build_feature_matrix", return_value=feature_matrix)
        )
        stack.enter_context(patch("detection.model_inference.RiskScorer", return_value=fake_scorer))
        # Reload config so env overrides take effect
        importlib.reload(cfg_module)
        stack.enter_context(patch.object(run_pipeline, "config", cfg_module.Config()))

        run_pipeline.main()

    # Verify the score was stored on-chain
    last_exc = None
    for _ in range(_RETRIES):
        try:
            result = live_client.get_score(wallet, asset_pair)
            assert result["score"] == 85
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(_RETRY_DELAY)

    raise last_exc  # type: ignore[misc]
