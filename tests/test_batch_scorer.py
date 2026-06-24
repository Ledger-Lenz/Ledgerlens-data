"""Integration tests for detection.batch_scorer."""
import time
from unittest.mock import patch

from detection.batch_scorer import score_batch

WALLETS = [f"wallet_{i}" for i in range(20)]


def _fake_score_one(wallet: str) -> dict:
    time.sleep(0.05)
    return {"wallet": wallet, "score": 0.5, "xlm_balance": 1000.0}


def _bad_score_one(wallet: str) -> dict:
    if wallet == "wallet_0":
        raise ValueError("simulated API failure")
    return {"wallet": wallet, "score": 0.1, "xlm_balance": 0.0}


@patch("detection.batch_scorer._score_one", side_effect=_fake_score_one)
def test_all_wallets_return_results(mock_score):
    results = score_batch(WALLETS)
    assert len(results) == 20
    assert all("wallet" in r for r in results)


@patch("detection.batch_scorer._score_one", side_effect=_fake_score_one)
def test_parallel_is_faster_than_sequential(mock_score):
    """20 wallets × 0.05 s = 1 s sequential; parallel should finish in < 0.4 s."""
    start = time.perf_counter()
    score_batch(WALLETS, max_workers=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.4, f"Batch took {elapsed:.2f}s — not fast enough"


@patch("detection.batch_scorer._score_one", side_effect=_bad_score_one)
def test_individual_error_does_not_crash_batch(mock_score):
    results = score_batch(WALLETS)
    assert len(results) == 20
    errors = [r for r in results if "error" in r]
    assert len(errors) == 1
    assert errors[0]["wallet"] == "wallet_0"
    assert "simulated API failure" in errors[0]["error"]


@patch("detection.batch_scorer._score_one", side_effect=_fake_score_one)
def test_worker_count_respected(mock_score):
    results = score_batch(WALLETS[:5], max_workers=1)
    assert len(results) == 5
