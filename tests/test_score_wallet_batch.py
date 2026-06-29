import json
from unittest.mock import MagicMock, patch

import pytest


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
        scorer_instance.list_override.check.return_value = None
        scorer_instance.models = {"random_forest": MagicMock()}
        yield scorer_instance


@pytest.fixture
def mock_load_trades():
    with patch("scripts.score_wallet.load_trades") as m_trades:
        m_trades.return_value = iter([])
        yield m_trades


def _wallets(n: int) -> list[str]:
    return [f"G{str(i).zfill(55)}" for i in range(n)]


def test_batch_scores_ten_wallets_from_file(tmp_path, capsys, mock_scorer, mock_load_trades):
    """Integration test: score 10 synthetic wallets from a temp file."""
    wallets = _wallets(10)
    lines = ["# a comment line", "", *wallets, "", "# trailing comment"]
    wallets_file = tmp_path / "wallets.txt"
    wallets_file.write_text("\n".join(lines))

    with patch(
        "sys.argv",
        [
            "score_wallet.py",
            "--wallets-file",
            str(wallets_file),
            "--pair",
            "USDC:G...",
            "--workers",
            "4",
        ],
    ):
        from scripts.score_wallet import main

        main()

    out, _ = capsys.readouterr()
    results = [json.loads(line) for line in out.strip().splitlines()]

    # Blank lines and '#' comments must be skipped — exactly 10 results.
    assert len(results) == 10
    assert {r["wallet"] for r in results} == set(wallets)
    for r in results:
        assert r["error"] is None
        assert r["score"] == 83


def test_batch_per_wallet_error_does_not_abort_others(tmp_path, capsys, mock_scorer, mock_load_trades):
    """A failure scoring one wallet must surface in its `error` field, not crash the batch."""
    good_wallets = _wallets(4)
    bad_wallet = "NOT-A-VALID-WALLET"
    wallets_file = tmp_path / "wallets.txt"
    wallets_file.write_text("\n".join([*good_wallets, bad_wallet]))

    with patch(
        "sys.argv",
        ["score_wallet.py", "--wallets-file", str(wallets_file), "--pair", "USDC:G..."],
    ):
        from scripts.score_wallet import main

        main()

    out, _ = capsys.readouterr()
    results = {r["wallet"]: r for r in (json.loads(line) for line in out.strip().splitlines())}

    assert len(results) == 5
    for w in good_wallets:
        assert results[w]["error"] is None
        assert results[w]["score"] == 83

    assert results[bad_wallet]["error"] is not None
    assert results[bad_wallet]["score"] is None


def test_batch_workers_argument_controls_pool_size(tmp_path, capsys, mock_scorer, mock_load_trades):
    wallets_file = tmp_path / "wallets.txt"
    wallets_file.write_text("\n".join(_wallets(3)))

    with patch("scripts.score_wallet.ThreadPoolExecutor", wraps=__import__(
        "concurrent.futures", fromlist=["ThreadPoolExecutor"]
    ).ThreadPoolExecutor) as mock_pool:
        with patch(
            "sys.argv",
            [
                "score_wallet.py",
                "--wallets-file",
                str(wallets_file),
                "--pair",
                "USDC:G...",
                "--workers",
                "2",
            ],
        ):
            from scripts.score_wallet import main

            main()

    mock_pool.assert_called_once_with(max_workers=2)


def test_load_wallets_from_file_skips_blank_and_comment_lines(tmp_path):
    from scripts.score_wallet import _load_wallets_from_file

    path = tmp_path / "wallets.txt"
    path.write_text("GAAA\n\n# a comment\nGBBB\n   \n#another\nGCCC\n")

    assert _load_wallets_from_file(path) == ["GAAA", "GBBB", "GCCC"]
