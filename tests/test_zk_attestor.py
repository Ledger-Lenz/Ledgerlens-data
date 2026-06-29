import pandas as pd

from integrations.zk_attestor import ZKAttestor


def test_trade_data_hash_is_deterministic_across_row_and_column_order():
    attestor = ZKAttestor()

    trades_a = pd.DataFrame(
        [
            {"ledger": 2, "amount": 12.5, "wallet": "GA2"},
            {"ledger": 1, "amount": 9.0, "wallet": "GA1"},
        ]
    )
    trades_b = pd.DataFrame(
        [
            {"wallet": "GA1", "amount": 9.0, "ledger": 1},
            {"wallet": "GA2", "amount": 12.5, "ledger": 2},
        ]
    )

    assert attestor.trade_data_hash(trades_a) == attestor.trade_data_hash(trades_b)


def test_generate_and_verify_receipt():
    attestor = ZKAttestor()
    trades = pd.DataFrame(
        [
            {"ledger": 1, "amount": 100.0, "wallet": "GA1"},
            {"ledger": 2, "amount": 99.5, "wallet": "GA1"},
        ]
    )

    receipt = attestor.generate_receipt(
        wallet="GA1",
        trades=trades,
        score=84,
        model_version_hash="sha256:feedface",
    )

    assert attestor.verify_receipt(receipt, trades) is True
    assert receipt.commitment == attestor.build_commitment(
        "GA1", receipt.trade_data_hash, "sha256:feedface", 84
    )


def test_verify_receipt_rejects_tampering():
    attestor = ZKAttestor()
    trades = pd.DataFrame(
        [
            {"ledger": 1, "amount": 100.0, "wallet": "GA1"},
        ]
    )

    receipt = attestor.generate_receipt(
        wallet="GA1",
        trades=trades,
        score=10,
        model_version_hash="sha256:feedface",
    )

    tampered = trades.copy()
    tampered.loc[0, "amount"] = 101.0

    assert attestor.verify_receipt(receipt, tampered) is False
    assert attestor.verify_receipt(receipt, trades) is True
    assert receipt.commitment != attestor.build_commitment(
        "GA1", receipt.trade_data_hash, "sha256:feedface", 11
    )
