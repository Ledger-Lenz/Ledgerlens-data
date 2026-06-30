import pandas as pd

from detection.feature_engineering import (
    compute_order_cancellation_rate,
    compute_trade_pattern_features,
)
from ingestion.orderbook_loader import _action_for_operation, _to_orderbook_event
from tests.factories import make_clean_trades


def test_action_for_operation_created():
    record = {"type": "manage_sell_offer", "amount": "100.0", "offer_id": "0"}
    assert _action_for_operation(record) == "created"


def test_action_for_operation_cancelled():
    record = {"type": "manage_sell_offer", "amount": "0", "offer_id": "12345"}
    assert _action_for_operation(record) == "cancelled"


def test_action_for_operation_updated():
    record = {"type": "manage_buy_offer", "amount": "50.0", "offer_id": "12345"}
    assert _action_for_operation(record) == "updated"


def test_action_for_operation_noop():
    record = {"type": "manage_sell_offer", "amount": "0", "offer_id": "0"}
    assert _action_for_operation(record) is None


def test_action_for_operation_passive_offer_is_created():
    record = {"type": "create_passive_sell_offer", "amount": "10.0", "offer_id": "0"}
    assert _action_for_operation(record) == "created"


def sample_operation_record(**overrides) -> dict:
    record = {
        "id": "op-1",
        "type": "manage_sell_offer",
        "source_account": "GABC",
        "created_at": "2024-01-01T00:00:00Z",
        "selling_asset_type": "credit_alphanum4",
        "selling_asset_code": "USDC",
        "selling_asset_issuer": "GISSUER",
        "buying_asset_type": "native",
        "amount": "100.0",
        "offer_id": "0",
        "price": "0.5",
    }
    record.update(overrides)
    return record


def test_to_orderbook_event_maps_fields():
    event = _to_orderbook_event(sample_operation_record())
    assert event.account == "GABC"
    assert event.action == "created"
    assert event.selling.code == "USDC"
    assert event.buying.code == "XLM"
    assert event.price == 0.5


def test_to_orderbook_event_returns_none_for_noop():
    record = sample_operation_record(amount="0", offer_id="0")
    assert _to_orderbook_event(record) is None


def orderbook_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "1", "account": "A", "action": "created"},
            {"event_id": "2", "account": "A", "action": "cancelled"},
            {"event_id": "3", "account": "A", "action": "cancelled"},
            {"event_id": "4", "account": "B", "action": "created"},
        ]
    )


def test_compute_order_cancellation_rate():
    events = orderbook_events_df()
    assert compute_order_cancellation_rate("A", events) == 2 / 3
    assert compute_order_cancellation_rate("B", events) == 0.0
    assert compute_order_cancellation_rate("C", events) == 0.0


def test_compute_order_cancellation_rate_handles_none():
    assert compute_order_cancellation_rate("A", None) == 0.0


def test_compute_trade_pattern_features_includes_cancellation_rate():
    features = compute_trade_pattern_features("A", pd.DataFrame(), orderbook_events_df())
    assert features["order_cancellation_rate"] == 2 / 3
