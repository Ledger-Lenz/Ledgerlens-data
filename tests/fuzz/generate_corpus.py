#!/usr/bin/env python3
"""Generate valid seed inputs for the fuzz testing corpus.

This script creates 10+ seed inputs (serialized Avro trade records and JSON
API responses) that serve as the starting point for libFuzzer mutation.

Valid seed inputs help the fuzzer explore the valid input space more
efficiently before discovering edge cases and crashes.

Usage:
    python tests/fuzz/generate_corpus.py
"""

import json
import sys
import io
import time
import os
from pathlib import Path
from datetime import datetime, UTC

# Add repo root to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fastavro
from ingestion.data_models import Asset, Trade
from tests.factories import WashTradeFactory, CleanTradeFactory, RingTradeFactory


def _trade_to_record(trade: Trade) -> dict:
    """Convert a Trade to Avro record dict (inline version of avro_codec.trade_to_record)."""
    return {
        "trade_id": trade.trade_id,
        "base_account": trade.base_account,
        "counter_account": trade.counter_account,
        "base_amount": float(trade.base_amount),
        "counter_amount": float(trade.counter_amount),
        "price": float(trade.price),
        "asset_pair": trade.base_asset.pair_id(trade.counter_asset),
        "ledger_close_time": trade.ledger_close_time,
        "ingestion_timestamp_ms": int(time.time() * 1000),
    }


def _serialize(record: dict, schema: dict) -> bytes:
    """Inline avro_codec.serialize to avoid circular imports."""
    fastavro.validation.validate(record, schema, raise_errors=True)
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, schema, record)
    return buffer.getvalue()


def _load_schema() -> dict:
    """Load Avro schema from JSON file."""
    schema_path = os.path.join(
        Path(__file__).parent.parent.parent, 
        "data", 
        "trade_avro_schema.json"
    )
    with open(schema_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return fastavro.parse_schema(raw)


def _generate_avro_seeds() -> list[tuple[str, bytes]]:
    """Generate valid Avro-serialized trade records for the corpus."""
    schema = _load_schema()
    seeds = []
    
    # Generate diverse trade patterns
    factories = [
        ("clean", CleanTradeFactory),
        ("wash_same_amount", WashTradeFactory),
        ("ring_trades", RingTradeFactory),
    ]
    
    for pattern_name, factory in factories:
        # Generate 3 trades from each pattern
        trades = factory.create_batch(3)
        for i, trade in enumerate(trades):
            record = _trade_to_record(trade)
            avro_bytes = _serialize(record, schema)
            seed_name = f"avro_{pattern_name}_{i}.bin"
            seeds.append((seed_name, avro_bytes))
    
    return seeds


def _generate_json_seeds() -> list[tuple[str, bytes]]:
    """Generate valid JSON API responses for the Pydantic fuzz target."""
    seeds = []
    
    # Generate valid Trade JSON
    trades = CleanTradeFactory.create_batch(3)
    for i, trade in enumerate(trades):
        trade_json = {
            "trade_id": trade.trade_id,
            "ledger_close_time": trade.ledger_close_time.isoformat(),
            "base_account": trade.base_account,
            "counter_account": trade.counter_account,
            "base_asset": {
                "code": trade.base_asset.code,
                "issuer": trade.base_asset.issuer,
            },
            "counter_asset": {
                "code": trade.counter_asset.code,
                "issuer": trade.counter_asset.issuer,
            },
            "base_amount": trade.base_amount,
            "counter_amount": trade.counter_amount,
            "price": trade.price,
        }
        seeds.append((f"json_trade_{i}.json", json.dumps(trade_json).encode("utf-8")))
    
    # Generate valid OrderBookEvent JSON
    now = datetime.now(UTC)
    for i in range(3):
        event_json = {
            "event_id": f"event_{i}",
            "account": "GVXYZ1234567890ABCDEFGHIJKLMNOPQRSTUV",
            "ledger_close_time": now.isoformat(),
            "selling": {
                "code": "USDC",
                "issuer": "GBUQWP3BOUZX34ULNQG23RQ6F4OFSAI5BC2D3ZKAB2ZXUCNRC572BL5Z",
            },
            "buying": {
                "code": "XLM",
                "issuer": None,
            },
            "amount": 100.0 + i * 10,
            "price": 1.2 + i * 0.01,
            "action": "created",
        }
        seeds.append((f"json_orderbook_{i}.json", json.dumps(event_json).encode("utf-8")))
    
    # Generate valid AccountActivity JSON
    for i in range(3):
        activity_json = {
            "account_id": f"GACCOUNT{i:06d}0000000000000000000000000000",
            "account_created_at": now.isoformat(),
            "funding_account": f"GFUNDER{i:07d}0000000000000000000000000000",
            "home_domain": f"example{i}.com",
        }
        seeds.append((f"json_activity_{i}.json", json.dumps(activity_json).encode("utf-8")))
    
    # Generate valid Asset JSON
    for i in range(2):
        asset_json = {
            "code": f"USD{i}",
            "issuer": f"GISSUER{i:08d}000000000000000000000000",
        }
        seeds.append((f"json_asset_{i}.json", json.dumps(asset_json).encode("utf-8")))
    
    return seeds


def main():
    """Generate and write all corpus seeds to disk."""
    corpus_dir = Path(__file__).parent / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating corpus seeds in {corpus_dir}...")
    
    # Generate Avro seeds
    avro_seeds = _generate_avro_seeds()
    for seed_name, data in avro_seeds:
        path = corpus_dir / seed_name
        path.write_bytes(data)
        print(f"  ✓ {seed_name} ({len(data)} bytes)")
    
    # Generate JSON seeds
    json_seeds = _generate_json_seeds()
    for seed_name, data in json_seeds:
        path = corpus_dir / seed_name
        path.write_bytes(data)
        print(f"  ✓ {seed_name} ({len(data)} bytes)")
    
    total = len(avro_seeds) + len(json_seeds)
    print(f"\nGenerated {total} seed inputs.")


if __name__ == "__main__":
    main()
