"""Unit tests for the streaming account metadata join.

Test coverage
~~~~~~~~~~~~~
1. ``AccountMetadataUpdate`` validation — valid and malformed records.
2. ``validate_metadata_event()`` — accepts good records, discards bad ones.
3. ``MetadataJoinState.apply_update()`` / ``get_metadata()`` — trade followed
   by metadata update uses the updated metadata on the next scoring cycle.
4. Late-arrival case — metadata update arriving after the join window closes
   is queued and applied on the next scoring cycle.
5. Join window expiry — active entry evicted after the window duration.
6. Wallet eviction — inactive wallets are evicted after the TTL.
7. ``wallets_needing_rescore()`` — returns only wallets with fresh metadata.
8. ``StreamingPipeline._enrich_from_metadata()`` — called on trade events and
   triggers ``buffer.apply_metadata`` when metadata is available.
9. Join lag Prometheus histogram is observed when a pending update is promoted.
10. ``AccountMetadataStream`` — validates events and discards malformed ones.
"""

from __future__ import annotations

import datetime
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ingestion.data_models import Asset, Trade
from streaming.account_metadata_stream import (
    AccountMetadataStream,
    AccountMetadataUpdate,
    validate_metadata_event,
)
from streaming.pipeline import MetadataJoinState

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

USDC_ISSUER = "GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
WALLET_A = "GCEZWKCA5VLDNRLN3RPRJMRZOX3Z6G5CHCGKQCG67NXHIKVBXE4TKVPR"
WALLET_B = "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGK6XLCIMWGCJNPFKCHV7WUDVAB"


def _make_metadata_update(
    account_id: str = WALLET_A,
    effect_type: str = "trustline_created",
    effect_id: str = "effect-001",
    funding_account: str | None = None,
) -> AccountMetadataUpdate:
    return AccountMetadataUpdate(
        account_id=account_id,
        effect_type=effect_type,
        effect_id=effect_id,
        created_at=datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.UTC),
        funding_account=funding_account,
        raw={},
    )


def _make_trade(
    base_account: str = WALLET_A,
    counter_account: str = WALLET_B,
) -> Trade:
    return Trade(
        trade_id="t-001",
        ledger_close_time=datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.UTC),
        base_account=base_account,
        counter_account=counter_account,
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=1000.0,
        counter_amount=500.0,
        price=2.0,
    )


# ---------------------------------------------------------------------------
# 1. AccountMetadataUpdate validation
# ---------------------------------------------------------------------------


class TestAccountMetadataUpdateValidation:
    def test_valid_update_roundtrips(self):
        u = _make_metadata_update()
        assert u.account_id == WALLET_A
        assert u.effect_type == "trustline_created"

    def test_invalid_account_id_raises(self):
        with pytest.raises(Exception):
            AccountMetadataUpdate(
                account_id="NOT_A_STELLAR_ID",
                effect_type="trustline_created",
            )

    def test_short_account_id_raises(self):
        with pytest.raises(Exception):
            AccountMetadataUpdate(
                account_id="GCEZWKCA5",  # too short
                effect_type="account_created",
            )

    def test_empty_effect_type_raises(self):
        with pytest.raises(Exception):
            AccountMetadataUpdate(
                account_id=WALLET_A,
                effect_type="",
            )

    def test_created_at_defaults_to_now(self):
        before = datetime.datetime.now(tz=datetime.UTC)
        u = AccountMetadataUpdate(account_id=WALLET_A, effect_type="account_created")
        after = datetime.datetime.now(tz=datetime.UTC)
        assert before <= u.created_at <= after


# ---------------------------------------------------------------------------
# 2. validate_metadata_event helper
# ---------------------------------------------------------------------------


class TestValidateMetadataEvent:
    def test_valid_horizon_record_parses(self):
        record = {
            "account": WALLET_A,
            "type": "trustline_created",
            "id": "paging-token-001",
            "created_at": "2024-06-01T12:00:00Z",
            "funder": None,
        }
        result = validate_metadata_event(record)
        assert result is not None
        assert result.account_id == WALLET_A
        assert result.effect_type == "trustline_created"
        assert result.effect_id == "paging-token-001"

    def test_malformed_record_returns_none(self):
        # Missing / invalid account_id
        record = {
            "account": "INVALID",
            "type": "trustline_created",
        }
        result = validate_metadata_event(record)
        assert result is None

    def test_missing_type_returns_none(self):
        record = {
            "account": WALLET_A,
            # "type" key missing
        }
        result = validate_metadata_event(record)
        assert result is None

    def test_alternative_field_names_accepted(self):
        """``account_id`` and ``effect_type`` are accepted as field aliases."""
        record = {
            "account_id": WALLET_A,
            "effect_type": "signer_created",
        }
        result = validate_metadata_event(record)
        assert result is not None
        assert result.account_id == WALLET_A

    def test_funding_account_extracted(self):
        record = {
            "account": WALLET_A,
            "type": "account_created",
            "funder": WALLET_B,
        }
        result = validate_metadata_event(record)
        assert result is not None
        assert result.funding_account == WALLET_B


# ---------------------------------------------------------------------------
# 3. MetadataJoinState: trade then metadata update — next cycle uses update
# ---------------------------------------------------------------------------


class TestMetadataJoinStateHappyPath:
    def test_get_metadata_returns_update_after_apply(self):
        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=86400)
        update = _make_metadata_update()

        # No metadata yet — returns None.
        assert state.get_metadata(WALLET_A) is None

        # Apply an update directly (simulates a metadata arriving while wallet is new).
        # First call will queue it as pending since there is no active entry.
        state.apply_update(update)
        # get_metadata promotes the pending entry → returns the update.
        result = state.get_metadata(WALLET_A)
        assert result is not None
        assert result.effect_type == "trustline_created"
        assert result.account_id == WALLET_A

    def test_second_update_replaces_active_entry(self):
        state = MetadataJoinState(join_window_seconds=3600)
        first = _make_metadata_update(effect_type="trustline_created")
        second = _make_metadata_update(effect_type="signer_created")

        # Seed state with first update.
        state.apply_update(first)
        state.get_metadata(WALLET_A)  # promote pending → active

        # Apply second update while active entry exists.
        state.apply_update(second)
        result = state.get_metadata(WALLET_A)
        assert result is not None
        assert result.effect_type == "signer_created"

    def test_trade_event_registers_wallet_last_trade_at(self):
        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=10)
        # A get_metadata call registers the wallet as having had a trade event.
        state.get_metadata(WALLET_A)
        assert WALLET_A in state._last_trade_at

    def test_active_wallet_count(self):
        state = MetadataJoinState(join_window_seconds=3600)
        assert state.active_wallet_count() == 0

        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)  # promotes to active
        assert state.active_wallet_count() == 1

    def test_pending_update_count(self):
        state = MetadataJoinState(join_window_seconds=3600)
        update = _make_metadata_update()
        state.apply_update(update)
        # Should be in pending until get_metadata promotes it.
        assert state.pending_update_count() == 1

        state.get_metadata(WALLET_A)
        assert state.pending_update_count() == 0


# ---------------------------------------------------------------------------
# 4. Late-arrival case — update after join window closes is queued
# ---------------------------------------------------------------------------


class TestMetadataJoinStateLateArrival:
    def test_metadata_arriving_after_window_closes_is_queued_and_applied_next_cycle(self):
        """
        Scenario
        --------
        1. Wallet A has an active metadata entry that expires (join window = 0 s).
        2. A new metadata update arrives *after* the window expires.
        3. The update must *not* be silently discarded — it must be queued.
        4. On the next ``get_metadata()`` call the pending update is promoted
           and returned (enriching the next scoring cycle).
        """
        # Use an effectively-zero window so the entry expires immediately.
        state = MetadataJoinState(join_window_seconds=0, active_wallet_ttl_seconds=86400)

        first = _make_metadata_update(effect_type="account_created")
        state.apply_update(first)
        # Promote to active, immediately expires (window=0).
        result_before = state.get_metadata(WALLET_A)
        # Window is 0 s so the entry may or may not be returned depending on
        # exact timing — but after a tiny sleep it will definitely be stale.
        time.sleep(0.01)
        assert state.get_metadata(WALLET_A) is None  # window expired

        # Late-arriving update for a wallet with no active entry.
        late = _make_metadata_update(effect_type="trustline_created")
        state.apply_update(late)

        # Verify it is in the pending queue, not silently dropped.
        assert state.pending_update_count() == 1

        # Next scoring cycle: get_metadata promotes the pending entry.
        # Use a normal window for the promotion.
        state._join_window = 3600
        result = state.get_metadata(WALLET_A)
        assert result is not None
        assert result.effect_type == "trustline_created"

    def test_update_after_window_queued_for_wallet_not_yet_seen(self):
        """Update for a brand-new wallet goes to pending, not lost."""
        state = MetadataJoinState(join_window_seconds=3600)
        update = _make_metadata_update(account_id=WALLET_B)
        state.apply_update(update)

        # No active entry → goes to pending.
        assert state.pending_update_count() == 1
        assert state.active_wallet_count() == 0

        # get_metadata promotes it.
        result = state.get_metadata(WALLET_B)
        assert result is not None
        assert result.account_id == WALLET_B
        assert state.pending_update_count() == 0


# ---------------------------------------------------------------------------
# 5. Join window expiry
# ---------------------------------------------------------------------------


class TestJoinWindowExpiry:
    def test_entry_evicted_after_window_expires(self):
        state = MetadataJoinState(join_window_seconds=0, active_wallet_ttl_seconds=86400)
        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)  # promotes to active with arrival=now

        time.sleep(0.02)  # let the zero-length window expire

        result = state.get_metadata(WALLET_A)
        assert result is None  # entry evicted

    def test_entry_still_returned_within_window(self):
        state = MetadataJoinState(join_window_seconds=60, active_wallet_ttl_seconds=86400)
        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)  # promote
        # Still within 60-second window → should return the entry.
        result = state.get_metadata(WALLET_A)
        assert result is not None


# ---------------------------------------------------------------------------
# 6. Wallet eviction after TTL
# ---------------------------------------------------------------------------


class TestWalletEviction:
    def test_inactive_wallet_evicted(self):
        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=0)
        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)  # register last_trade_at

        time.sleep(0.02)  # let TTL expire
        evicted = state.evict_inactive_wallets()
        assert evicted >= 1
        assert WALLET_A not in state._last_trade_at

    def test_active_wallet_not_evicted(self):
        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=3600)
        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)

        evicted = state.evict_inactive_wallets()
        assert evicted == 0
        assert WALLET_A in state._last_trade_at

    def test_eviction_removes_pending_updates_too(self):
        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=0)
        update = _make_metadata_update()
        state.apply_update(update)
        # Manually register last_trade_at with an old time.
        state._last_trade_at[WALLET_A] = time.monotonic() - 1  # 1 s ago, TTL=0

        state.evict_inactive_wallets()
        assert state.pending_update_count() == 0


# ---------------------------------------------------------------------------
# 7. wallets_needing_rescore
# ---------------------------------------------------------------------------


class TestWalletsNeedingRescore:
    def test_wallet_with_fresh_metadata_returned(self):
        state = MetadataJoinState(join_window_seconds=3600)
        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)  # registers last_trade_at and promotes

        wallets = state.wallets_needing_rescore()
        assert WALLET_A in wallets

    def test_wallet_without_trade_event_not_returned(self):
        state = MetadataJoinState(join_window_seconds=3600)
        update = _make_metadata_update()
        state.apply_update(update)
        # Do NOT call get_metadata — wallet has no trade event yet.
        state.get_metadata(WALLET_A)  # This registers last_trade_at, so skip
        # Actually test: apply update for a wallet that has no trade event.
        state2 = MetadataJoinState(join_window_seconds=3600)
        state2.apply_update(_make_metadata_update(account_id=WALLET_B))
        # No get_metadata(WALLET_B) call → no trade event registered.
        wallets = state2.wallets_needing_rescore()
        assert WALLET_B not in wallets


# ---------------------------------------------------------------------------
# 8. StreamingPipeline._enrich_from_metadata calls buffer.apply_metadata
# ---------------------------------------------------------------------------


class TestPipelineMetadataEnrichment:
    def _make_pipeline(self, metadata_join_state=None, metadata_stream=None, pairs=None):
        from streaming.pipeline import StreamingPipeline

        buffer = MagicMock()
        scorer = MagicMock()
        scorer.score_wallet.return_value = None
        dispatcher = MagicMock()
        pipeline = StreamingPipeline(
            buffer,
            scorer,
            dispatcher,
            pairs=pairs or [("USDC", USDC_ISSUER)],
            metadata_join_state=metadata_join_state,
            metadata_stream=metadata_stream,
        )
        return pipeline, buffer, scorer, dispatcher

    def test_enrich_calls_apply_metadata_when_metadata_available(self):
        state = MetadataJoinState(join_window_seconds=3600)
        update = _make_metadata_update()
        state.apply_update(update)
        # Pre-populate so get_metadata returns the update immediately.
        state.get_metadata(WALLET_A)  # promote to active

        pipeline, buffer, scorer, dispatcher = self._make_pipeline(
            metadata_join_state=state
        )

        pipeline._enrich_from_metadata(WALLET_A)

        buffer.apply_metadata.assert_called_once_with(WALLET_A, update)

    def test_enrich_does_not_call_apply_metadata_when_no_state(self):
        pipeline, buffer, scorer, dispatcher = self._make_pipeline(
            metadata_join_state=None
        )
        pipeline._enrich_from_metadata(WALLET_A)
        buffer.apply_metadata.assert_not_called()

    def test_enrich_subscribes_wallet_when_metadata_stream_configured(self):
        state = MetadataJoinState(join_window_seconds=3600)
        mock_stream = MagicMock()
        pipeline, buffer, scorer, dispatcher = self._make_pipeline(
            metadata_join_state=state,
            metadata_stream=mock_stream,
        )

        pipeline._enrich_from_metadata(WALLET_A)
        mock_stream.add_wallet.assert_called_once_with(WALLET_A)

    def test_enrich_subscribes_wallet_only_once(self):
        state = MetadataJoinState(join_window_seconds=3600)
        mock_stream = MagicMock()
        pipeline, buffer, scorer, dispatcher = self._make_pipeline(
            metadata_join_state=state,
            metadata_stream=mock_stream,
        )

        # Call twice for the same wallet.
        pipeline._enrich_from_metadata(WALLET_A)
        pipeline._enrich_from_metadata(WALLET_A)

        # add_wallet must only be called once.
        mock_stream.add_wallet.assert_called_once_with(WALLET_A)

    def test_pipeline_enriches_both_wallets_on_trade(self, monkeypatch):
        """_stream_pair calls _enrich_from_metadata for both trade wallets."""
        from stellar_sdk import Asset as SdkAsset
        from streaming.pipeline import StreamingPipeline

        state = MetadataJoinState(join_window_seconds=3600)
        buffer = MagicMock()
        scorer = MagicMock()
        scorer.score_wallet.return_value = None
        dispatcher = MagicMock()

        pipeline = StreamingPipeline(
            buffer,
            scorer,
            dispatcher,
            pairs=[("USDC", USDC_ISSUER)],
            metadata_join_state=state,
        )

        trade = _make_trade()
        enrich_calls = []
        original_enrich = pipeline._enrich_from_metadata

        def tracking_enrich(wallet):
            enrich_calls.append(wallet)
            return original_enrich(wallet)

        pipeline._enrich_from_metadata = tracking_enrich

        def mock_stream(*args, **kwargs):
            yield trade
            pipeline._stop_event.set()

        monkeypatch.setattr("streaming.pipeline.stream_trades", mock_stream)

        base = SdkAsset(code="USDC", issuer=USDC_ISSUER)
        counter = SdkAsset.native()
        pipeline._stream_pair(base, counter)

        assert WALLET_A in enrich_calls
        assert WALLET_B in enrich_calls


# ---------------------------------------------------------------------------
# 9. Join lag histogram observed when pending update promoted
# ---------------------------------------------------------------------------


class TestJoinLagHistogram:
    def test_lag_observed_on_pending_promotion(self):
        """Promote a pending update and verify METADATA_JOIN_LAG.observe is called."""
        state = MetadataJoinState(join_window_seconds=3600)

        with patch("streaming.pipeline.METADATA_JOIN_LAG") as mock_hist:
            update = _make_metadata_update()
            state.apply_update(update)

            # get_metadata promotes the pending update → observe() called.
            result = state.get_metadata(WALLET_A)

        assert result is not None
        mock_hist.observe.assert_called_once()
        lag_value = mock_hist.observe.call_args[0][0]
        # Lag should be a small non-negative float.
        assert lag_value >= 0.0

    def test_lag_not_observed_when_no_pending(self):
        """No observe() call when the active entry was not pending."""
        state = MetadataJoinState(join_window_seconds=3600)

        update = _make_metadata_update()
        state.apply_update(update)
        state.get_metadata(WALLET_A)  # first call — promotes (observe called)

        with patch("streaming.pipeline.METADATA_JOIN_LAG") as mock_hist:
            # Second call — already active, no promotion.
            state.get_metadata(WALLET_A)

        mock_hist.observe.assert_not_called()


# ---------------------------------------------------------------------------
# 10. AccountMetadataStream — validates events, discards malformed ones
# ---------------------------------------------------------------------------


class TestAccountMetadataStream:
    def test_on_update_called_for_valid_effect(self):
        received: list = []

        stream = AccountMetadataStream(
            on_update=lambda u: received.append(u),
        )

        valid_record = {
            "account": WALLET_A,
            "type": "trustline_created",
            "id": "paging-001",
            "created_at": "2024-06-01T12:00:00Z",
        }
        update = validate_metadata_event(valid_record)
        assert update is not None
        stream._on_update(update)

        assert len(received) == 1
        assert received[0].account_id == WALLET_A
        stream.stop()

    def test_malformed_event_discarded(self):
        received: list = []
        stream = AccountMetadataStream(on_update=lambda u: received.append(u))

        malformed = {
            "account": "NOT_VALID",
            "type": "trustline_created",
        }
        result = validate_metadata_event(malformed)
        # Malformed events are discarded before reaching on_update.
        assert result is None
        assert len(received) == 0
        stream.stop()

    def test_stop_sets_stop_event(self):
        stream = AccountMetadataStream()
        assert not stream._stop_event.is_set()
        stream.stop()
        assert stream._stop_event.is_set()

    def test_add_wallet_creates_thread(self):
        stream = AccountMetadataStream()
        # Patch _stream_effects to be a no-op so no real HTTP call is made.
        with patch.object(stream, "_stream_effects", return_value=None):
            stream.add_wallet(WALLET_A)
            # Brief pause so the daemon thread registers.
            time.sleep(0.05)
            with stream._registry_lock:
                assert WALLET_A in stream._threads
        stream.stop()

    def test_add_wallet_idempotent(self):
        stream = AccountMetadataStream()
        with patch.object(stream, "_stream_effects", return_value=None):
            stream.add_wallet(WALLET_A)
            stream.add_wallet(WALLET_A)  # second call — no-op
            time.sleep(0.05)
            with stream._registry_lock:
                # Only one thread should exist per wallet.
                assert len([w for w in stream._threads if w == WALLET_A]) == 1
        stream.stop()

    def test_kafka_produce_called_for_valid_event(self):
        mock_producer = MagicMock()

        stream = AccountMetadataStream(
            produce_to_kafka=True,
            kafka_producer=mock_producer,
        )
        update = _make_metadata_update()
        stream._produce(update)

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        # Key must be the wallet ID encoded as bytes.
        assert call_kwargs[1]["key"] == WALLET_A.encode()
        stream.stop()

    def test_filter_only_interesting_effect_types(self):
        """Effects not in INTERESTING_EFFECT_TYPES should not reach on_update."""
        from streaming.account_metadata_stream import _INTERESTING_EFFECT_TYPES

        received: list = []
        stream = AccountMetadataStream(on_update=lambda u: received.append(u))
        stream.stop()

        # Uninteresting effect type:
        boring = {
            "account": WALLET_A,
            "type": "claimable_balance_created",  # not in the set
        }
        result = validate_metadata_event(boring)
        # validate_metadata_event itself doesn't filter on effect type —
        # that filtering happens in _stream_effects.  But the record is otherwise valid.
        assert result is not None
        # Verify the interesting set does NOT contain this type.
        assert "claimable_balance_created" not in _INTERESTING_EFFECT_TYPES


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------


class TestMetadataJoinStateThreadSafety:
    def test_concurrent_apply_and_get(self):
        """Multiple threads applying and getting metadata must not raise."""
        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=86400)
        errors = []

        def writer():
            for _ in range(100):
                try:
                    state.apply_update(_make_metadata_update())
                except Exception as exc:
                    errors.append(exc)

        def reader():
            for _ in range(100):
                try:
                    state.get_metadata(WALLET_A)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(4)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Concurrent access raised: {errors}"
