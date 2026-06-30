"""Horizon account-effects subscriber + Kafka producer for account metadata updates.

This module implements the ingestion side of the streaming metadata join:

1. ``AccountMetadataStream`` subscribes to the Stellar Horizon
   ``/accounts/{wallet}/effects`` SSE endpoint for a set of wallets.
2. Each received effect is validated against the ``AccountMetadataUpdate``
   schema (malformed events are discarded and logged).
3. Valid events are serialised to JSON and produced to
   ``METADATA_TOPIC`` (default: ``ledgerlens.account_metadata``), keyed by
   ``wallet_id`` so all updates for a wallet land in the same partition.

Security
--------
All incoming Horizon effect payloads are validated before being admitted to
join state.  Validation checks that required fields are present and that
``account_id`` is a plausible Stellar account ID (G-prefixed, 56 chars).
Malformed events are logged at WARNING level and silently discarded — they
never reach the ``MetadataJoinState`` or the scorer.

Threading
---------
One daemon thread is spawned per watched wallet (or per batch when
``batch_size`` is set).  Threads share a single ``_stop_event`` so that
``stop()`` terminates all subscription threads promptly.

Kafka integration
-----------------
When ``STREAMING_BACKEND=kafka`` the module produces metadata update events
to ``METADATA_TOPIC``; the ``MetadataJoinState`` inside ``StreamingPipeline``
then consumes them.  When the SSE backend is used the caller can inject
events directly via ``MetadataJoinState.apply_update()``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from config import config
from utils.logging import get_logger
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Effect types we care about — anything that changes account state relevant to
# feature computation (trust lines, signers, account creation).
_INTERESTING_EFFECT_TYPES = frozenset(
    {
        "account_created",
        "account_removed",
        "trustline_created",
        "trustline_removed",
        "trustline_updated",
        "trustline_authorized",
        "trustline_deauthorized",
        "signer_created",
        "signer_removed",
        "signer_updated",
        "data_created",
        "data_removed",
        "data_updated",
        "account_inflation_destination_updated",
        "account_thresholds_updated",
        "account_home_domain_updated",
        "account_flags_updated",
    }
)

# Stellar account ID format: G + 55 base-32 characters (uppercase A-Z and 2-7).
_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class AccountMetadataUpdate(BaseModel):
    """Validated account metadata update event from Horizon effects.

    Fields mirror the Horizon effect record format where possible so
    that the raw response can be passed almost directly to the constructor.

    Only ``account_id`` and ``effect_type`` are required; all others are
    optional because different effect types carry different payloads.
    """

    account_id: str = Field(description="The Stellar account ID (G…) affected by the effect")
    effect_type: str = Field(description="Horizon effect type string, e.g. 'trustline_created'")
    effect_id: str = Field(default="", description="Horizon paging token for this effect")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Ledger-close timestamp for the effect",
    )
    funding_account: str | None = Field(
        default=None, description="Funder (only present on account_created effects)"
    )
    home_domain: str | None = Field(
        default=None, description="Home domain (present on account_home_domain_updated)"
    )
    # Raw payload preserved for downstream consumers that need additional fields.
    raw: dict = Field(default_factory=dict, description="Original Horizon effect record")

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        if not _STELLAR_ACCOUNT_RE.match(v):
            raise ValueError(
                f"account_id {v!r} is not a valid Stellar account ID — "
                "expected G followed by 55 base-32 characters"
            )
        return v

    @field_validator("effect_type")
    @classmethod
    def validate_effect_type(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("effect_type must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Validation helper (used by pipeline join state for inbound Kafka messages)
# ---------------------------------------------------------------------------


def validate_metadata_event(record: dict) -> AccountMetadataUpdate | None:
    """Parse and validate a raw effect dict into an ``AccountMetadataUpdate``.

    Returns ``None`` and logs a warning if validation fails so that callers
    can safely discard malformed events without raising exceptions.
    """
    try:
        # Horizon uses "account" as the field name for the affected wallet.
        account_id = record.get("account") or record.get("account_id", "")
        effect_type = record.get("type") or record.get("effect_type", "")
        return AccountMetadataUpdate(
            account_id=account_id,
            effect_type=effect_type,
            effect_id=str(record.get("id") or record.get("effect_id") or ""),
            created_at=_parse_created_at(record.get("created_at")),
            funding_account=record.get("funder") or record.get("funding_account"),
            home_domain=record.get("home_domain"),
            raw=record,
        )
    except Exception as exc:
        logger.warning(
            "Discarding malformed account metadata event (validation failed): %s — record keys: %s",
            exc,
            list(record.keys()),
        )
        return None


def _parse_created_at(value) -> datetime:
    if value is None:
        return datetime.now(tz=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Horizon effects subscriber
# ---------------------------------------------------------------------------


class AccountMetadataStream:
    """Subscribes to Horizon effects for a set of wallets and forwards updates.

    Two usage modes:
    * **SSE mode** (``STREAMING_BACKEND=sse``): the ``on_update`` callback is
      called directly in the subscription thread with each
      ``AccountMetadataUpdate``.  Use this to feed a ``MetadataJoinState``
      instance in the same process.
    * **Kafka mode** (``STREAMING_BACKEND=kafka``): validated events are
      serialised to JSON and produced to ``METADATA_TOPIC``.

    Parameters
    ----------
    wallets:
        Initial set of wallet IDs to subscribe to.  Call ``add_wallet()`` at
        runtime to subscribe to additional wallets.
    on_update:
        Callback invoked (in the subscription thread) for every validated
        ``AccountMetadataUpdate``.  Must be thread-safe.
    kafka_producer:
        A ``confluent_kafka.Producer`` instance.  Required when
        ``STREAMING_BACKEND=kafka`` and ``produce_to_kafka=True``.
    produce_to_kafka:
        When ``True``, validated events are also produced to Kafka.
    """

    def __init__(
        self,
        wallets: list[str] | None = None,
        on_update: Callable[[AccountMetadataUpdate], None] | None = None,
        kafka_producer=None,
        produce_to_kafka: bool = False,
    ) -> None:
        self._on_update = on_update
        self._kafka_producer = kafka_producer
        self._produce_to_kafka = produce_to_kafka and kafka_producer is not None
        self._stop_event = threading.Event()
        # Registry: wallet_id → Thread (one per wallet).
        self._registry_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

        for wallet in wallets or []:
            self.add_wallet(wallet)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_wallet(self, wallet: str) -> None:
        """Start a subscription thread for *wallet* if not already subscribed."""
        with self._registry_lock:
            if wallet in self._threads:
                return
            t = threading.Thread(
                target=self._subscribe,
                args=(wallet,),
                name=f"metadata-{wallet[:8]}",
                daemon=True,
            )
            self._threads[wallet] = t
            t.start()
            logger.debug("Started metadata subscription for wallet %s", wallet)

    def stop(self) -> None:
        """Signal all subscription threads to exit."""
        self._stop_event.set()

    def is_running(self) -> bool:
        """Return True if any subscription thread is still alive."""
        with self._registry_lock:
            return any(t.is_alive() for t in self._threads.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _subscribe(self, wallet: str) -> None:
        """Subscription loop for a single wallet — runs in its own thread."""
        cursor = "now"
        consecutive_failures = 0
        max_failures = config.HORIZON_MAX_RETRIES

        while not self._stop_event.is_set():
            try:
                self._stream_effects(wallet, cursor)
                # Generator exhausted without error — reconnect from latest cursor.
                consecutive_failures = 0
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                consecutive_failures += 1
                backoff = min(2 ** consecutive_failures, 60)
                logger.warning(
                    "Effects stream for %s failed (attempt %d/%d): %s — "
                    "reconnecting in %ds",
                    wallet,
                    consecutive_failures,
                    max_failures,
                    exc,
                    backoff,
                )
                if consecutive_failures >= max_failures:
                    logger.error(
                        "Effects stream for %s exceeded %d consecutive failures — "
                        "giving up subscription",
                        wallet,
                        max_failures,
                    )
                    return
                time.sleep(backoff)

    def _stream_effects(self, wallet: str, cursor: str) -> None:
        """Open a single SSE connection and iterate until disconnection."""
        from stellar_sdk import Server

        server = Server(horizon_url=config.HORIZON_URL)
        call_builder = (
            server.effects()
            .for_account(wallet)
            .cursor(cursor)
            .order(asc=True)
        )

        for record in call_builder.stream():
            if self._stop_event.is_set():
                return

            # Filter to effect types that affect wallet-graph features.
            if record.get("type") not in _INTERESTING_EFFECT_TYPES:
                continue

            update = validate_metadata_event(record)
            if update is None:
                # Validation already logged a warning.
                continue

            if self._on_update is not None:
                try:
                    self._on_update(update)
                except Exception as exc:
                    logger.error(
                        "on_update callback raised for wallet %s: %s",
                        wallet,
                        exc,
                    )

            if self._produce_to_kafka:
                self._produce(update)

    def _produce(self, update: AccountMetadataUpdate) -> None:
        """Serialise *update* to JSON and produce it to the metadata Kafka topic."""
        topic = config.METADATA_TOPIC
        key = update.account_id.encode()
        value = json.dumps(
            {
                "account_id": update.account_id,
                "effect_type": update.effect_type,
                "effect_id": update.effect_id,
                "created_at": update.created_at.isoformat(),
                "funding_account": update.funding_account,
                "home_domain": update.home_domain,
                "raw": update.raw,
            }
        ).encode()

        try:
            self._kafka_producer.produce(
                topic,
                key=key,
                value=value,
                on_delivery=self._on_delivery,
            )
            self._kafka_producer.poll(0)  # Trigger delivery callbacks.
        except Exception as exc:
            logger.error(
                "Failed to produce metadata update for wallet %s to topic %s: %s",
                update.account_id,
                topic,
                exc,
            )

    @staticmethod
    def _on_delivery(err, msg) -> None:
        if err:
            logger.error(
                "Metadata event delivery failed to %s[%d]: %s",
                msg.topic(),
                msg.partition(),
                err,
            )
