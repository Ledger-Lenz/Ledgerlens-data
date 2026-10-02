"""Validation contract for data crossing the untrusted boundary.

Every Horizon / AMM-pool loader converts a raw external API record (JSON
from a REST response or an SSE event) into one of the domain models in
`ingestion.data_models`. Those records originate from a network service
LedgerLens does not control -- a misbehaving relay, a MITM'd HTTP proxy, or
a Horizon fork with a bug can all hand back a record that is *structurally*
valid JSON but semantically nonsense (a NaN amount, a zero-denominator
price fraction, a 10 KB "asset code", an account ID that isn't a real
Stellar key). Nothing downstream -- Benford analysis, the wallet graph,
forensic reports -- validates these fields again, so an untrusted record
that isn't caught here silently corrupts detection output or crashes the
ingestion worker.

The contract: every loader that turns a raw external record into a `Trade`
/ `OrderBookEvent` / `AccountActivity` must pass the result through
`validate_trade` / `validate_orderbook_event` / `validate_account_activity`
before yielding it. Validation failures raise `UntrustedInputError`
(a `ValueError` subclass, so it composes with existing
`except (ValueError, ValidationError)` handlers) -- callers own the
decision to skip, log, and continue rather than crash the whole page or
stream on one bad record. Nothing here mutates or partially accepts a
record: validation is all-or-nothing per record.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, NoReturn

from stellar_sdk import Asset as SdkAsset
from stellar_sdk.exceptions import AssetCodeInvalidError
from stellar_sdk.strkey import StrKey

from ingestion.data_models import AccountActivity, OrderBookEvent, Trade

try:
    from prometheus_client import Counter

    ledgerlens_untrusted_records_rejected_total: Counter | None = Counter(
        "ledgerlens_untrusted_records_rejected_total",
        "Total records from untrusted external sources rejected at the ingestion boundary",
        ["source", "field"],
    )
except ImportError:  # pragma: no cover - exercised only without prometheus_client installed
    ledgerlens_untrusted_records_rejected_total = None

# Defense-in-depth string length cap. Stellar identifiers are all well under
# this; anything longer signals either a malformed upstream or an attempt to
# waste memory/CPU downstream (pandas, logging, forensic report rendering).
MAX_STRING_FIELD_LENGTH = 128

# Stellar public network genesis ledger closed 2015-09-30. Any ledger close
# time before this is not physically possible.
_STELLAR_GENESIS = datetime(2015, 9, 30, tzinfo=UTC)
# Small allowance for clock skew between this host and Horizon.
_FUTURE_SKEW_TOLERANCE = timedelta(minutes=5)

_VALID_ORDERBOOK_ACTIONS = {"created", "cancelled", "updated"}


class UntrustedInputError(ValueError):
    """A record from an untrusted external source failed the ingestion contract.

    Carries `field`, `reason`, and `source` as structured attributes (in
    addition to the human-readable message) so callers can log and
    increment metrics without re-parsing the message string.
    """

    def __init__(self, field: str, reason: str, *, source: str):
        self.field = field
        self.reason = reason
        self.source = source
        super().__init__(f"[{source}] {field}: {reason}")


def _reject(field: str, reason: str, *, source: str) -> NoReturn:
    if ledgerlens_untrusted_records_rejected_total is not None:
        ledgerlens_untrusted_records_rejected_total.labels(source=source, field=field).inc()
    raise UntrustedInputError(field, reason, source=source)


def safe_ratio(numerator: object, denominator: object, *, default: float = 0.0) -> float:
    """Divide two untrusted numeric-ish values, never raising.

    Horizon encodes prices as a `{"n": ..., "d": ...}` fraction; a
    malformed or adversarial record can make `d` zero, non-numeric, or
    absent entirely. Returns `default` for any of those cases (or a
    non-finite result) instead of propagating `ZeroDivisionError`,
    `TypeError`, or `ValueError` into the caller.
    """
    try:
        result = float(numerator) / float(denominator)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _check_bounded_string(value: object, field: str, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        _reject(field, f"expected a non-empty string, got {type(value).__name__}", source=source)
    if len(value) > MAX_STRING_FIELD_LENGTH:
        _reject(
            field,
            f"exceeds max length {MAX_STRING_FIELD_LENGTH} ({len(value)} chars)",
            source=source,
        )
    return value


def _check_finite_amount(
    value: object, field: str, *, source: str, allow_zero: bool = True
) -> float:
    if not isinstance(value, int | float | Decimal) or isinstance(value, bool):
        _reject(field, f"expected a numeric amount, got {type(value).__name__}", source=source)
    if math.isnan(value) or math.isinf(value):  # type: ignore[arg-type]
        _reject(field, "amount is NaN or Inf", source=source)
    if value < 0 or (value == 0 and not allow_zero):  # type: ignore[operator]
        bound = ">= 0" if allow_zero else "> 0"
        _reject(field, f"amount must be {bound}, got {value}", source=source)
    return float(value)  # type: ignore[arg-type]


def _check_account_id(value: object, field: str, *, source: str) -> str:
    account_id = _check_bounded_string(value, field, source=source)
    if not StrKey.is_valid_ed25519_public_key(account_id):
        _reject(field, f"not a valid Stellar ed25519 account ID: {account_id!r}", source=source)
    return account_id


def _check_asset_code(value: object, field: str, *, source: str) -> str:
    code = _check_bounded_string(value, field, source=source)
    if code in ("XLM", "native"):
        return code
    try:
        SdkAsset.check_if_asset_code_is_valid(code)
    except AssetCodeInvalidError as exc:
        _reject(field, str(exc), source=source)
    return code


def _check_ledger_close_time(value: datetime, field: str, *, source: str) -> datetime:
    ts = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    if ts < _STELLAR_GENESIS:
        _reject(field, f"predates Stellar mainnet genesis: {ts.isoformat()}", source=source)
    if ts > now + _FUTURE_SKEW_TOLERANCE:
        _reject(field, f"timestamp is in the future: {ts.isoformat()}", source=source)
    return value


def validate_trade(trade: Trade, *, source: str) -> Trade:
    """Validate a `Trade` built from an untrusted external record.

    Raises `UntrustedInputError` on the first failing field. Returns the
    same `trade` unchanged on success (for convenient chaining).
    """
    _check_bounded_string(trade.trade_id, "trade_id", source=source)
    _check_account_id(trade.base_account, "base_account", source=source)
    _check_account_id(trade.counter_account, "counter_account", source=source)
    _check_asset_code(trade.base_asset.code, "base_asset.code", source=source)
    _check_asset_code(trade.counter_asset.code, "counter_asset.code", source=source)
    _check_finite_amount(trade.base_amount, "base_amount", source=source, allow_zero=False)
    _check_finite_amount(trade.counter_amount, "counter_amount", source=source, allow_zero=False)
    _check_finite_amount(trade.price, "price", source=source, allow_zero=False)
    _check_ledger_close_time(trade.ledger_close_time, "ledger_close_time", source=source)
    return trade


def validate_orderbook_event(event: OrderBookEvent, *, source: str) -> OrderBookEvent:
    """Validate an `OrderBookEvent` built from an untrusted external record."""
    _check_bounded_string(event.event_id, "event_id", source=source)
    _check_account_id(event.account, "account", source=source)
    _check_asset_code(event.selling.code, "selling.code", source=source)
    _check_asset_code(event.buying.code, "buying.code", source=source)
    _check_finite_amount(event.amount, "amount", source=source, allow_zero=True)
    _check_finite_amount(event.price, "price", source=source, allow_zero=True)
    _check_ledger_close_time(event.ledger_close_time, "ledger_close_time", source=source)
    if event.action not in _VALID_ORDERBOOK_ACTIONS:
        _reject("action", f"unrecognised action {event.action!r}", source=source)
    return event


def validate_account_activity(activity: AccountActivity, *, source: str) -> AccountActivity:
    """Validate an `AccountActivity` built from an untrusted external record."""
    _check_account_id(activity.account_id, "account_id", source=source)
    if activity.funding_account is not None:
        _check_account_id(activity.funding_account, "funding_account", source=source)
    _check_ledger_close_time(activity.account_created_at, "account_created_at", source=source)
    return activity


# --- Schema-first raw payload validation (Issue #908) ------------------------
#
# Size and nesting depth are checked on the raw bytes *before* ``json.loads``
# runs, so an oversized or deeply-nested payload never reaches the
# deserializer (which is recursive and would otherwise burn memory/CPU or
# raise RecursionError). Only after parsing succeeds is the structure checked
# against the expected top-level schema. Loaders must call
# ``parse_untrusted_json`` instead of ``json.loads`` on external responses.

MAX_PAYLOAD_BYTES = 5 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_CONTAINER_ITEMS = 10_000


def _check_json_depth(raw: str, max_depth: int, *, source: str) -> None:
    """Scans raw JSON text for bracket depth without deserializing it."""
    depth = 0
    in_string = escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > max_depth:
                _reject("payload", f"nesting depth exceeds {max_depth}", source=source)
        elif ch in "]}":
            depth -= 1


def _check_container_sizes(value: Any, max_items: int, *, source: str) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if len(item) > max_items:
                _reject("payload", f"object has more than {max_items} keys", source=source)
            stack.extend(item.values())
        elif isinstance(item, list):
            if len(item) > max_items:
                _reject("payload", f"array has more than {max_items} items", source=source)
            stack.extend(item)
        elif isinstance(item, str) and len(item) > MAX_PAYLOAD_BYTES:
            _reject("payload", "string value too large", source=source)


def parse_untrusted_json(
    raw: bytes | str,
    *,
    source: str,
    expected_type: type = dict,
    required_keys: tuple[str, ...] = (),
    max_bytes: int = MAX_PAYLOAD_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
    max_items: int = MAX_CONTAINER_ITEMS,
) -> Any:
    """Validates then deserializes an untrusted JSON payload.

    Order: type -> byte size -> UTF-8 decode -> depth scan -> ``json.loads``
    -> container sizes -> top-level schema. Any failure raises
    ``UntrustedInputError``; no other exception type escapes.
    """
    if not isinstance(raw, (bytes, bytearray, str)):
        _reject("payload", f"expected bytes or str, got {type(raw).__name__}", source=source)
    size = len(raw) if isinstance(raw, (bytes, bytearray)) else len(raw.encode("utf-8", "replace"))
    if size > max_bytes:
        _reject("payload", f"size {size} exceeds {max_bytes} bytes", source=source)
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            _reject("payload", "not valid UTF-8", source=source)
    _check_json_depth(raw, max_depth, source=source)
    try:
        value = json.loads(raw, parse_constant=lambda c: _reject("payload", f"{c} not allowed", source=source))
    except UntrustedInputError:
        raise
    except (ValueError, RecursionError) as exc:
        _reject("payload", f"malformed JSON: {type(exc).__name__}", source=source)
    _check_container_sizes(value, max_items, source=source)
    if not isinstance(value, expected_type):
        _reject("payload", f"expected {expected_type.__name__}", source=source)
    if isinstance(value, dict):
        missing = [k for k in required_keys if k not in value]
        if missing:
            _reject("payload", f"missing required keys {missing}", source=source)
    return value
