"""Top-level orchestrator for the real-time detection pipeline.

StreamingPipeline starts one daemon thread per watched asset pair, drives
each through stream_trades(), and wires the FeatureBuffer → StreamingScorer →
AlertDispatcher chain.

Reconnection on Horizon SSE failures is handled at two levels:
  1. stream_trades() retries internally (up to max_reconnect_attempts).
  2. _stream_pair() restarts the generator if stream_trades() raises after
     exhausting its own retries.

Circuit breakers
----------------
Pass ``circuit_breakers`` to wrap the model-inference call with independent
per-component ``CircuitBreaker`` instances.  Supported keys:
  - ``"inference"``  — wraps ``scorer.score_wallet()``
  - ``"db_write"``   — wraps ``dispatcher.dispatch()``
  - ``"benford"``    — wraps any explicit Benford computation (future extension)

When a circuit is OPEN the pipeline falls back to the last cached score for
the wallet-pair so alert delivery can continue uninterrupted.

Shutdown
--------
Call pipeline.run() from the main thread.  SIGINT (Ctrl-C) sets the internal
stop event via a signal handler; the main loop wakes up, joins all worker
threads with a 5-second timeout, and returns.
"""

import signal
import threading
import time

from stellar_sdk import Asset as SdkAsset

from config import config
from ingestion.horizon_streamer import stream_trades
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer, StreamingScorer
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.logging import get_logger

logger = get_logger(__name__)


class StreamingPipeline:
    """Orchestrates one SSE thread per pair and wires the scoring pipeline."""

    def __init__(
        self,
        buffer: FeatureBuffer,
        scorer: StreamingScorer,
        dispatcher: AlertDispatcher,
        pairs: list[tuple[str, str]] | None = None,
        circuit_breakers: dict[str, CircuitBreaker] | None = None,
    ):
        self._buffer = buffer
        self._scorer = scorer
        self._dispatcher = dispatcher
        self._pairs = list(pairs) if pairs is not None else list(config.WATCHED_ASSET_PAIRS)
        self._stop_event = threading.Event()
        self._worker_threads: list[threading.Thread] = []
        self._circuit_breakers: dict[str, CircuitBreaker] = circuit_breakers or {}
        # Wallet → last successfully computed score (used as fallback when OPEN)
        self._score_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start one thread per pair, block until KeyboardInterrupt or stop()."""
        sdk_pairs = self._build_sdk_pairs()
        if not sdk_pairs:
            logger.warning("No asset pairs configured — streaming pipeline has nothing to do")
            return

        # Install SIGINT handler when called from the main thread so that
        # Ctrl-C sets the stop event rather than raising KeyboardInterrupt
        # mid-iteration inside a worker thread.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())

        self._worker_threads = []
        for base_asset, counter_asset in sdk_pairs:
            t = threading.Thread(
                target=self._stream_pair,
                args=(base_asset, counter_asset),
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

        logger.info("Streaming pipeline running with %d pair(s)", len(self._worker_threads))

        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            logger.info("Shutting down — joining worker threads (timeout=5s)")
            for t in self._worker_threads:
                t.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_wallet_with_cb(self, wallet: str) -> dict | None:
        cb = self._circuit_breakers.get("inference")
        if cb is None:
            score = self._scorer.score_wallet(wallet)
        else:
            try:
                score = cb.call(self._scorer.score_wallet, wallet)
            except CircuitOpenError:
                score = self._score_cache.get(wallet)
                if score is not None:
                    logger.debug(
                        "Inference circuit OPEN — serving cached score for %s", wallet
                    )
        if score is not None:
            self._score_cache[wallet] = score
        return score

    def _dispatch_with_cb(self, wallet: str, score: dict, pair_id: str) -> None:
        cb = self._circuit_breakers.get("db_write")
        if cb is None:
            self._dispatcher.dispatch(wallet, score, pair_id)
        else:
            try:
                cb.call(self._dispatcher.dispatch, wallet, score, pair_id)
            except CircuitOpenError:
                logger.debug("DB-write circuit OPEN — alert for %s suppressed", wallet)

    def _build_sdk_pairs(self) -> list[tuple[SdkAsset, SdkAsset]]:
        xlm = SdkAsset.native()
        pairs = []
        for code, issuer in self._pairs:
            asset = SdkAsset.native() if issuer == "native" else SdkAsset(code, issuer)
            if asset == xlm:
                continue
            pairs.append((asset, xlm))
        return pairs

    def _stream_pair(self, base_asset: SdkAsset, counter_asset: SdkAsset) -> None:
        pair_label = (
            f"{base_asset.code}:{getattr(base_asset, 'issuer', None) or 'native'}"
            f"/{counter_asset.code}:{getattr(counter_asset, 'issuer', None) or 'native'}"
        )
        while not self._stop_event.is_set():
            try:
                for trade in stream_trades(base_asset, counter_asset):
                    if self._stop_event.is_set():
                        return
                    self._buffer.update(trade)
                    pair_id = trade.base_asset.pair_id(trade.counter_asset)
                    for wallet in (trade.base_account, trade.counter_account):
                        score = self._score_wallet_with_cb(wallet)
                        if score is not None:
                            self._dispatch_with_cb(wallet, score, pair_id)
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                logger.warning(
                    "Stream error for pair %s: %s — will reconnect",
                    pair_label,
                    exc,
                )
