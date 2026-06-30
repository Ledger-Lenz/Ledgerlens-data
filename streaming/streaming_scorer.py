"""Thread-safe wrapper around RiskScorer for real-time wallet scoring.

Phase 1 of the real-time detection pipeline (Issue #12).

``RiskScorer.score()`` is stateless (no mutable model state per call), so
``StreamingScorer`` needs no additional locking — concurrent calls from
multiple threads are safe.

GNN incremental inference
-------------------------
When a :class:`~detection.gnn_encoder.GNNEncoder` is supplied, new edges
observed in the stream are forwarded to
:meth:`~detection.gnn_encoder.GNNEncoder.update_node`, which re-computes
only the 1-hop neighbourhood of the affected wallet instead of re-encoding
the full graph.  This keeps latency well under 50 ms per update for graphs
with up to 10,000 nodes.
"""

from __future__ import annotations

import os
import threading

import networkx as nx

from config import config
from detection.feature_cache import FeatureCache
from detection.model_inference import RiskScorer
from streaming.feature_buffer import FeatureBuffer
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Adaptive micro-batch controller (Issue #243)
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_P95 = float(os.getenv("STREAM_TARGET_P95_LATENCY_SECONDS", "2.0"))
_DEFAULT_MIN_BATCH = int(os.getenv("STREAM_MIN_BATCH_SIZE", "1"))
_DEFAULT_MAX_BATCH = int(os.getenv("STREAM_MAX_BATCH_SIZE", "500"))
_DEFAULT_KP = float(os.getenv("STREAM_PID_KP", "0.5"))
_DEFAULT_KI = float(os.getenv("STREAM_PID_KI", "0.1"))
_DEFAULT_KD = float(os.getenv("STREAM_PID_KD", "0.05"))

try:
    from prometheus_client import Gauge, REGISTRY as _PROM_REGISTRY

    _batch_size_gauge: "Gauge | None" = (
        _PROM_REGISTRY._names_to_collectors.get("ledgerlens_adaptive_batch_size")  # type: ignore[attr-defined]
        or Gauge("ledgerlens_adaptive_batch_size", "Current adaptive micro-batch size")
    )
    _target_latency_gauge: "Gauge | None" = (
        _PROM_REGISTRY._names_to_collectors.get("ledgerlens_batch_target_latency_seconds")  # type: ignore[attr-defined]
        or Gauge(
            "ledgerlens_batch_target_latency_seconds",
            "Target p95 latency for adaptive batch controller (seconds)",
        )
    )
except Exception:  # pragma: no cover
    _batch_size_gauge = None
    _target_latency_gauge = None


class AdaptiveBatchController:
    """PID controller that adjusts streaming micro-batch size to meet a p95 latency target.

    The controller decreases batch size when observed p95 latency exceeds the
    target (error > 0) and increases it when latency is comfortably below the
    target, bounded by ``[min_batch, max_batch]``.

    Anti-windup is applied by clamping the integral term to ``_INTEGRAL_CLAMP``,
    preventing unbounded accumulation during sustained overload.

    Parameters
    ----------
    target_p95_latency:
        Desired p95 end-to-end latency in seconds (``STREAM_TARGET_P95_LATENCY_SECONDS``).
    min_batch, max_batch:
        Inclusive bounds on batch size (``STREAM_MIN_BATCH_SIZE`` / ``STREAM_MAX_BATCH_SIZE``).
    kp, ki, kd:
        PID gains (``STREAM_PID_KP`` / ``STREAM_PID_KI`` / ``STREAM_PID_KD``).
        Defaults (0.5, 0.1, 0.05) are tuned for a 2-second target; scale Kp
        down for slower pipelines with high natural variance.
    """

    _INTEGRAL_CLAMP = 50.0

    def __init__(
        self,
        target_p95_latency: float = _DEFAULT_TARGET_P95,
        min_batch: int = _DEFAULT_MIN_BATCH,
        max_batch: int = _DEFAULT_MAX_BATCH,
        kp: float = _DEFAULT_KP,
        ki: float = _DEFAULT_KI,
        kd: float = _DEFAULT_KD,
    ) -> None:
        self.target = target_p95_latency
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._batch_size: float = float(min(32, max_batch))
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._lock = threading.Lock()

        if _target_latency_gauge is not None:
            _target_latency_gauge.set(target_p95_latency)

    @property
    def batch_size(self) -> int:
        return int(self._batch_size)

    def update(self, observed_p95_latency: float) -> int:
        """Observe *observed_p95_latency* and return the new batch size.

        A positive error (latency above target) shrinks the batch; a negative
        error (latency below target) grows it.  Batch size adjustments are
        logged at DEBUG level for PID oscillation diagnosis.
        """
        with self._lock:
            error = observed_p95_latency - self.target

            # Anti-windup: clamp integral before accumulating
            self._integral = max(
                -self._INTEGRAL_CLAMP,
                min(self._INTEGRAL_CLAMP, self._integral + error),
            )
            derivative = error - self._prev_error
            self._prev_error = error

            # Positive error → too slow → reduce batch size (negative adjustment)
            adjustment = -(self.kp * error + self.ki * self._integral + self.kd * derivative)
            self._batch_size = max(
                float(self.min_batch),
                min(float(self.max_batch), self._batch_size + adjustment),
            )

        logger.debug(
            "AdaptiveBatchController: p95=%.3fs error=%.3f adj=%.2f -> batch=%d",
            observed_p95_latency,
            error,
            adjustment,
            self.batch_size,
        )

        if _batch_size_gauge is not None:
            _batch_size_gauge.set(self.batch_size)

        return self.batch_size


class StreamingScorer:
    """Scores a wallet on demand using its buffered trades.

    Returns ``None`` when the wallet has fewer than ``min_trades`` buffered
    trades (not enough history for a reliable score).

    Parameters
    ----------
    model_dir:
        Directory containing trained model artifacts.
    gnn_encoder:
        Optional :class:`~detection.gnn_encoder.GNNEncoder` instance.
        When provided, GNN embeddings are recomputed incrementally on every
        new edge observation via :meth:`observe_new_edges`.
    funding_graph:
        The current wallet funding/co-trade graph.  Required when
        *gnn_encoder* is provided.  May be updated externally as new
        account-activity events arrive.
    feature_cache:
        Optional :class:`~detection.feature_cache.FeatureCache` instance.
        When a wallet is re-scored within the cache's TTL, the buffered
        feature matrix is reused instead of being rebuilt from scratch —
        the dominant cost of repeatedly scoring the same wallet during a
        burst of trade activity. Defaults to a fresh cache configured from
        ``config.FEATURE_CACHE_TTL_SECONDS`` / ``config.FEATURE_CACHE_MAXSIZE``.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        gnn_encoder: GNNEncoder | None = None,  # type: ignore[name-defined]  # noqa: F821
        funding_graph: nx.DiGraph | None = None,
        feature_cache: FeatureCache | None = None,
    ) -> None:
        self._risk_scorer = RiskScorer(model_dir=model_dir)
        self.min_trades: int = config.MIN_TRADES_FOR_SCORING
        self._gnn_encoder = gnn_encoder
        self._funding_graph: nx.DiGraph = (
            funding_graph if funding_graph is not None else nx.DiGraph()
        )

    # ------------------------------------------------------------------
    # Incremental GNN update
    # ------------------------------------------------------------------

    def observe_new_edges(
        self,
        wallet: str,
        new_edges: list[tuple[str, str]],
    ) -> np.ndarray | None:  # type: ignore[name-defined]  # noqa: F821
        """Notify the GNN encoder of new edges and return the updated embedding.

        Re-computes only the 1-hop neighbourhood of *wallet* (not the full
        graph), completing in < 50 ms for a graph with 10,000 nodes.

        Parameters
        ----------
        wallet:
            The wallet whose neighbourhood changed.
        new_edges:
            List of ``(src, dst)`` tuples being added.

        Returns
        -------
        np.ndarray or None
            Updated embedding for *wallet*, or ``None`` if the encoder is
            not configured or torch is unavailable.
        """
        if self._gnn_encoder is None:
            return None

        # Add new edges to the shared graph
        for src, dst in new_edges:
            self._funding_graph.add_edge(src, dst)

        try:
            return self._gnn_encoder.update_node(
                wallet,
                new_edges,
                self._funding_graph,
            )
        except Exception as exc:
            logger.warning("GNN incremental update failed for wallet %s: %s", wallet, exc)
            return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_wallet(self, wallet: str, buffer: FeatureBuffer) -> dict | None:
        """Build feature row from *buffer* and score *wallet*.

        Returns a risk-score dict ``{score, benford_flag, ml_flag, confidence}``
        or ``None`` if the wallet has fewer than ``min_trades`` buffered trades.
        """
        override_val = self._risk_scorer.list_override.check(wallet)
        if override_val in (0, 100):
            return {
                "score": override_val,
                "benford_flag": False,
                "ml_flag": bool(override_val >= 50),
                "confidence": 100,
            }

        if buffer.wallet_trade_count(wallet) < self.min_trades:
            return None

        feature_row = self._feature_cache.get(wallet)
        if feature_row is None:
            feature_row = buffer.get_feature_row(wallet)
            if feature_row is None:
                return None
            self._feature_cache.put(wallet, feature_row)

        try:
            import time
            t0 = time.time()

            # Use score_with_uncertainty when calibration artifacts are available;
            # fall back to score() otherwise.
            if self._risk_scorer.calibrators:
                res = self._risk_scorer.score_with_uncertainty(feature_row)
            else:
                res = self._risk_scorer.score(feature_row)

            latency_ms = (time.time() - t0) * 1000
            model_version = self._risk_scorer.metadata.get("model_version", "unknown") if self._risk_scorer.metadata else "unknown"
            
            logger.info("Wallet scored", extra={
                "wallet": wallet,
                "score": res["score"],
                "latency_ms": latency_ms,
                "model_version": model_version,
                "asset_pair": "unknown"
            })
            return res
        except Exception as exc:
            logger.warning("Scoring failed", exc_info=True, extra={
                "wallet": wallet,
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            })
            return None
