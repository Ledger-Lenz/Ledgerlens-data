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

import networkx as nx

from config import config
from detection.feature_cache import FeatureCache
from detection.model_inference import RiskScorer
from streaming.feature_buffer import FeatureBuffer
from utils.logging import get_logger

logger = get_logger(__name__)


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
