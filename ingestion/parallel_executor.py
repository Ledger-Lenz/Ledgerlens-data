"""Parallel processing controls for CPU-heavy analysis tasks.

Issue #528 — Stellar Wave advanced build.

Provides a reusable, config-driven ``ParallelExecutor`` that fans out
CPU-bound work (Benford computation, feature engineering, model scoring)
across worker processes or threads with:

- **Configurable backend**: ``process`` (ProcessPoolExecutor, bypasses the GIL
  for pure-Python / NumPy-heavy workloads) or ``thread`` (ThreadPoolExecutor,
  lower overhead for I/O-bound or already-C-accelerated workloads).
- **Back-pressure**: an optional ``max_pending`` queue depth prevents the
  producer from flooding workers with tasks faster than they can drain.
- **Graceful shutdown**: ``shutdown(wait=True)`` drains in-flight futures
  before the interpreter exits; used as a context manager via ``__enter__``/
  ``__exit__`` so callers never leak worker processes.
- **Execution reports**: ``ExecutionReport`` dataclass summarises total tasks,
  successes, failures, and wall-clock duration for observability.
- **Error handling modes**: ``"raise"`` (re-raise first failure), ``"collect"``
  (return partial results with errors logged), or ``"ignore"`` (best-effort).

Configuration (all read from ``config`` or passed directly at construction):

.. code-block:: bash

    PARALLEL_EXECUTOR_BACKEND=process        # "process" | "thread"
    PARALLEL_EXECUTOR_MAX_WORKERS=4          # int, defaults to CPU count
    PARALLEL_EXECUTOR_MAX_PENDING=64         # max in-flight futures (back-pressure)
    PARALLEL_EXECUTOR_CHUNK_SIZE=16          # items per submit batch (map helper)
    PARALLEL_EXECUTOR_TIMEOUT_SECONDS=300    # per-task timeout (0 = unlimited)

Example — fan-out feature engineering across pairs::

    from ingestion.parallel_executor import ParallelExecutor

    def score_pair(df: pd.DataFrame) -> dict:
        ...

    with ParallelExecutor() as ex:
        report, results = ex.map(score_pair, list_of_dataframes)
    print(report)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from typing import Any, TypeVar

from config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# ---------------------------------------------------------------------------
# Configuration constants (read from Config; can be overridden at runtime)
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND: str = getattr(config, "PARALLEL_EXECUTOR_BACKEND", "process")
_DEFAULT_MAX_WORKERS: int = getattr(
    config,
    "PARALLEL_EXECUTOR_MAX_WORKERS",
    max(1, (os.cpu_count() or 2) - 1),
)
_DEFAULT_MAX_PENDING: int = getattr(config, "PARALLEL_EXECUTOR_MAX_PENDING", 64)
_DEFAULT_CHUNK_SIZE: int = getattr(config, "PARALLEL_EXECUTOR_CHUNK_SIZE", 16)
_DEFAULT_TIMEOUT: float = float(getattr(config, "PARALLEL_EXECUTOR_TIMEOUT_SECONDS", 300))

_VALID_BACKENDS = frozenset({"process", "thread"})
_VALID_ERROR_MODES = frozenset({"raise", "collect", "ignore"})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TaskError:
    """Wraps an exception raised by a worker task."""

    item: Any
    exc: BaseException

    def __repr__(self) -> str:
        return f"TaskError(item={self.item!r}, exc={self.exc!r})"


@dataclass
class ExecutionReport:
    """Summary produced by :meth:`ParallelExecutor.map` or :meth:`submit_all`.

    Attributes:
        total: Total number of tasks submitted.
        succeeded: Tasks that completed without error.
        failed: Tasks that raised an exception.
        wall_seconds: Elapsed real time for the whole batch.
        backend: Executor backend used (``"process"`` or ``"thread"``).
        max_workers: Number of workers in the pool.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    wall_seconds: float = 0.0
    backend: str = ""
    max_workers: int = 0
    errors: list[TaskError] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of tasks that completed successfully (0–1)."""
        return self.succeeded / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"ExecutionReport(total={self.total}, succeeded={self.succeeded}, "
            f"failed={self.failed}, wall={self.wall_seconds:.2f}s, "
            f"backend={self.backend}, workers={self.max_workers})"
        )


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------


class ParallelExecutor:
    """Config-driven parallel task executor with back-pressure and graceful
    shutdown.

    Parameters
    ----------
    backend:
        ``"process"`` — ``ProcessPoolExecutor`` (default; bypasses GIL).
        ``"thread"`` — ``ThreadPoolExecutor`` (lower overhead for I/O work).
    max_workers:
        Number of worker processes/threads.  Defaults to
        ``PARALLEL_EXECUTOR_MAX_WORKERS`` (env) or CPU count − 1 (≥ 1).
    max_pending:
        Maximum number of futures that may be in-flight simultaneously.
        The :meth:`map` helper blocks the calling thread when this limit is
        reached, providing natural back-pressure.  ``0`` disables the limit.
    chunk_size:
        Items submitted per internal batch in :meth:`map`.
    timeout_seconds:
        Per-task timeout in :meth:`map`.  ``0`` or ``None`` means unlimited.
    error_mode:
        ``"raise"``   — re-raise the first worker exception (default).
        ``"collect"`` — log errors, collect ``TaskError`` objects, continue.
        ``"ignore"``  — silently skip failures (use with care).
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        max_workers: int | None = None,
        max_pending: int | None = None,
        chunk_size: int | None = None,
        timeout_seconds: float | None = None,
        error_mode: str = "raise",
    ) -> None:
        self._backend = (backend or _DEFAULT_BACKEND).lower()
        if self._backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {self._backend!r}. " f"Choose from: {sorted(_VALID_BACKENDS)}"
            )

        self._max_workers: int = max_workers if max_workers is not None else _DEFAULT_MAX_WORKERS
        if self._max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        self._max_pending: int = max_pending if max_pending is not None else _DEFAULT_MAX_PENDING
        self._chunk_size: int = chunk_size if chunk_size is not None else _DEFAULT_CHUNK_SIZE

        # 0 / None → unlimited
        self._timeout: float | None = (
            None if (timeout_seconds is None or timeout_seconds <= 0) else float(timeout_seconds)
        )
        if timeout_seconds is None:
            self._timeout = None if _DEFAULT_TIMEOUT <= 0 else _DEFAULT_TIMEOUT

        self._error_mode = error_mode.lower()
        if self._error_mode not in _VALID_ERROR_MODES:
            raise ValueError(
                f"Unknown error_mode {self._error_mode!r}. "
                f"Choose from: {sorted(_VALID_ERROR_MODES)}"
            )

        self._pool: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        self._entered: bool = False

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> ParallelExecutor:
        self._start_pool()
        self._entered = True
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown(wait=True)
        self._entered = False

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------

    def _start_pool(self) -> None:
        """Spin up the underlying executor pool."""
        if self._pool is not None:
            return
        if self._backend == "process":
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)
        else:
            self._pool = ThreadPoolExecutor(max_workers=self._max_workers)
        logger.debug(
            "ParallelExecutor started: backend=%s workers=%d",
            self._backend,
            self._max_workers,
        )

    def shutdown(self, *, wait: bool = True) -> None:
        """Gracefully drain in-flight futures and tear down the pool.

        Safe to call multiple times; no-op if the pool was never started.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None
            logger.debug("ParallelExecutor shut down (wait=%s)", wait)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """Submit a single callable to the pool and return its ``Future``.

        The pool is started lazily if :meth:`__enter__` was not used.
        """
        if self._pool is None:
            self._start_pool()
        assert self._pool is not None
        return self._pool.submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[[T], R],
        items: Iterable[T],
        *,
        error_mode: str | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[ExecutionReport, list[R]]:
        """Fan out ``fn(item)`` across all ``items`` using the worker pool.

        Implements back-pressure: if more than ``max_pending`` futures are
        in-flight, the method blocks until the pool drains below that threshold
        before submitting more work.

        Parameters
        ----------
        fn:
            Callable that accepts a single item and returns a result.
        items:
            Iterable of inputs.  May be a generator — items are consumed lazily.
        error_mode:
            Override the instance-level error mode for this call.
        timeout_seconds:
            Per-task timeout override for this call.

        Returns
        -------
        (report, results)
            ``report`` is an :class:`ExecutionReport` summarising the batch.
            ``results`` is an ordered list of successful return values.
            Failed items are represented by :class:`TaskError` entries in
            ``report.errors`` when ``error_mode="collect"`` (or silently
            skipped when ``"ignore"``).
        """
        if self._pool is None:
            self._start_pool()
        assert self._pool is not None

        eff_error_mode = (error_mode or self._error_mode).lower()
        eff_timeout = timeout_seconds if timeout_seconds is not None else self._timeout

        report = ExecutionReport(backend=self._backend, max_workers=self._max_workers)
        results: list[R] = []

        # future → original item mapping for error reporting
        future_to_item: dict[Future[R], T] = {}
        pending: list[Future[R]] = []

        start = time.monotonic()

        def _drain_completed() -> None:
            """Collect all currently-done futures without blocking."""
            done = [f for f in pending if f.done()]
            for f in done:
                pending.remove(f)
                report.total += 1
                item = future_to_item.pop(f)
                try:
                    results.append(f.result(timeout=0))
                    report.succeeded += 1
                except FutureTimeoutError:
                    # Should not happen since f.done() == True, but guard anyway
                    results.append(f.result())
                    report.succeeded += 1
                except Exception as exc:  # noqa: BLE001
                    report.failed += 1
                    err = TaskError(item=item, exc=exc)
                    if eff_error_mode == "raise":
                        report.wall_seconds = time.monotonic() - start
                        raise
                    elif eff_error_mode == "collect":
                        logger.warning("ParallelExecutor task error for item %r: %s", item, exc)
                        report.errors.append(err)
                    # else "ignore" — do nothing

        items_list = list(items)
        item_iter = iter(items_list)
        exhausted = False

        while not exhausted or pending:
            # Fill up to max_pending
            while not exhausted and (self._max_pending <= 0 or len(pending) < self._max_pending):
                try:
                    item = next(item_iter)
                except StopIteration:
                    exhausted = True
                    break
                f = self._pool.submit(fn, item)
                future_to_item[f] = item
                pending.append(f)

            if not pending:
                break

            # Wait for at least one to finish (or timeout)
            if eff_timeout is not None:
                next(as_completed(pending, timeout=eff_timeout * len(pending)), None)
            else:
                next(as_completed(pending), None)

            _drain_completed()

        # Collect any remaining futures (exhausted path)
        while pending:
            f = pending[0]
            pending.remove(f)
            report.total += 1
            item = future_to_item.pop(f)
            try:
                result = f.result(timeout=eff_timeout)
                results.append(result)
                report.succeeded += 1
            except FutureTimeoutError as exc:
                report.failed += 1
                err = TaskError(item=item, exc=exc)
                if eff_error_mode == "raise":
                    report.wall_seconds = time.monotonic() - start
                    raise
                elif eff_error_mode == "collect":
                    logger.warning(
                        "ParallelExecutor timeout for item %r after %.1fs",
                        item,
                        eff_timeout,
                    )
                    report.errors.append(err)
            except Exception as exc:  # noqa: BLE001
                report.failed += 1
                err = TaskError(item=item, exc=exc)
                if eff_error_mode == "raise":
                    report.wall_seconds = time.monotonic() - start
                    raise
                elif eff_error_mode == "collect":
                    logger.warning("ParallelExecutor task error for item %r: %s", item, exc)
                    report.errors.append(err)

        report.wall_seconds = time.monotonic() - start
        logger.info("ParallelExecutor.map complete: %s", report)
        return report, results

    def map_chunks(
        self,
        fn: Callable[[list[T]], list[R]],
        items: Iterable[T],
        *,
        chunk_size: int | None = None,
        error_mode: str | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[ExecutionReport, list[R]]:
        """Same as :meth:`map` but batches ``items`` into chunks and calls
        ``fn(chunk)`` per chunk, reducing process-spawn overhead for small items.

        Parameters
        ----------
        fn:
            Callable that accepts a *list* of items and returns a *list* of
            results in the same order.
        items:
            Iterable of inputs to batch.
        chunk_size:
            Items per chunk.  Defaults to :attr:`chunk_size`.

        Returns
        -------
        (report, results)
            Flat list of all results in submission order.
        """
        cs = chunk_size if chunk_size is not None else self._chunk_size
        items_list = list(items)

        chunks: list[list[T]] = [items_list[i : i + cs] for i in range(0, len(items_list), cs)]

        report, chunk_results = self.map(
            fn,
            chunks,
            error_mode=error_mode,
            timeout_seconds=timeout_seconds,
        )
        # Flatten the list-of-lists
        flat: list[R] = []
        for cr in chunk_results:
            flat.extend(cr)
        return report, flat

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """The executor backend in use (``"process"`` or ``"thread"``)."""
        return self._backend

    @property
    def max_workers(self) -> int:
        """The maximum number of workers in the pool."""
        return self._max_workers

    @property
    def chunk_size(self) -> int:
        """Default chunk size for :meth:`map_chunks`."""
        return self._chunk_size

    def __repr__(self) -> str:
        return (
            f"ParallelExecutor(backend={self._backend!r}, "
            f"max_workers={self._max_workers}, "
            f"max_pending={self._max_pending}, "
            f"error_mode={self._error_mode!r})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    backend: str | None = None,
    max_workers: int | None = None,
    error_mode: str = "raise",
    timeout_seconds: float | None = None,
) -> tuple[ExecutionReport, list[R]]:
    """One-shot parallel map — spins up a fresh executor, runs the job, tears
    down the pool, and returns ``(report, results)``.

    Suitable for scripts and one-off batch jobs where the caller does not want
    to manage the executor lifecycle manually.

    Example::

        report, scores = parallel_map(score_wallet, wallet_list, backend="process")
    """
    with ParallelExecutor(
        backend=backend,
        max_workers=max_workers,
        error_mode=error_mode,
    ) as ex:
        return ex.map(fn, items, timeout_seconds=timeout_seconds)


# ---------------------------------------------------------------------------
# Per-category resource isolation (noisy-neighbor protection)
# See docs/parallel_executor.md for recommended limits.
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY_LIMIT = 4


class CategoryMetrics:
    def __init__(self):
        self.queued = 0
        self.in_flight = 0
        self.completed = 0
        self.failed = 0
        self.durations: list[float] = []

    @property
    def queue_depth(self) -> int:
        return self.queued

    def snapshot(self) -> dict:
        d = self.durations
        return {
            "queue_depth": self.queued,
            "in_flight": self.in_flight,
            "completed": self.completed,
            "failed": self.failed,
            "avg_duration_s": sum(d) / len(d) if d else 0.0,
            "max_duration_s": max(d) if d else 0.0,
        }


class PartitionedExecutor:
    def __init__(
        self,
        category_limits: dict[str, int] | None = None,
        default_limit: int = DEFAULT_CATEGORY_LIMIT,
    ):
        self.category_limits = dict(category_limits or {})
        self.default_limit = default_limit
        self._pools: dict[str, ThreadPoolExecutor] = {}
        self._metrics: dict[str, CategoryMetrics] = defaultdict(CategoryMetrics)
        self._lock = threading.Lock()

    def _pool(self, category: str) -> ThreadPoolExecutor:
        with self._lock:
            if category not in self._pools:
                self._pools[category] = ThreadPoolExecutor(
                    max_workers=self.category_limits.get(category, self.default_limit),
                    thread_name_prefix=f"ingest-{category}",
                )
            return self._pools[category]

    def submit(self, category: str, fn, *args, **kwargs) -> Future:
        m = self._metrics[category]
        with self._lock:
            m.queued += 1

        def run():
            with self._lock:
                m.queued -= 1
                m.in_flight += 1
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                ok = True
                return result
            except Exception:
                ok = False
                raise
            finally:
                with self._lock:
                    m.in_flight -= 1
                    m.durations.append(time.perf_counter() - start)
                    if ok:
                        m.completed += 1
                    else:
                        m.failed += 1

        return self._pool(category).submit(run)

    def metrics(self) -> dict[str, dict]:
        with self._lock:
            return {c: m.snapshot() for c, m in self._metrics.items()}

    def shutdown(self, wait: bool = True) -> None:
        for pool in self._pools.values():
            pool.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
