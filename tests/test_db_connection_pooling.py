"""Integration tests for database connection pooling.

Tests that concurrent WebSocket scoring can write to the database without
'OperationalError: database is locked' errors.
"""

import concurrent.futures
import tempfile
import threading
import time
from pathlib import Path

from config import config
from detection.persistence import get_engine, get_session_factory
from detection.risk_score_store import RiskScoreStore


class TestDatabaseConnectionPooling:
    """Integration tests for database connection pooling functionality."""

    def test_concurrent_writes_with_single_store(self):
        """Test that a single RiskScoreStore can handle concurrent writes."""
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            temp_db_url = f"sqlite:///{tmp_file.name}"

        try:
            # Create store with pooled engine
            engine = get_engine(temp_db_url)
            session_factory = get_session_factory(engine)
            store = RiskScoreStore(session_factory)

            def write_scores(thread_id: int, num_writes: int = 10) -> list[str]:
                """Write scores from a single thread."""
                results = []
                for i in range(num_writes):
                    wallet = (
                        f"GTEST{thread_id:03d}{i:03d}" + "A" * 46
                    )  # Valid Stellar wallet format
                    asset_pair = f"XLM:native/TEST{thread_id}:issuer{i}"
                    risk_score = {
                        "score": 50 + i,
                        "benford_flag": i % 2 == 0,
                        "ml_flag": i % 3 == 0,
                        "confidence": 80 + i % 20,
                    }

                    try:
                        store.upsert(wallet, asset_pair, risk_score)
                        results.append(f"thread_{thread_id}_write_{i}_success")
                    except Exception as e:
                        results.append(f"thread_{thread_id}_write_{i}_error_{type(e).__name__}")

                return results

            # Run 10 threads concurrently, each writing 10 scores
            num_threads = 10
            writes_per_thread = 10

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for thread_id in range(num_threads):
                    future = executor.submit(write_scores, thread_id, writes_per_thread)
                    futures.append(future)

                # Collect all results
                all_results = []
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    all_results.extend(results)

            # Verify all writes succeeded
            successful_writes = [r for r in all_results if r.endswith("_success")]
            failed_writes = [r for r in all_results if "_error_" in r]

            assert (
                len(successful_writes) == num_threads * writes_per_thread
            ), f"Expected {num_threads * writes_per_thread} successful writes, got {len(successful_writes)}"

            assert len(failed_writes) == 0, f"Got unexpected failed writes: {failed_writes}"

            print(f"✅ Successfully completed {len(successful_writes)} concurrent database writes")

        finally:
            # Cleanup temp database
            Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)

    def test_multiple_stores_concurrent_writes(self):
        """Test that multiple RiskScoreStore instances can write concurrently."""
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            temp_db_url = f"sqlite:///{tmp_file.name}"

        try:

            def worker_with_own_store(thread_id: int, num_writes: int = 5) -> list[str]:
                """Each thread creates its own store instance."""
                # Each thread gets its own store instance (but shared pooled engine)
                engine = get_engine(temp_db_url)
                session_factory = get_session_factory(engine)
                store = RiskScoreStore(session_factory)

                results = []
                for i in range(num_writes):
                    wallet = (
                        f"GMULT{thread_id:03d}{i:03d}" + "A" * 45
                    )  # Valid Stellar wallet format
                    asset_pair = f"XLM:native/MULTI{thread_id}:issuer{i}"
                    risk_score = {
                        "score": 60 + i,
                        "benford_flag": i % 2 == 1,
                        "ml_flag": i % 3 == 1,
                        "confidence": 90 + i % 10,
                    }

                    try:
                        store.upsert(wallet, asset_pair, risk_score)
                        results.append(f"store_{thread_id}_write_{i}_success")
                    except Exception as e:
                        results.append(f"store_{thread_id}_write_{i}_error_{type(e).__name__}")

                return results

            # Run 20 threads concurrently with separate store instances
            num_threads = 20
            writes_per_thread = 5

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for thread_id in range(num_threads):
                    future = executor.submit(worker_with_own_store, thread_id, writes_per_thread)
                    futures.append(future)

                # Collect all results
                all_results = []
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    all_results.extend(results)

            # Verify all writes succeeded
            successful_writes = [r for r in all_results if r.endswith("_success")]
            failed_writes = [r for r in all_results if "_error_" in r]

            expected_total = num_threads * writes_per_thread
            assert (
                len(successful_writes) == expected_total
            ), f"Expected {expected_total} successful writes, got {len(successful_writes)}"

            assert len(failed_writes) == 0, f"Got unexpected failed writes: {failed_writes}"

            print(
                f"✅ Successfully completed {len(successful_writes)} writes from {num_threads} separate stores"
            )

        finally:
            # Cleanup temp database
            Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)

    def test_read_write_concurrency(self):
        """Test that reads and writes can happen concurrently without blocking."""
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            temp_db_url = f"sqlite:///{tmp_file.name}"

        try:
            engine = get_engine(temp_db_url)
            session_factory = get_session_factory(engine)

            # Pre-populate with some data
            store = RiskScoreStore(session_factory)
            for i in range(100):
                wallet = f"GPREAD{i:03d}" + "A" * 47
                asset_pair = f"XLM:native/READ{i}:issuer"
                risk_score = {
                    "score": i % 100,
                    "benford_flag": i % 2 == 0,
                    "ml_flag": i % 3 == 0,
                    "confidence": i % 100,
                }
                store.upsert(wallet, asset_pair, risk_score)

            results = []
            results_lock = threading.Lock()

            def reader_worker(thread_id: int, num_reads: int = 20):
                """Continuously read data."""
                store = RiskScoreStore(session_factory)
                read_results = []

                for i in range(num_reads):
                    try:
                        # Read existing data
                        wallet = f"GPREAD{(thread_id * num_reads + i) % 100:03d}" + "A" * 47
                        asset_pair = f"XLM:native/READ{(thread_id * num_reads + i) % 100}:issuer"
                        record = store.get(wallet, asset_pair)

                        if record:
                            read_results.append(f"reader_{thread_id}_read_{i}_success")
                        else:
                            read_results.append(f"reader_{thread_id}_read_{i}_notfound")

                        time.sleep(0.001)  # Small delay to interleave with writes
                    except Exception as e:
                        read_results.append(f"reader_{thread_id}_read_{i}_error_{type(e).__name__}")

                with results_lock:
                    results.extend(read_results)

            def writer_worker(thread_id: int, num_writes: int = 20):
                """Continuously write new data."""
                store = RiskScoreStore(session_factory)
                write_results = []

                for i in range(num_writes):
                    try:
                        wallet = f"GWRITE{thread_id:03d}{i:03d}" + "A" * 45
                        asset_pair = f"XLM:native/WRITE{thread_id}:issuer{i}"
                        risk_score = {
                            "score": (thread_id * 10 + i) % 100,
                            "benford_flag": (thread_id + i) % 2 == 0,
                            "ml_flag": (thread_id + i) % 3 == 0,
                            "confidence": (thread_id * 5 + i) % 100,
                        }

                        store.upsert(wallet, asset_pair, risk_score)
                        write_results.append(f"writer_{thread_id}_write_{i}_success")

                        time.sleep(0.001)  # Small delay to interleave with reads
                    except Exception as e:
                        write_results.append(
                            f"writer_{thread_id}_write_{i}_error_{type(e).__name__}"
                        )

                with results_lock:
                    results.extend(write_results)

            # Run 5 readers and 5 writers concurrently
            num_readers = 5
            num_writers = 5

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_readers + num_writers
            ) as executor:
                futures = []

                # Start readers
                for thread_id in range(num_readers):
                    future = executor.submit(reader_worker, thread_id)
                    futures.append(future)

                # Start writers
                for thread_id in range(num_writers):
                    future = executor.submit(writer_worker, thread_id)
                    futures.append(future)

                # Wait for all to complete
                concurrent.futures.wait(futures)

            # Analyze results
            successful_reads = [
                r
                for r in results
                if "_read_" in r and (r.endswith("_success") or r.endswith("_notfound"))
            ]
            failed_reads = [r for r in results if "_read_" in r and "_error_" in r]
            successful_writes = [r for r in results if "_write_" in r and r.endswith("_success")]
            failed_writes = [r for r in results if "_write_" in r and "_error_" in r]

            print(f"Reads: {len(successful_reads)} successful, {len(failed_reads)} failed")
            print(f"Writes: {len(successful_writes)} successful, {len(failed_writes)} failed")

            # All operations should succeed
            assert len(failed_reads) == 0, f"Got unexpected failed reads: {failed_reads}"
            assert len(failed_writes) == 0, f"Got unexpected failed writes: {failed_writes}"
            assert len(successful_writes) == num_writers * 20, f"Expected {num_writers * 20} writes"
            assert len(successful_reads) > 0, "Should have successful reads"

            print("✅ Successfully completed concurrent read/write operations")

        finally:
            # Cleanup temp database
            Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)

    def test_existing_api_unchanged(self):
        """Test that the existing RiskScoreStore API works exactly as before."""
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            temp_db_url = f"sqlite:///{tmp_file.name}"

        try:
            # Test default constructor (should use config values)
            original_db_url = config.RISK_SCORE_DB_URL
            config.RISK_SCORE_DB_URL = temp_db_url

            try:
                store = RiskScoreStore()  # No arguments - should work as before

                # Test basic operations
                wallet = "GTEST" + "A" * 52
                asset_pair = "XLM:native/USDC:issuer"
                risk_score = {"score": 75, "benford_flag": True, "ml_flag": False, "confidence": 85}

                # Test upsert
                record = store.upsert(wallet, asset_pair, risk_score)
                assert record.wallet == wallet
                assert record.asset_pair == asset_pair
                assert record.score == 75
                assert record.benford_flag is True
                assert record.ml_flag is False
                assert record.confidence == 85

                # Test get
                retrieved = store.get(wallet, asset_pair)
                assert retrieved is not None
                assert retrieved.wallet == wallet
                assert retrieved.score == 75

                # Test list_flagged
                flagged = list(store.list_flagged(70))
                assert len(flagged) == 1
                assert flagged[0].wallet == wallet

                # Test get non-existent
                missing = store.get("GMISSING" + "A" * 48, "XLM:native/MISSING:issuer")
                assert missing is None

                print("✅ Existing API works unchanged")

            finally:
                config.RISK_SCORE_DB_URL = original_db_url

        finally:
            # Cleanup temp database
            Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)

    def test_pool_configuration_from_env(self):
        """Test that pool settings can be configured via environment variables."""
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            temp_db_url = f"sqlite:///{tmp_file.name}"

        try:
            # Temporarily override config values
            original_pool_size = config.DB_POOL_SIZE
            original_max_overflow = config.DB_MAX_OVERFLOW
            original_pool_timeout = config.DB_POOL_TIMEOUT

            config.DB_POOL_SIZE = 3
            config.DB_MAX_OVERFLOW = 7
            config.DB_POOL_TIMEOUT = 15

            try:
                engine = get_engine(temp_db_url)

                # Verify pool configuration
                assert engine.pool.size() == 3, f"Expected pool size 3, got {engine.pool.size()}"
                assert (
                    engine.pool._max_overflow == 7
                ), f"Expected max overflow 7, got {engine.pool._max_overflow}"
                assert (
                    engine.pool._timeout == 15
                ), f"Expected timeout 15, got {engine.pool._timeout}"

                print("✅ Pool configuration from environment variables works")

            finally:
                config.DB_POOL_SIZE = original_pool_size
                config.DB_MAX_OVERFLOW = original_max_overflow
                config.DB_POOL_TIMEOUT = original_pool_timeout

        finally:
            # Cleanup temp database
            Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)


# Standalone function for the acceptance criteria test
def test_20_threads_concurrent_scoring():
    """Integration test: 20 threads write scores concurrently, all succeed.

    This is the specific acceptance criteria test mentioned in the issue.
    """
    # Use temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        temp_db_url = f"sqlite:///{tmp_file.name}"

    try:

        def scoring_thread(thread_id: int, scores_per_thread: int = 5) -> dict:
            """Simulate concurrent WebSocket scoring."""
            engine = get_engine(temp_db_url)
            session_factory = get_session_factory(engine)
            store = RiskScoreStore(session_factory)

            results = {
                "thread_id": thread_id,
                "successful_writes": 0,
                "failed_writes": 0,
                "errors": [],
            }

            for i in range(scores_per_thread):
                wallet = f"GSCORE{thread_id:03d}{i:03d}" + "A" * 45
                asset_pair = f"XLM:native/SCORE{thread_id}:issuer{i}"
                risk_score = {
                    "score": (thread_id * 10 + i) % 100,
                    "benford_flag": (thread_id + i) % 2 == 0,
                    "ml_flag": (thread_id + i) % 3 == 0,
                    "confidence": (thread_id * 5 + i) % 100,
                    "propagated_risk": float((thread_id + i) % 100) / 100.0,
                }

                try:
                    store.upsert(wallet, asset_pair, risk_score)
                    results["successful_writes"] += 1

                    # Verify the write worked by reading it back
                    retrieved = store.get(wallet, asset_pair)
                    if not retrieved or retrieved.score != risk_score["score"]:
                        results["errors"].append(f"Write verification failed for {wallet}")

                except Exception as e:
                    results["failed_writes"] += 1
                    results["errors"].append(f"Write failed: {type(e).__name__}: {str(e)}")

            return results

        # Run 20 threads concurrently
        num_threads = 20
        scores_per_thread = 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                future = executor.submit(scoring_thread, thread_id, scores_per_thread)
                futures.append(future)

            # Collect all results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                all_results.append(result)

        # Analyze results
        total_successful = sum(r["successful_writes"] for r in all_results)
        total_failed = sum(r["failed_writes"] for r in all_results)
        all_errors = []
        for r in all_results:
            all_errors.extend(r["errors"])

        expected_total = num_threads * scores_per_thread

        print(f"Total writes: {total_successful + total_failed}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Errors: {len(all_errors)}")

        if all_errors:
            for error in all_errors[:5]:  # Show first 5 errors
                print(f"  Error: {error}")
            if len(all_errors) > 5:
                print(f"  ... and {len(all_errors) - 5} more errors")

        # Acceptance criteria: all writes must succeed
        assert (
            total_successful == expected_total
        ), f"Expected {expected_total} successful writes, got {total_successful}"

        assert (
            total_failed == 0
        ), f"Expected 0 failed writes, got {total_failed}. Errors: {all_errors}"

        print(
            f"✅ ACCEPTANCE CRITERIA MET: {num_threads} threads wrote {scores_per_thread} scores each - all succeeded!"
        )

    finally:
        # Cleanup temp database
        Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run the acceptance criteria test standalone
    test_20_threads_concurrent_scoring()
