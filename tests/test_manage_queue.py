"""Unit tests for scripts/manage_queue.py.

Tests each subcommand using a temp queue file.  HMAC signing is enabled
so that the annotated/skip re-sign code paths are exercised.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from detection.active_learning.queue_io import save_queue

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SECRET = "test-hmac-secret-for-manage-queue-tests"


def _read_json(path: Path) -> list[dict]:
    """Read the queue file and extract annotations from dict or list format."""
    data = json.loads(path.read_bytes())
    if isinstance(data, dict):
        return data.get("annotations", [])
    if isinstance(data, list):
        return data
    return []


@pytest.fixture
def queue_path(tmp_path: Path) -> Path:
    """Return a temp path and seed it with a few queue entries."""
    p = tmp_path / "annotation_queue.json"
    annotations = [
        {"wallet": "GAAA...", "status": "pending", "query_strategy": "entropy"},
        {"wallet": "GBBB...", "status": "pending", "query_strategy": "margin"},
        {
            "wallet": "GCCC...",
            "status": "annotated",
            "label": 1,
            "annotator_id": "alice",
            "annotated_at": "2026-06-01T12:00:00",
            "notes": "obvious wash",
            "annotation_hmac": "",
        },
        {
            "wallet": "GDDD...",
            "status": "skipped",
            "query_strategy": "entropy",
        },
    ]
    save_queue(p, annotations, SECRET)
    # Fix the HMAC for the annotated entry so it passes verification
    data = json.loads(p.read_text())
    for ann in data["annotations"]:
        if ann.get("status") == "annotated":
            msg = f"{ann['wallet']}|{ann['label']}|{ann['annotator_id']}|{ann['annotated_at']}".encode()
            ann["annotation_hmac"] = hmac.new(
                SECRET.encode(), msg, hashlib.sha256
            ).hexdigest()
    save_queue(p, data["annotations"], SECRET)
    return p


def _run(args: list[str], queue: Path) -> str:
    """Run manage_queue.main() with the given CLI args and return stdout."""
    from scripts.manage_queue import main

    test_argv = ["manage_queue.py", "--queue", str(queue)] + args
    old_argv = sys.argv
    old_stdout = sys.stdout
    sys.argv = test_argv
    captured = StringIO()
    sys.stdout = captured
    try:
        main()
    except SystemExit as exc:
        if exc.code != 0:
            raise RuntimeError(f"manage_queue exited with code {exc.code}")
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
    return captured.getvalue()


# ===================================================================
# Tests
# ===================================================================


class TestList:
    def test_list_all(self, queue_path: Path):
        """list without filters shows all entries."""
        out = _run(["list"], queue=queue_path)
        assert "GAAA..." in out
        assert "GBBB..." in out
        assert "GCCC..." in out
        assert "GDDD..." in out
        assert "Showing 4 of 4" in out

    def test_list_status_filter(self, queue_path: Path):
        """list --status pending shows only pending entries."""
        out = _run(["list", "--status", "pending"], queue=queue_path)
        assert "GAAA..." in out
        assert "GBBB..." in out
        assert "GCCC..." not in out
        assert "GDDD..." not in out

    def test_list_limit(self, queue_path: Path):
        """list --limit 1 shows only one entry."""
        out = _run(["list", "--limit", "1"], queue=queue_path)
        assert "Showing 1 of 4" in out

    def test_list_empty(self, tmp_path: Path):
        """list on non-existent queue shows 'No entries'."""
        p = tmp_path / "nonexistent.json"
        out = _run(["list"], queue=p)
        assert "No entries" in out


class TestAnnotate:
    def test_annotate_existing_wallet(self, queue_path: Path):
        """annotate updates an existing pending wallet."""
        out = _run(
            ["annotate", "GAAA...", "1", "--comment", "wash found", "--annotator-id", "bob"],
            queue=queue_path,
        )
        assert "Annotated" in out
        assert "GAAA..." in out
        # Verify the queue was updated using _read_json directly
        queue = _read_json(queue_path)
        ann = next(a for a in queue if a["wallet"] == "GAAA...")
        assert ann["status"] == "annotated"
        assert ann["label"] == 1
        assert ann["annotator_id"] == "bob"
        assert ann["notes"] == "wash found"
        assert ann.get("annotation_hmac", "") != ""

    def test_annotate_new_wallet(self, queue_path: Path):
        """annotate adds a new wallet that was not in the queue."""
        out = _run(
            ["annotate", "GEEE...", "0", "--comment", "clean", "--annotator-id", "bob"],
            queue=queue_path,
        )
        assert "Annotated" in out
        queue = _read_json(queue_path)
        ann = next(a for a in queue if a["wallet"] == "GEEE...")
        assert ann["status"] == "annotated"
        assert ann["label"] == 0

    def test_annotate_invalid_label(self, queue_path: Path):
        """annotate with label other than 0/1 should error."""
        with pytest.raises(RuntimeError, match="exited with code"):
            _run(["annotate", "GAAA...", "2"], queue=queue_path)


class TestSkip:
    def test_skip_existing_wallet(self, queue_path: Path):
        """skip marks a pending wallet as skipped."""
        out = _run(
            ["skip", "GAAA...", "--reason", "insufficient data"],
            queue=queue_path,
        )
        assert "Skipped" in out
        queue = _read_json(queue_path)
        skipped = next(a for a in queue if a["wallet"] == "GAAA...")
        assert skipped["status"] == "skipped"
        assert skipped.get("skip_reason") == "insufficient data"

    def test_skip_nonexistent_wallet(self, queue_path: Path):
        """skip on a nonexistent wallet should error."""
        with pytest.raises(RuntimeError, match="exited with code"):
            _run(["skip", "GUNKNOWN..."], queue=queue_path)


class TestExport:
    def test_export_csv(self, queue_path: Path, tmp_path: Path):
        """export writes a valid CSV with wallet, label, annotator, timestamp, notes."""
        out_path = tmp_path / "annotations.csv"
        out = _run(
            ["export", "--output", str(out_path)],
            queue=queue_path,
        )
        assert "Exported" in out
        assert str(out_path) in out

        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Only one annotated entry in the fixture
        assert len(rows) == 1
        assert rows[0]["wallet"] == "GCCC..."
        assert rows[0]["label"] == "1"
        assert rows[0]["annotator_id"] == "alice"


class TestStats:
    def test_stats_output(self, queue_path: Path):
        """stats shows status breakdown, annotator counts, and date range."""
        out = _run(["stats"], queue=queue_path)
        assert "Status Breakdown" in out
        assert "pending" in out
        assert "annotated" in out
        assert "skipped" in out
        assert "By Annotator" in out
        assert "alice" in out
        assert "Total entries: 4" in out

    def test_stats_empty(self, tmp_path: Path):
        """stats on empty queue says 'Queue is empty'."""
        p = tmp_path / "empty.json"
        save_queue(p, [], SECRET)
        out = _run(["stats"], queue=p)
        assert "Queue is empty" in out
