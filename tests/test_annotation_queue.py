"""Tests for detection/active_learning/annotation_queue.py — AnnotationQueue class
and legacy add_annotation / export_labelled functions."""

from __future__ import annotations

import json
import os
import stat

import pytest

from detection.active_learning.annotation_queue import (
    AnnotationQueue,
    _compute_hmac,
    add_annotation,
    export_labelled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _queue_path(tmp_path) -> str:
    return str(tmp_path / "annotation_queue.json")


def _make_queue(tmp_path) -> AnnotationQueue:
    return AnnotationQueue(queue_path=_queue_path(tmp_path))


# ---------------------------------------------------------------------------
# AnnotationQueue.push / pop_batch
# ---------------------------------------------------------------------------


def test_push_adds_pending_wallets(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1", "W2"], strategy_name="entropy")
    batch = q.pop_batch(10)
    assert [i["wallet"] for i in batch] == ["W1", "W2"]
    assert all(i["status"] == "pending" for i in batch)


def test_push_skips_duplicate_wallets(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1", "W2"], strategy_name="entropy")
    q.push(["W2", "W3"], strategy_name="entropy")
    batch = q.pop_batch(10)
    wallets = [i["wallet"] for i in batch]
    assert wallets.count("W2") == 1
    assert "W3" in wallets


def test_pop_batch_returns_only_pending(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1", "W2", "W3"], strategy_name="entropy")
    q.skip("W2")
    batch = q.pop_batch(10)
    assert all(i["wallet"] != "W2" for i in batch)


def test_pop_batch_respects_n(tmp_path):
    q = _make_queue(tmp_path)
    q.push([f"W{i}" for i in range(10)], strategy_name="entropy")
    assert len(q.pop_batch(3)) == 3


# ---------------------------------------------------------------------------
# AnnotationQueue.annotate
# ---------------------------------------------------------------------------


def test_annotate_records_verdict(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    q.annotate("W1", label=1, annotator_id="alice")
    with open(_queue_path(tmp_path)) as f:
        data = json.load(f)
    w1 = next(d for d in data if d["wallet"] == "W1")
    assert w1["label"] == 1
    assert w1["status"] == "annotated"
    assert w1["annotator_id"] == "alice"


def test_annotate_is_idempotent(tmp_path):
    """Second call with same wallet updates, doesn't append a duplicate."""
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    q.annotate("W1", label=1, annotator_id="alice")
    q.annotate("W1", label=0, annotator_id="alice")  # update label
    with open(_queue_path(tmp_path)) as f:
        data = json.load(f)
    w1_entries = [d for d in data if d["wallet"] == "W1"]
    assert len(w1_entries) == 1
    assert w1_entries[0]["label"] == 0


def test_annotate_rejects_empty_annotator_id(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    with pytest.raises(ValueError, match="annotator_id"):
        q.annotate("W1", label=1, annotator_id="")


def test_annotate_rejects_invalid_label(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    with pytest.raises(ValueError, match="label"):
        q.annotate("W1", label=5, annotator_id="alice")


def test_annotate_adds_wallet_not_in_queue(tmp_path):
    """Annotating a wallet not yet pushed should add it inline."""
    q = _make_queue(tmp_path)
    q.annotate("W_NEW", label=0, annotator_id="bob")
    with open(_queue_path(tmp_path)) as f:
        data = json.load(f)
    assert any(d["wallet"] == "W_NEW" and d["label"] == 0 for d in data)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_original_intact_on_rename_failure(tmp_path, monkeypatch):
    """Simulate os.rename raising; original file must remain unchanged."""
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")

    # Capture the original content
    with open(_queue_path(tmp_path)) as f:
        original_content = f.read()

    import detection.active_learning.annotation_queue as aq_mod

    def _raise(*args, **kwargs):
        raise OSError("Simulated rename failure")

    monkeypatch.setattr(aq_mod.os, "rename", _raise)

    # The annotation write should raise (because rename fails)
    with pytest.raises(OSError):
        q.annotate("W1", label=1, annotator_id="alice")

    # Original file must be intact
    with open(_queue_path(tmp_path)) as f:
        content_after = f.read()
    assert content_after == original_content


# ---------------------------------------------------------------------------
# File permissions
# ---------------------------------------------------------------------------


def test_queue_file_written_with_mode_0o600(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    path = _queue_path(tmp_path)
    mode = oct(stat.S_IMODE(os.stat(path).st_mode))
    assert mode == oct(0o600), f"Expected 0o600, got {mode}"


# ---------------------------------------------------------------------------
# AnnotationQueue.export_labelled
# ---------------------------------------------------------------------------


def test_export_labelled_only_annotated(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1", "W2", "W3"], strategy_name="entropy")
    q.annotate("W1", label=1, annotator_id="alice")
    q.annotate("W2", label=0, annotator_id="bob")
    q.skip("W3")

    out = str(tmp_path / "out.parquet")
    df = q.export_labelled(out)
    assert set(df["wallet"].tolist()) == {"W1", "W2"}
    assert os.path.exists(out)


def test_export_labelled_excludes_tampered_hmac(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    q.annotate("W1", label=1, annotator_id="alice")

    with open(_queue_path(tmp_path)) as f:
        data = json.load(f)
    data[0]["annotation_hmac"] = "badhash"
    with open(_queue_path(tmp_path), "w") as f:
        json.dump(data, f)

    out = str(tmp_path / "out.parquet")
    df = q.export_labelled(out)
    assert df.empty


def test_export_labelled_empty_when_none_annotated(tmp_path):
    q = _make_queue(tmp_path)
    q.push(["W1"], strategy_name="entropy")
    out = str(tmp_path / "out.parquet")
    df = q.export_labelled(out)
    assert df.empty


# ---------------------------------------------------------------------------
# Legacy add_annotation / export_labelled
# ---------------------------------------------------------------------------


def test_add_annotation_creates_queue_with_hmac(tmp_path):
    path = _queue_path(tmp_path)
    ann = add_annotation(path, "GABC", 1, "alice", "2026-06-20T00:00:00Z")
    assert os.path.exists(path)
    assert ann["annotation_hmac"] == _compute_hmac("GABC", 1, "alice", "2026-06-20T00:00:00Z")


def test_legacy_export_labelled_returns_valid(tmp_path):
    path = _queue_path(tmp_path)
    add_annotation(path, "GABC", 1, "alice", "2026-06-20T00:00:00Z")
    add_annotation(path, "GXYZ", 0, "bob", "2026-06-20T01:00:00Z")
    result = export_labelled(path)
    assert len(result) == 2


def test_legacy_export_labelled_rejects_tampered_label(tmp_path):
    path = _queue_path(tmp_path)
    add_annotation(path, "GABC", 1, "alice", "2026-06-20T00:00:00Z")
    with open(path) as f:
        queue = json.load(f)
    queue[0]["label"] = 0
    with open(path, "w") as f:
        json.dump(queue, f)
    result = export_labelled(path)
    assert len(result) == 0


def test_legacy_export_labelled_logs_warning_for_invalid_hmac(tmp_path, caplog):
    import logging

    path = _queue_path(tmp_path)
    add_annotation(path, "GABC", 1, "alice", "2026-06-20T00:00:00Z")
    with open(path) as f:
        queue = json.load(f)
    queue[0]["annotation_hmac"] = "badhash"
    with open(path, "w") as f:
        json.dump(queue, f)
    with caplog.at_level(logging.WARNING):
        export_labelled(path)
    assert any("Invalid HMAC" in r.message for r in caplog.records)


def test_legacy_export_labelled_excludes_invalid_includes_valid(tmp_path):
    path = _queue_path(tmp_path)
    add_annotation(path, "GABC", 1, "alice", "2026-06-20T00:00:00Z")
    add_annotation(path, "GXYZ", 0, "bob", "2026-06-20T01:00:00Z")
    with open(path) as f:
        queue = json.load(f)
    queue[0]["wallet"] = "GTAMPERED"
    with open(path, "w") as f:
        json.dump(queue, f)
    result = export_labelled(path)
    assert len(result) == 1
    assert result[0]["wallet"] == "GXYZ"


def test_legacy_export_labelled_empty_when_no_file(tmp_path):
    result = export_labelled(str(tmp_path / "nonexistent.json"))
    assert result == []
