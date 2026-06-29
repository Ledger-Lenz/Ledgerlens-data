"""Unit tests for multi-annotator inter-rater agreement (Issue #265).

Covers:
- Perfect agreement → Cohen's Kappa = 1.0
- Total disagreement (binary, 2 annotators) → Kappa = -1.0
- Disputed wallet (Kappa < 0.6) appears in senior review queue
"""

import os
import pytest

from detection.active_learning.annotation_queue import AnnotationQueue, DISPUTE_KAPPA_THRESHOLD


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """AnnotationQueue backed by a temp file with a known HMAC secret."""
    monkeypatch.setenv("ANNOTATION_HMAC_SECRET", "test-secret-key-265")
    # Reload config so the env var is picked up
    import importlib
    import config as cfg_module
    importlib.reload(cfg_module)
    from config import config as cfg
    monkeypatch.setattr(cfg, "ANNOTATION_HMAC_SECRET", "test-secret-key-265")

    return AnnotationQueue(queue_path=str(tmp_path / "queue.json"))


def _push_wallet(queue: AnnotationQueue, wallet: str) -> None:
    queue.push([wallet], strategy_name="committee_disagreement")


# ---------------------------------------------------------------------------
# Test: perfect agreement → Kappa = 1.0
# ---------------------------------------------------------------------------

def test_kappa_perfect_agreement(queue):
    """Two annotators who always agree produce Kappa = 1.0."""
    wallet = "GAGREEMENT111111111111111111111111111111111111111111111111"
    _push_wallet(queue, wallet)
    queue.multi_annotate(wallet, label=1, annotator_id="anon-alice")
    queue.multi_annotate(wallet, label=1, annotator_id="anon-bob")

    result = queue.compute_inter_annotator_agreement(wallet)

    assert result["n_annotators"] == 2
    assert result["kappa"] == pytest.approx(1.0)
    assert result["disputed"] is False


# ---------------------------------------------------------------------------
# Test: total disagreement → Kappa = -1.0
# ---------------------------------------------------------------------------

def test_kappa_total_disagreement(queue):
    """Two annotators who always disagree on a binary label produce Kappa = -1.0."""
    wallet = "GDISAGREE1111111111111111111111111111111111111111111111111"
    _push_wallet(queue, wallet)
    queue.multi_annotate(wallet, label=1, annotator_id="anon-alice")
    queue.multi_annotate(wallet, label=0, annotator_id="anon-bob")

    result = queue.compute_inter_annotator_agreement(wallet)

    assert result["n_annotators"] == 2
    assert result["kappa"] == pytest.approx(-1.0)
    assert result["disputed"] is True


# ---------------------------------------------------------------------------
# Test: disputed wallet appears in senior review queue
# ---------------------------------------------------------------------------

def test_disputed_wallet_in_senior_review_queue(queue):
    """A wallet with Kappa < 0.6 must appear in get_senior_review_queue()."""
    # Disagreed wallet — kappa = -1.0 → disputed
    disputed_wallet = "GDISPUTED111111111111111111111111111111111111111111111111"
    _push_wallet(queue, disputed_wallet)
    queue.multi_annotate(disputed_wallet, label=1, annotator_id="anon-alice")
    queue.multi_annotate(disputed_wallet, label=0, annotator_id="anon-bob")

    # Agreed wallet — kappa = 1.0 → not disputed
    clean_wallet = "GAGREED1111111111111111111111111111111111111111111111111"
    _push_wallet(queue, clean_wallet)
    queue.multi_annotate(clean_wallet, label=0, annotator_id="anon-alice")
    queue.multi_annotate(clean_wallet, label=0, annotator_id="anon-bob")

    senior_queue = queue.get_senior_review_queue()

    assert disputed_wallet in senior_queue
    assert clean_wallet not in senior_queue


# ---------------------------------------------------------------------------
# Test: fewer than 2 annotations raises ValueError
# ---------------------------------------------------------------------------

def test_agreement_requires_min_two_annotations(queue):
    wallet = "GSINGLE11111111111111111111111111111111111111111111111111"
    _push_wallet(queue, wallet)
    queue.multi_annotate(wallet, label=1, annotator_id="anon-solo")

    with pytest.raises(ValueError, match="need at least 2"):
        queue.compute_inter_annotator_agreement(wallet)


# ---------------------------------------------------------------------------
# Test: duplicate annotator is rejected
# ---------------------------------------------------------------------------

def test_duplicate_annotator_rejected(queue):
    wallet = "GDUPLICATE111111111111111111111111111111111111111111111111"
    _push_wallet(queue, wallet)
    queue.multi_annotate(wallet, label=1, annotator_id="anon-alice")
    queue.multi_annotate(wallet, label=0, annotator_id="anon-alice")  # duplicate

    result = queue.compute_inter_annotator_agreement.__func__  # access but don't call
    # Only 1 annotation should be stored
    labels = queue._verified_labels(wallet)
    assert len(labels) == 1


# ---------------------------------------------------------------------------
# Test: DISPUTE_KAPPA_THRESHOLD constant is 0.6
# ---------------------------------------------------------------------------

def test_dispute_threshold_value():
    assert DISPUTE_KAPPA_THRESHOLD == 0.6
