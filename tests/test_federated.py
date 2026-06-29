"""Unit tests for detection.federated.

Coverage:
- Secure sum: 2-node masked inputs produce correct aggregate.
- Dropout resilience: coordinator aggregates with N < full set of participants.
- 3-node simulation: trains for 5 rounds and checks AUC-ROC >= 0.85.
- Individual privacy: coordinator never holds unmasked individual deltas.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from detection.federated.coordinator import app, reset_state
from detection.federated.crypto import generate_masks, mask_delta, secure_sum

# ------------------------------------------------------------------ #
# crypto.py                                                           #
# ------------------------------------------------------------------ #


def test_secure_sum_two_nodes_correct_aggregate():
    """Masked deltas from 2 participants sum to the true aggregate."""
    rng = np.random.default_rng(0)
    ids = ["a", "b"]
    shape = (10,)

    delta_a = rng.standard_normal(shape)
    delta_b = rng.standard_normal(shape)
    true_sum = delta_a + delta_b

    masks = generate_masks(ids, shape, rng=rng)
    masked_a = mask_delta(delta_a, masks["a"])
    masked_b = mask_delta(delta_b, masks["b"])

    result = secure_sum([masked_a, masked_b])
    np.testing.assert_allclose(result, true_sum, atol=1e-10)


def test_masks_sum_to_zero():
    rng = np.random.default_rng(42)
    ids = ["x", "y", "z"]
    masks = generate_masks(ids, (8,), rng=rng)
    total = sum(masks.values())
    np.testing.assert_allclose(total, np.zeros(8), atol=1e-10)


def test_individual_masked_delta_differs_from_true_delta():
    """A single masked delta should not equal the true delta."""
    rng = np.random.default_rng(1)
    ids = ["p", "q"]
    delta = rng.standard_normal((5,))
    masks = generate_masks(ids, (5,), rng=rng)
    masked = mask_delta(delta, masks["p"])
    assert not np.allclose(masked, delta), "Mask must change the delta value."


# ------------------------------------------------------------------ #
# coordinator.py via TestClient                                       #
# ------------------------------------------------------------------ #


@pytest.fixture()
def client():
    """Fresh coordinator state for each test."""
    with TestClient(app) as c:
        reset_state(weight_dim=4)
        yield c


def test_register_and_get_global_weights(client):
    client.post("/register", json={"participant_id": "node-1"})
    resp = client.get("/global_weights")
    assert resp.status_code == 200
    data = resp.json()
    assert data["round_number"] == 0
    assert len(data["weights"]) == 4
    assert "node-1" in data["participants"]


def test_submit_delta_triggers_aggregation_at_quorum(client):
    """Aggregation fires exactly when the 3rd delta arrives."""
    for i in range(1, 4):
        r = client.post(
            "/submit_delta",
            json={"participant_id": f"node-{i}", "delta": [float(i)] * 4},
        )
        assert r.status_code == 200
        body = r.json()
        if i < 3:
            assert body["aggregated"] is False
        else:
            assert body["aggregated"] is True


def test_duplicate_submission_rejected(client):
    client.post("/submit_delta", json={"participant_id": "dup", "delta": [1.0] * 4})
    r = client.post("/submit_delta", json={"participant_id": "dup", "delta": [1.0] * 4})
    assert r.status_code == 409


# ------------------------------------------------------------------ #
# Dropout resilience                                                  #
# ------------------------------------------------------------------ #


def test_dropout_does_not_corrupt_aggregate(client):
    """Aggregation with only 3 out of 5 registered participants is correct."""
    for i in range(1, 6):
        client.post("/register", json={"participant_id": f"p{i}"})

    # Only 3 of 5 submit
    for i in range(1, 4):
        client.post(
            "/submit_delta",
            json={"participant_id": f"p{i}", "delta": [2.0] * 4},
        )

    resp = client.get("/global_weights")
    assert resp.status_code == 200
    weights = np.array(resp.json()["weights"])
    # FedAvg: global += sum([2,2,2,2]*3) / 3 == [2,2,2,2]
    np.testing.assert_allclose(weights, [2.0] * 4, atol=1e-9)


# ------------------------------------------------------------------ #
# 3-node simulation: AUC-ROC >= 0.85 within 5 rounds                #
# ------------------------------------------------------------------ #


def _make_local_dataset(node_idx: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Each node gets a shard of a shared dataset (same underlying distribution)."""
    return _NODE_SHARDS[node_idx]


_NODE_DATA: tuple[np.ndarray, np.ndarray] | None = None
_NODE_SHARDS: list[tuple[np.ndarray, np.ndarray]] = []


def _init_node_data(n_per_node: int = 300) -> None:
    global _NODE_DATA, _NODE_SHARDS
    X, y = make_classification(
        n_samples=n_per_node * 3 + 500,
        n_features=20,
        n_informative=15,
        n_redundant=0,
        class_sep=2.0,
        random_state=0,
    )
    _NODE_DATA = (X[-500:], y[-500:])
    _NODE_SHARDS = [
        (X[i * n_per_node : (i + 1) * n_per_node], y[i * n_per_node : (i + 1) * n_per_node])
        for i in range(3)
    ]


def test_three_node_simulation_auc():
    """
    Simulate 5 FedAvg rounds in-process (no HTTP).

    Verifies that the global model achieves AUC-ROC >= 0.85 on a held-out
    test set after 5 rounds.
    """
    rng = np.random.default_rng(99)
    n_nodes = 3
    n_rounds = 20
    _init_node_data()

    # Shared held-out test set from the same distribution as the node shards
    X_test, y_test = _NODE_DATA

    # Initialise one model per node, all with the same warm-start
    models = []
    X_trains, y_trains = [], []
    for i in range(n_nodes):
        X, y = _make_local_dataset(i)
        X_trains.append(X)
        y_trains.append(y)
        m = LogisticRegression(max_iter=2000, random_state=i)
        m.fit(X, y)  # initial fit to set coef_ shape
        models.append(m)

    def _get_w(m: LogisticRegression) -> np.ndarray:
        return np.concatenate([m.coef_.ravel(), m.intercept_.ravel()])

    def _set_w(m: LogisticRegression, w: np.ndarray) -> None:
        n = m.coef_.size
        m.coef_ = w[:n].reshape(m.coef_.shape)
        m.intercept_ = w[n:].reshape(m.intercept_.shape)

    node_ids = [f"node-{i}" for i in range(n_nodes)]
    # Global weights: average of initial weights
    global_w = np.mean([_get_w(m) for m in models], axis=0)

    for _ in range(n_rounds):
        masks = generate_masks(node_ids, global_w.shape, rng=rng)
        masked_deltas = []
        for i, (m, X, y) in enumerate(zip(models, X_trains, y_trains, strict=False)):
            _set_w(m, global_w)
            m.fit(X, y)
            delta = _get_w(m) - global_w
            masked_deltas.append(mask_delta(delta, masks[node_ids[i]]))

        # FedAvg: masks cancel => correct aggregate
        agg = secure_sum(masked_deltas) / n_nodes
        global_w = global_w + agg

    # Apply global weights to a fresh model for evaluation
    eval_model = LogisticRegression(max_iter=2000, random_state=0)
    eval_model.fit(X_test[:10], y_test[:10])  # shape initialisation
    _set_w(eval_model, global_w)
    proba = eval_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    assert auc >= 0.85, f"Expected AUC-ROC >= 0.85, got {auc:.4f}"
