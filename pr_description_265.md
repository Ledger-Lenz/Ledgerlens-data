# PR: Annotation Disagreement Detector — Multi-Annotator Inter-Rater Agreement Scoring

## Summary

This PR implements the annotation disagreement detector described in Issue #265.
When multiple analysts independently label the same wallet, the system now
measures their agreement, flags systematic disagreements, and routes disputed
wallets to a senior reviewer — eliminating silent label noise that degrades
model quality.

---

## What Changed

### `detection/active_learning/annotation_queue.py`

**Schema extension — multi-label support**

Each wallet entry in `data/annotation_queue.json` now has an `"annotations"`
list that holds one record per annotator:

```json
{
  "wallet": "GABCD...",
  "status": "disputed",
  "agreement_kappa": -1.0,
  "annotations": [
    {
      "label": 1,
      "annotator_id": "anon-7f3a",
      "annotated_at": "2026-06-27T18:00:00+00:00",
      "annotation_hmac": "<hmac>"
    },
    {
      "label": 0,
      "annotator_id": "anon-2b9c",
      "annotated_at": "2026-06-27T18:05:00+00:00",
      "annotation_hmac": "<hmac>"
    }
  ]
}
```

Every annotation is HMAC-SHA256 protected (same key as the existing single-annotator
HMAC). Annotator IDs are **pseudonymous opaque strings** only — email addresses
are explicitly rejected to protect annotator privacy.

**New public API**

| Method | Purpose |
|---|---|
| `multi_annotate(wallet, label, annotator_id)` | Add an annotation from a second (or third) annotator. Duplicate annotator IDs are rejected with a warning. |
| `compute_inter_annotator_agreement(wallet_id)` | Return `{kappa, alpha, n_annotators, disputed}` for a wallet with ≥ 2 verified annotations. |
| `get_senior_review_queue()` | Return wallet IDs where Cohen's Kappa < `DISPUTE_KAPPA_THRESHOLD` (0.6). |

**Agreement metrics**

- **Cohen's Kappa** — implemented directly (closed-form formula) to handle the
  single-item-per-annotator case correctly. `sklearn.cohen_kappa_score` requires
  at least one sample of each class across both raters, which breaks for a single
  binary rating; the direct formula `κ = (P_o − P_e) / (1 − P_e)` handles all
  cases correctly, including the κ = −1.0 boundary (total disagreement).
- **Krippendorff's Alpha** — computed via the `krippendorff` library. Handles
  missing annotations naturally (not every wallet is double-annotated). A
  `try/except` guard means the system degrades gracefully if the library is
  unavailable at runtime.

**Dispute detection**

When `multi_annotate` adds a label that brings the wallet to ≥ 2 annotations, it
immediately recomputes kappa and updates `item["status"]` to `"disputed"` (κ < 0.6)
or `"multi_annotated"` (κ ≥ 0.6). This keeps the queue file consistent without
requiring a separate scan pass.

**Backward compatibility**

The `annotate()` method, `add_annotation()` and `export_labelled()` legacy
functions are untouched. All five existing `test_annotation_queue.py` tests
continue to pass.

---

### `requirements.txt`

```
krippendorff>=0.6.1
```

Added under a clearly labelled comment section.

---

### `tests/test_inter_annotator_agreement.py` (new file)

Six unit tests covering all three issue requirements plus edge cases:

| Test | What it verifies |
|---|---|
| `test_kappa_perfect_agreement` | Two annotators agreeing → κ = 1.0, `disputed=False` |
| `test_kappa_total_disagreement` | Two annotators on opposite binary labels → κ = −1.0, `disputed=True` |
| `test_disputed_wallet_in_senior_review_queue` | Disagreed wallet appears in `get_senior_review_queue()`; agreed wallet does not |
| `test_agreement_requires_min_two_annotations` | `ValueError` raised with < 2 annotations |
| `test_duplicate_annotator_rejected` | Same annotator labelling twice → only 1 stored |
| `test_dispute_threshold_value` | `DISPUTE_KAPPA_THRESHOLD == 0.6` |

All six tests pass (`pytest tests/test_inter_annotator_agreement.py -v`).

---

### `monitoring/grafana/dashboards/ledgerlens-kafka.json`

Added panel **id=5** — "Inter-annotator Kappa (rolling)" — spanning the full
24-column row below the existing four panels (`y=16`):

- Plots `inter_annotator_kappa` (mean Cohen's Kappa over time) and
  `inter_annotator_disputed_total` (cumulative disputed wallet count).
- Threshold lines at κ = 0.4 (red), 0.6 (yellow), 0.8 (green) for instant
  visual feedback on annotation quality.
- Y-axis range: [−1, 1].

---

### `docs/active_learning.md`

New **"Multi-Annotator Workflow"** section covering:

- When to use Cohen's Kappa vs Krippendorff's Alpha and why.
- Step-by-step dispute resolution process (second annotator → senior review → tie-break).
- Code example for `multi_annotate` + `compute_inter_annotator_agreement`.
- Privacy requirement: pseudonymous annotator IDs only.
- Grafana dashboard pointer.
- Configuration table (`DISPUTE_KAPPA_THRESHOLD`).

---

## Why Cohen's Kappa vs Krippendorff's Alpha?

| | Cohen's Kappa | Krippendorff's Alpha |
|---|---|---|
| **Label type** | Binary (0/1) | Nominal, ordinal, interval, ratio |
| **Missing data** | Not handled | Handled natively |
| **Fixed annotators** | Exactly 2 | Any number |
| **Use in LedgerLens** | Primary metric for binary wash/clean labels | Extension when confidence levels or multi-class labels are used |

For the current binary wash-trading label schema, κ is the primary gate.
Alpha is computed alongside it for forward-compatibility with future ordinal
confidence labels (e.g. 0 = clean, 1 = suspicious, 2 = wash).

---

## Testing

```bash
# New tests
pytest tests/test_inter_annotator_agreement.py -v   # 6 passed

# Regression — existing queue tests
pytest tests/test_annotation_queue.py -v            # 5 passed
```

---

closes #265
