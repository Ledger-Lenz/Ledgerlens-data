# Model Rollback Runbook

This runbook covers manual intervention when the automatic shadow deployment
and rollback mechanism in `scripts/retrain_if_drifted.py` fails or is
triggered repeatedly.

---

## Background

After concept drift is detected, `retrain_if_drifted.py` trains a new ensemble
and starts a **shadow deployment** rather than promoting immediately.  During
the shadow period (`SHADOW_PERIOD_HOURS`, default 24 h), a configurable
fraction of live scoring requests (`SHADOW_TRAFFIC_PERCENT`, default 20%) are
scored by both the production and candidate models.  Disagreements larger than
`SHADOW_DRIFT_THRESHOLD_POINTS` (default 15 points) are counted as **shadow
drift events**.

At the end of the shadow period the script is expected to be re-run with
`--check-shadow`.  It then automatically:

1. **Promotes** the candidate if drift rate < `SHADOW_DRIFT_MAX_RATE` (5%) and
   candidate FP rate does not exceed production by more than
   `SHADOW_FP_RATE_MAX_EXCESS` (10%).
2. **Rolls back** (retains production, discards candidate) otherwise, and logs
   a warning referencing this runbook.

---

## When Automatic Rollback Is Triggered Repeatedly

Repeated rollbacks indicate that the retrained model consistently disagrees
with production or produces more false positives.  Common causes:

| Symptom | Likely cause |
|---------|--------------|
| High shadow drift rate (> 5%) | Distribution shift in training data; corrupted batch |
| FP rate regression > 10% | Mislabelled training examples; class imbalance spike |
| Rollback on every cycle | Upstream data pipeline feeding corrupt/synthetic rows |

---

## Step-by-Step Manual Intervention

### 1. Check current shadow state

```bash
cat "${MODEL_DIR}/shadow_deployment_state.json"
```

Key fields: `version_id`, `candidate_dir`, `shadow_start`, `drift_rate`.

### 2. Inspect drift report

```bash
ls reports/retrain_report_*.json | sort | tail -3
cat reports/retrain_report_<latest>.json | python -m json.tool
```

Look at `drift_report.features_drifted` — which features triggered retraining?

### 3. Audit the training data

```bash
python -m scripts.build_labelled_dataset --output data/audit_$(date +%Y%m%d).parquet
```

Check label distribution and feature statistics against the archive:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/audit_<date>.parquet')
print(df['label'].value_counts())
print(df.describe())
"
```

### 4. Force rollback to archived production model

If the candidate dir still exists and you want to discard it immediately:

```bash
# Identify the most recent archive
ls -lt models/archive/ | head -5
ARCHIVE=models/archive/<timestamp>

# Restore production from archive
cp ${ARCHIVE}/*.joblib ${MODEL_DIR}/
cp ${ARCHIVE}/model_metadata.json ${MODEL_DIR}/
cp ${ARCHIVE}/metrics.json ${MODEL_DIR}/

# Clear shadow state
rm -f ${MODEL_DIR}/shadow_deployment_state.json

# Remove candidate dir
rm -rf ${MODEL_DIR}_new
```

Verify the restored artifact signatures:

```bash
python -c "
from detection.model_inference import verify_model_artifact_signature
ok = verify_model_artifact_signature('${MODEL_DIR}', 'manual-restore')
print('Signatures OK:', ok)
"
```

### 5. Promote a candidate manually (bypass shadow)

If you are confident the candidate is correct (e.g., you have reviewed the
training data and confirmed the drift is real):

```bash
python -m scripts.retrain_if_drifted \
    --retrain-data-path data/verified_dataset.parquet \
    --no-shadow
```

Exit code 2 means the model was promoted immediately.

### 6. Adjust thresholds to reduce rollback sensitivity

If legitimate model updates are being blocked by overly tight thresholds,
adjust via environment variables before re-running:

```bash
export SHADOW_DRIFT_MAX_RATE=0.10       # allow up to 10% drift (default 5%)
export SHADOW_FP_RATE_MAX_EXCESS=0.15  # allow up to 15% FP excess (default 10%)
export SHADOW_PERIOD_HOURS=12          # shorten shadow period (default 24)
python -m scripts.retrain_if_drifted --check-shadow
```

### 7. Disable shadow deployment temporarily

```bash
python -m scripts.retrain_if_drifted \
    --retrain-data-path data/dataset.parquet \
    --no-shadow
```

Re-enable by removing `--no-shadow` once the root cause is resolved.

---

## Alert: Repeated Rollback

If rollback is triggered more than 3 times in 7 days:

1. Open a P2 incident — the production model may be degrading while retraining
   is blocked.
2. Page the ML-Ops on-call rotation.
3. Consider temporarily pinning the model version by setting `MODEL_DIR` to the
   last known-good archive path.

---

## Key Files

| File | Purpose |
|------|---------|
| `${MODEL_DIR}/shadow_deployment_state.json` | Persisted shadow state (version ID, start time, drift rate) |
| `models/archive/<timestamp>/` | Point-in-time snapshots of every production model |
| `reports/retrain_report_<timestamp>.json` | Per-retraining audit report |
| `detection/model_inference.py` | `ShadowScorer`, `verify_model_artifact_signature` |
| `scripts/retrain_if_drifted.py` | Main retraining + shadow orchestration script |

---

## Related Documentation

- `docs/drift_detection.md` — Feature drift monitoring (PSI thresholds, MMD)
- `docs/model_governance.md` — Model approval workflow
- `docs/gnn_architecture.md` — Incremental graph update strategy
