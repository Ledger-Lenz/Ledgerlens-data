# Monitoring Runbook

Operational guidance for the monitoring stack: drift detection, alert triage, and
incident response.

## Drift Detection

Two drift signals are tracked and surfaced together in a single triage view:

- **Feature drift** (`monitoring/drift_detector.py`, infra-level): an input
  feature's distribution has shifted relative to its reference window.
- **Model-output drift** (`detection/drift_monitor.py`, model-level): the model's
  prediction distribution has shifted relative to its reference window.

When output drift is flagged, the triage view runs correlation logic that checks
whether any tracked input feature also drifted **in the same window**. The result
is annotated on the unified dashboard/report so an operator can immediately tell
whether the output drift is explained by upstream feature drift or is
unexplained.

### Correlation annotation

| Output drift | Any input feature drift (same window) | Annotation |
| --- | --- | --- |
| flagged | yes | `explained` — output drift correlates with upstream feature drift |
| flagged | no | `unexplained` — output drift with no upstream feature drift |
| not flagged | yes | `input_only` — feature drift without output drift |
| not flagged | no | `none` — no drift |

## Triage Decision Tree

Use the combination of signals from the unified view to decide the response.

```
Output drift flagged?
├─ Yes
│  └─ Any tracked input feature drifted in the same window?
│     ├─ Yes  → BOTH DRIFTED (explained)
│     │         Root cause is likely upstream data. Investigate the drifted
│     │         feature(s): upstream pipeline, schema, or source distribution.
│     │         Fixing the feature drift should resolve the output drift.
│     └─ No   → ONLY OUTPUT DRIFTED (unexplained)
│               No upstream feature drift explains the shift. Investigate the
│               model itself: recent deploys, weights, serving config, or
│               post-processing. Escalate to the model owner.
└─ No
   └─ Any tracked input feature drifted in the same window?
      ├─ Yes  → ONLY INPUT DRIFTED
      │         Output is currently stable despite feature drift. Monitor
      │         closely; the model may absorb the shift or degrade later.
      │         Review the drifted feature and its downstream impact.
      └─ No   → NO DRIFT
                No action required.
```

### Response summary

- **Both drifted** — treat as upstream data issue; fix the feature drift first.
- **Only output drifted** — treat as a model issue; escalate to the model owner.
- **Only input drifted** — no immediate output impact; monitor and review the
  drifted feature.

## Verification

Correlation logic is verified against synthetic scenarios covering all three
drift-combination cases (both drifted, only output drifted, only input drifted)
before changes to the triage view are merged.
