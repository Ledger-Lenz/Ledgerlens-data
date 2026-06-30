# Model Governance

LedgerLens generates a **Model Card** after every training run to ensure model documentation is always current and auditable.

## Model Card Standard

Model Cards follow the [Google Model Card specification](https://modelcards.withgoogle.com/about) (Mitchell et al., 2019). Each card documents:

- Intended use and out-of-scope uses
- Training data characteristics (version + SHA-256 fingerprint)
- Performance metrics per asset pair
- Known limitations
- Hyperparameters
- SHAP feature importance chart reference

Cards are written to `models/MODEL_CARD_{model_name}_{version}.md` after every training run by `detection/model_training.py`.

## Metadata Schema

The input schema is defined in `reporting/schemas/model_metadata.json` (JSON Schema draft-07).

**Required fields:**

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Canonical model name (e.g. `xgboost`) |
| `training_date` | string | ISO 8601 training timestamp |
| `dataset_version` | string | Dataset version or path |
| `hyperparameters` | object | Key/value hyperparameter map |
| `performance_metrics` | object | Per-asset-pair precision/recall/F1 |
| `known_limitations` | string | Plain-text limitations |
| `intended_use` | string | Intended use cases |
| `out_of_scope_uses` | string | Explicitly excluded use cases |

**Optional fields (recommended for regulatory audits):**

| Field | Type | Description |
|---|---|---|
| `dataset_fingerprint` | string | SHA-256 of training Parquet file (computed at training time, not accepted as input) |
| `shap_importance_chart_path` | string | Relative path to SHAP chart image |
| `regulatory_contact` | string | Contact for regulatory enquiries |
| `data_retention_policy` | string | Data retention description |

## Generating a Model Card

Model cards are generated automatically during training. To regenerate manually:

```python
from reporting.model_card_generator import generate_model_card

generate_model_card(
    model_metadata_path="models/model_metadata.json",
    output_path="models/MODEL_CARD_xgboost_0.2.0.md",
    fmt="markdown",  # or "html"
)
```

A `MetadataValidationError` is raised with the missing field name if any required field is absent.

## Updating a Model Card After Re-evaluation

1. Update `models/model_metadata.json` with revised `performance_metrics` and any changed `known_limitations`.
2. Re-run `generate_model_card(...)` or trigger a new training run.
3. Commit the updated card alongside the new model weights.
4. Open a PR that links the re-evaluation results and the updated card.

## Data Provenance

The `dataset_fingerprint` field records the SHA-256 of the training Parquet file. This fingerprint is **computed at training time from the actual file** — it cannot be supplied by a caller — preventing spoofing of provenance claims.

To verify a card's provenance claim independently:

```bash
sha256sum data/synthetic_dataset.parquet
```

Compare the output against `dataset_fingerprint` in the card.
