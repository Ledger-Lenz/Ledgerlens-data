# Cold-Start Scoring with Neural Process Meta-Learning

## Overview

When a new asset pair is first listed on the Stellar DEX, the system has too few
trades to compute reliable Benford statistics or ML features.  The standard
ensemble would fall back to the global prior (average statistics across all
pairs), producing poorly calibrated scores for that pair.

The Neural Process (NP) meta-learning layer addresses this by learning **how to
adapt** from a small context set rather than relying on a fixed global fallback.

## Architecture

The implementation in `detection/neural_process.py` uses a **Conditional Neural
Process (CNP)**:

- **Encoder** — a two-layer MLP that maps each `(features, label)` context
  trade to a fixed-dimensional latent vector, then aggregates variable-size
  context sets via **mean pooling**.  This makes the encoder permutation-
  invariant and compatible with any context size from 1 to 50 trades.
- **Decoder** — a two-layer MLP that concatenates the pooled context embedding
  with a query feature vector and outputs a wash-trade probability.

The CNP design was chosen over a Latent NP (which adds a stochastic latent
variable) because calibration accuracy — not uncertainty quantification — is the
primary goal in the cold-start path.

## Cold-Start Threshold and Blending

```
NP_COLD_START_THRESHOLD = 50  # trades
```

When a pair has `trade_count < 50` labelled trades, the scorer blends the NP
score with the ensemble score **linearly**:

```
blend_weight = 1.0 - trade_count / threshold
blended_score = blend_weight * np_score + (1 - blend_weight) * ensemble_score
```

- At `trade_count = 0` → pure NP score (blend_weight = 1.0)
- At `trade_count = 25` → 50 / 50 mix
- At `trade_count ≥ 50` → pure ensemble score (blend_weight = 0.0)

This transition avoids a hard cutover and produces smooth score evolution as
trade history accumulates.

## Usage

```python
from detection.model_inference import RiskScorer
import numpy as np

scorer = RiskScorer()

# context_features: (n_context, feature_dim) array of seed trades
# context_labels:   binary wash-trade labels for context trades
result = scorer.score_cold_start(
    feature_row=feature_row,
    context_features=np.array([...]),
    context_labels=[0, 1, 0, 1, 0],
    trade_count=5,
)
# result["np_cold_start"] == True
# result["np_blend_weight"] == 0.9
```

## Security

Context trades provided at inference time are validated against the feature
schema before encoding.  Any trade with an unrecognised schema hash raises a
`RuntimeError` before the NP encoder is invoked, matching the behaviour of the
main ensemble scorer.

## Testing

```bash
pytest tests/test_neural_process.py -v
```

Key tests:
- **Consistency**: identical context and query sets must produce identical predictions.
- **Cold-start regression**: NP scores on known wash-trade pairs must exceed 50
  with as few as 5 context trades.
