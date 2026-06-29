# Scoring Architecture

LedgerLens normalises anomaly scores per asset pair to enable fair comparison across different trading patterns.

## Per-Pair Normalisation

Raw anomaly scores are normalised against a rolling historical percentile distribution for each asset pair:

- **Window size**: `SCORE_NORM_WINDOW_SIZE` (default 1000) scores per pair
- **Min samples**: `SCORE_NORM_MIN_SAMPLES` (default 50) samples required before normalisation
- **Algorithm**: Linear interpolation of percentile rank within the rolling window

## Redis Storage

Rolling windows are stored in Redis sorted sets:

- Key: `score_window:{asset_pair}`
- Members: Score values as strings
- Scores: The score value itself (for sorting)
- Eviction: `ZREMRANGEBYRANK` maintains exactly `SCORE_NORM_WINDOW_SIZE` entries

## Degradation

When fewer than `SCORE_NORM_MIN_SAMPLES` exist for a pair, normalisation is skipped and the raw score is returned with a `normalisation_skipped` flag.

## Interpreting Scores

- **Raw score**: Absolute anomaly score from the detection engine
- **Normalised score**: Percentile rank (0-1) relative to the pair's historical distribution
- **Normalised scores > 0.99**: Highly anomalous relative to the pair's baseline
- **Normalised scores ~ 0.5**: Typical behavior for the pair

## Security

Asset pair names used as Redis key suffixes are validated against an allowlist to prevent key injection.
