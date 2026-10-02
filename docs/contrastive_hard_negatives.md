# Contrastive pre-training: domain hard-negative mining (#890)

Generic random/ANN negatives rarely contain the wallets that matter most:
**legitimate high-frequency traders and market makers**, whose round-trip,
high-cadence flow looks near-identical to wash trading. `HardNegativeMiner`
now reserves part of the hard-negative budget for these wallets.

## Strategy

1. `build_clean_index(clean_embeddings)` — ANN over all confirmed-clean wallets.
2. `set_hft_negatives(hft_positions, clean_embeddings)` — second ANN over the
   known-legitimate HFT subset (row indices into `clean_embeddings`).
3. `mine_negatives(anchors, k, epoch)` splits the curriculum's hard budget:
   `round(hft_negative_fraction × n_hard)` nearest legitimate-HFT wallets per
   wash-trade anchor, the rest nearest generic clean wallets, then random easy
   negatives.

## Configuration

| Setting | Default | Notes |
|---|---|---|
| `CONTRASTIVE_HFT_NEGATIVE_FRACTION` env / `hft_negative_fraction=` | `0.5` | `0` disables (previous behaviour) |
| `CONTRASTIVE_CURRICULUM_EPOCHS` | `5` | Unchanged curriculum ramp |

## Cost

One extra ANN build over the HFT subset per index refresh and one extra k-NN
query per batch. Measure with `python -m benchmarks.hft_negative_benchmark`;
the HFT subset is small, so overhead is expected to be < 5 % of pre-training
wall-clock — the accepted budget for this feature.

## Downstream evaluation

Pretrain with `hft_negative_fraction=0.0` and with the default, fine-tune both
via `detection/contrastive/finetune.py`, and compare on the HFT-vs-wash slice
with `evaluation/backtest.py` (precision on legitimate-HFT wallets is the key
metric — false positives there are what domain negatives target).
