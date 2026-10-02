# Jitter-robust cross-chain hop matching (#882)

## Problem

`BehavioralMatcher.match_amount_fingerprints` links a Stellar leg to an external-chain
leg only when the two are within a fixed ±60 s window. An adversary evades this by
waiting a random amount of time between the legs. At 5 minutes of jitter, recall is
already down to 14%.

## Approach: `BehavioralMatcher.match_jitter_robust`

For each Stellar (source) leg `s`, every external leg `e` that arrives within
`[-max_clock_skew_seconds, max_delay_seconds]` (defaults: 120 s, 6 h) is a candidate,
with likelihood

```
L(s, e) = A(rel) · D(dt)
```

- **Amount-and-fee fingerprint `A`.** `rel = |e.amount − expected| / s.amount`, where
  `expected = s.amount·(1 − rate) − fixed` for the best-fitting `(fixed, rate)` in
  `fee_models`. `A` is Gaussian in `rel` (σ = `amount_tolerance`/2), truncated at
  `amount_tolerance`.
- **Delay prior `D`.** A mixture of an exponential (mean `typical_delay_seconds`, which
  models honest bridge latency) and a uniform over the whole window (the adversary may
  pick any delay), weighted by `adversarial_mix`. Negative delays (clock skew) decay
  quickly.
- **Distribution-aware null.** Candidates are normalised against each other and against
  "the true destination isn't here". The null weight scales with the number of chance
  amount collisions expected in the window. That count comes from the whole external
  set: both the exact band (spikes at round amounts) and a smoothed band
  `density_band`× wider (dense traffic). It is multiplied by the prior odds
  `(1 − match_prior)/match_prior`. As a result, a popular amount such as 1000.00 needs
  tight timing to be linked, while a distinctive amount such as 1043.3171 survives
  hours of jitter.
- **One-to-one assignment.** Pairs are taken greedily by posterior, each leg used at
  most once, and kept when the posterior ≥ `min_confidence` (0.6). The posterior is
  the emitted `confidence`, and the link type is `jitter_robust_fingerprint`.

## Measured results

`python -m scripts.cross_chain_evasion_simulator --seeds 10`. Each scenario has 150
bridged transfers, 300 background legs per chain, a 0.3% fee, 15% of amounts drawn
from popular round values, 3 days of traffic, and jitter ~ Uniform(0, J) on top of
~20 s honest latency. The naive matcher gets `tolerance = fee + 0.05%` so the
comparison isolates timing. The robust matcher gets `match_prior` = the scenario's
base rate (150/450).

| jitter J (s) | naive recall | naive precision | robust recall | robust precision |
|---:|---:|---:|---:|---:|
| 0 | 0.962 | 0.997 | 0.999 | 0.991 |
| 300 | 0.136 | 0.971 | 0.995 | 0.989 |
| 1800 | 0.021 | 0.835 | 0.871 | 0.981 |
| 3600 | 0.008 | 0.767 | 0.857 | 0.984 |
| 14400 | 0.001 | 0.500 | 0.845 | 0.982 |

Recall under jitter rises by **+0.84 to +0.86** (and by +0.86 already at 5 minutes of
jitter). Precision stays ≥ 0.98, and there is no regression without jitter. With no
round amounts (`round_amount_fraction=0`), robust recall at 1 h of jitter is 0.995. The
missing recall in the table is almost entirely round-amount transfers, which are
genuinely ambiguous.

`tests/test_jitter_robust_matching.py` locks in both regimes: the jittered margin, and
no loss on non-jittered cases.

## Assumptions

- The destination leg carries the source amount minus a known fee schedule
  (`fee_models`, one or more candidates).
- The maximum adversarial delay is bounded by `max_delay_seconds`. Beyond it, legs are
  never linked.
- Amount collisions in background traffic are roughly stationary over the observed
  span. The null uses whole-span frequency.

## Limitations

- **Dense decoy traffic.** With 10× more background legs than bridged ones
  (`n_background=1500`), the information limit shows. At the true base rate
  (`match_prior≈0.09`), jittered recall drops to 0.12 with precision 0.93. At
  `match_prior=0.5`, recall is 0.84 with precision 0.79. The naive matcher manages
  0.009 recall and 0.18 precision there. `match_prior` is the knob, so set it from
  ingestion coverage.
- **Amount splitting.** Splitting one transfer into several destination legs (or
  merging several) defeats the one-to-one fingerprint. That needs subset-sum matching
  and is out of scope.
- **Fee randomisation.** Unknown or variable fees widen the effective tolerance and
  reduce precision. Supply every fee schedule the bridge uses.
- Greedy assignment is not globally optimal. On contested candidates, a Hungarian
  assignment could recover a few extra pairs at higher cost.
