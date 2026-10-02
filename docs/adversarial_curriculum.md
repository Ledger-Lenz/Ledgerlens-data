# Adversarial Training Curriculum, Early Stopping & Experiment Tracking

Issue #872. This document covers the curriculum-scheduling extension to the
FGSM adversarial training loop introduced for Issue #191
(`detection.adversarial.robustness.run_adversarial_training`, invoked from
`detection.model_training.main` when `ADV_TRAINING_ENABLED=true`).

## Why a curriculum

Adversarial training at a single, fixed perturbation budget every epoch can
cause the clean/robust-accuracy tradeoff to collapse (the model overfits to
resisting one specific perturbation magnitude) or destabilize training
outright, especially early on when the model hasn't yet learned a stable
clean decision boundary. A **curriculum** — starting with weak perturbations
and ramping up to the target strength — lets the model first stabilize on
the clean task before the perturbation budget grows large enough to matter.

Note on file layout: this issue's description names
`scripts/adversarial_training_loop.py` as one of the two files that "apply
adversarial examples during training." That script is a separate GAN-style
loop that regenerates whole synthetic datasets each round from an attacker
*profile* (`NaiveAttacker` → `AdaptiveAttacker`), not a per-example
perturbation-budget loop — there's no `epsilon` there to curriculum-schedule
against. The actual per-epoch, per-example FGSM training loop with epochs
and a fixed `epsilon` is `run_adversarial_training` in
`detection/adversarial/robustness.py`; that's where this curriculum lives.

## `CurriculumScheduler`

```python
from detection.adversarial.augmentation import CurriculumScheduler

scheduler = CurriculumScheduler(start=0.025, end=0.1, epochs=5, strategy="linear")
scheduler.schedule()  # [0.025, 0.04375, 0.0625, 0.08125, 0.1]
```

Two strategies:

| Strategy | Behavior |
|---|---|
| `"linear"` | Continuous linear interpolation from `start` to `end` across all epochs. |
| `"step"` | A 4-stage staircase — coarser jumps, useful when a smooth per-epoch change isn't meaningful (e.g. very few epochs, or a very cheap per-epoch training cost where you'd rather hold each stage for several epochs). |

With `epochs <= 1` there's no room to ramp, so `epsilon_for_epoch` always
returns `end`.

## Using it via `run_adversarial_training`

```python
from detection.adversarial.robustness import run_adversarial_training

report = run_adversarial_training(
    df,
    epochs=8,
    epsilon=0.1,                    # target (final-epoch) budget
    curriculum="linear",
    curriculum_start_epsilon=0.025, # defaults to epsilon / 4 if omitted
    early_stopping_patience=2,
    experiment_tracker=tracker,     # optional, see below
)
```

- `report["epsilon_schedule"]` — the actual per-epoch training epsilon used.
- `report["epoch_log"][i]["train_epsilon"]` — same value, alongside that
  epoch's `clean_auc`/`adversarial_auc`.
- **The adversarial *validation* set is always perturbed at the final
  target `epsilon`**, not the current epoch's training epsilon — so
  `adversarial_auc` numbers are directly comparable across every epoch of a
  curriculum run, and against a fixed-epsilon (non-curriculum) baseline run
  at the same target `epsilon`.

All curriculum/early-stopping/tracking parameters are **opt-in** — omitting
them reproduces the exact pre-#872 behavior (fixed `epsilon` every epoch,
every requested epoch runs, no experiment log file is written).

## Early stopping on *robust* validation accuracy

```python
run_adversarial_training(df, epochs=10, epsilon=0.1, early_stopping_patience=2)
```

Training stops once `adversarial_auc` (the FGSM-perturbed validation set's
AUC-ROC) fails to improve by more than `early_stopping_min_delta` (default
`0.001`) for `early_stopping_patience` consecutive epochs. **Clean accuracy
never gates stopping** — this is the issue's explicit requirement, since a
model can keep improving on the perturbed validation set while clean
accuracy plateaus or wobbles slightly, and that's exactly the tradeoff this
loop is meant to manage.

`report["epochs_run"]` is the number of epochs actually executed;
`report["epochs"]` stays the originally requested value so callers can tell
early-stopped runs apart from full runs. `report["early_stopped"]` is a
plain boolean.

## Experiment tracking

```python
from mlops.experiment_tracking import JsonlExperimentTracker

tracker = JsonlExperimentTracker(path="models/experiments.jsonl")
report = run_adversarial_training(df, epochs=5, epsilon=0.1, experiment_tracker=tracker)
```

One `ExperimentRun` (name `"adversarial_training_epoch"`) is logged per
epoch actually run, with `params` (`epoch`, `train_epsilon`,
`target_epsilon`, `adv_ratio`, `curriculum`) and `metrics` (`clean_auc`,
`adversarial_auc`). The append-only JSONL file is the "experiment report"
referenced in the issue's acceptance criteria — inspect it directly, or via
`tracker.list_runs()`.

Passing no `experiment_tracker` (the default) skips logging entirely — no
file is created or written to.

## Pipeline configuration

Set these in `.env` (see `.env.example` for the full list with defaults):

| Variable | Default | Purpose |
|---|---|---|
| `ADV_TRAINING_CURRICULUM` | `""` (disabled) | `"linear"` or `"step"` |
| `ADV_TRAINING_CURRICULUM_START_EPSILON` | `""` (→ `ADV_TRAINING_EPSILON / 4`) | Starting epsilon for the ramp |
| `ADV_TRAINING_EARLY_STOP_PATIENCE` | `0` (disabled) | Epochs of stalled robust accuracy before stopping |
| `ADV_TRAINING_EXPERIMENT_LOG_PATH` | `""` (disabled) | JSONL path for per-epoch experiment logging |

## Recommended defaults

Based on the no-divergence check across 3 seeds
(`tests/test_adversarial_curriculum.py::TestNoDivergenceAcrossSeeds`) and the
existing Issue #191 tolerances (clean-accuracy degradation ≤ 3 points):

- **`ADV_TRAINING_CURRICULUM=linear`** with **`ADV_TRAINING_EPOCHS >= 5`**.
  A 2-3 epoch run barely has room to ramp — the curriculum's value comes
  from several intermediate stages, not just a start/end point.
- Leave `ADV_TRAINING_CURRICULUM_START_EPSILON` unset (defaults to a
  quarter of the target) unless you have a specific reason to start weaker
  or stronger.
- **`ADV_TRAINING_EARLY_STOP_PATIENCE=2`** once you're running enough
  epochs (`>= 5`) that a stall is meaningful — with very few epochs, a
  patience of 2 can trigger on ordinary epoch-to-epoch noise. Leave it at
  `0` (disabled) for short runs.
- `ADV_TRAINING_EARLY_STOP_PATIENCE` improvements are measured against
  `early_stopping_min_delta` (default `0.001`) — this is tuned for the
  0-1 AUC-ROC scale used throughout this codebase and shouldn't normally
  need adjustment.

## Acceptance-criteria verification

`tests/test_adversarial_curriculum.py` covers:

- `CurriculumScheduler` ramp correctness and boundary/validation behavior.
- Curriculum-disabled reproduces the pre-#872 report shape exactly
  (regression guard against the Issue #191 test suite,
  `tests/test_fgsm_adversarial_training.py`).
- Curriculum-enabled actually ramps `train_epsilon` per epoch while
  `adversarial_auc` stays measured at the fixed target epsilon.
- Early stopping triggers on a stalled robust-accuracy signal and leaves
  `epoch_log` truncated to the epochs actually run.
- Per-epoch clean/robust accuracy is written to the experiment tracker when
  one is supplied, and no file is written when it is not.
- **No training divergence across 3 seeds** (`TestNoDivergenceAcrossSeeds`):
  every epoch's `clean_auc`/`adversarial_auc` stays a finite value in
  `[0, 1]` for seeds 101/202/303 with `curriculum="linear"` enabled.
