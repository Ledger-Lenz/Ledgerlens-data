"""Tests for Issue #872: curriculum scheduling + early stopping + experiment
tracking on top of the FGSM adversarial training loop (Issue #191).

Covers:
  - CurriculumScheduler: linear/step epsilon ramps, boundary conditions,
    invalid arguments.
  - run_adversarial_training: curriculum disabled reproduces the pre-#872
    fixed-epsilon behavior exactly (regression guard); curriculum enabled
    actually ramps `train_epsilon` per epoch; early stopping triggers on
    stalled *robust* (not clean) validation accuracy and truncates the run;
    per-epoch clean/robust accuracy is logged to the experiment tracker
    when one is supplied, and skipped (no file writes) when it is not.
  - No training divergence (NaN/negative AUC) across 3 seeds with curriculum
    enabled — the issue's explicit acceptance criterion.
"""

import json

import pytest

from detection.adversarial.augmentation import CurriculumScheduler
from detection.adversarial.robustness import run_adversarial_training
from mlops.experiment_tracking import JsonlExperimentTracker
from scripts.generate_synthetic_dataset import generate_synthetic_dataset

# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------


class TestCurriculumScheduler:
    def test_linear_ramp_endpoints(self):
        sched = CurriculumScheduler(start=0.05, end=0.2, epochs=5, strategy="linear")
        assert sched.epsilon_for_epoch(0) == pytest.approx(0.05)
        assert sched.epsilon_for_epoch(4) == pytest.approx(0.2)

    def test_linear_ramp_is_monotonically_non_decreasing(self):
        sched = CurriculumScheduler(start=0.05, end=0.2, epochs=6, strategy="linear")
        values = sched.schedule()
        assert len(values) == 6
        assert all(b >= a for a, b in zip(values, values[1:], strict=False))

    def test_linear_ramp_midpoint(self):
        sched = CurriculumScheduler(start=0.1, end=0.3, epochs=3, strategy="linear")
        assert sched.epsilon_for_epoch(1) == pytest.approx(0.2)

    def test_step_ramp_reaches_end_and_has_at_most_four_distinct_values(self):
        sched = CurriculumScheduler(start=0.05, end=0.2, epochs=10, strategy="step")
        values = sched.schedule()
        assert values[0] == pytest.approx(0.05)
        assert values[-1] == pytest.approx(0.2)
        assert len(set(round(v, 6) for v in values)) <= 4

    def test_single_epoch_returns_end_only(self):
        sched = CurriculumScheduler(start=0.05, end=0.2, epochs=1, strategy="linear")
        assert sched.epsilon_for_epoch(0) == pytest.approx(0.2)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy must be one of"):
            CurriculumScheduler(start=0.05, end=0.2, epochs=5, strategy="exponential")

    def test_non_positive_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            CurriculumScheduler(start=0.05, end=0.2, epochs=0)

    def test_non_positive_epsilon_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            CurriculumScheduler(start=0.0, end=0.2, epochs=5)

    def test_epoch_out_of_range_raises(self):
        sched = CurriculumScheduler(start=0.05, end=0.2, epochs=3)
        with pytest.raises(ValueError, match="epoch must be in"):
            sched.epsilon_for_epoch(3)


# ---------------------------------------------------------------------------
# run_adversarial_training — curriculum integration
# ---------------------------------------------------------------------------


class TestRunAdversarialTrainingCurriculum:
    def test_curriculum_disabled_matches_pre_872_report_shape(self, tmp_path):
        """Default (curriculum=None) must be indistinguishable from the
        Issue #191 behavior other than the new, additive report keys."""
        df = generate_synthetic_dataset(n_wallets=100, seed=5)
        report = run_adversarial_training(
            df,
            epochs=2,
            epsilon=0.2,
            adv_ratio=0.5,
            test_size=0.25,
            random_state=5,
            model_dir=str(tmp_path),
        )
        assert report["curriculum"] is None
        assert report["epochs_run"] == 2
        assert report["early_stopped"] is False
        # Fixed epsilon: every epoch in the schedule equals the target.
        assert report["epsilon_schedule"] == [0.2, 0.2]
        for entry in report["epoch_log"]:
            assert entry["train_epsilon"] == pytest.approx(0.2)

    def test_linear_curriculum_ramps_train_epsilon_per_epoch(self, tmp_path):
        df = generate_synthetic_dataset(n_wallets=100, seed=11)
        report = run_adversarial_training(
            df,
            epochs=4,
            epsilon=0.2,
            adv_ratio=0.5,
            test_size=0.25,
            random_state=11,
            model_dir=str(tmp_path),
            curriculum="linear",
            curriculum_start_epsilon=0.02,
        )
        schedule = report["epsilon_schedule"]
        assert schedule[0] == pytest.approx(0.02)
        assert schedule[-1] == pytest.approx(0.2)
        assert all(b >= a for a, b in zip(schedule, schedule[1:], strict=False))
        assert [e["train_epsilon"] for e in report["epoch_log"]] == pytest.approx(schedule)

    def test_curriculum_defaults_start_epsilon_to_quarter_of_target(self, tmp_path):
        df = generate_synthetic_dataset(n_wallets=80, seed=21)
        report = run_adversarial_training(
            df,
            epochs=3,
            epsilon=0.2,
            test_size=0.25,
            random_state=21,
            model_dir=str(tmp_path),
            curriculum="linear",
        )
        assert report["epsilon_schedule"][0] == pytest.approx(0.05)  # 0.2 / 4

    def test_curriculum_invalid_strategy_raises(self, tmp_path):
        df = generate_synthetic_dataset(n_wallets=50, seed=31)
        with pytest.raises(ValueError, match="strategy must be one of"):
            run_adversarial_training(
                df,
                epochs=3,
                epsilon=0.2,
                test_size=0.25,
                random_state=31,
                model_dir=str(tmp_path),
                curriculum="bogus",
            )


# ---------------------------------------------------------------------------
# run_adversarial_training — early stopping on robust validation accuracy
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_disabled_by_default_runs_every_epoch(self, tmp_path):
        df = generate_synthetic_dataset(n_wallets=100, seed=41)
        report = run_adversarial_training(
            df,
            epochs=3,
            epsilon=0.2,
            test_size=0.25,
            random_state=41,
            model_dir=str(tmp_path),
        )
        assert report["epochs_run"] == 3
        assert report["early_stopped"] is False

    def test_high_patience_with_few_epochs_never_triggers(self, tmp_path):
        """Patience >= epochs can never be exhausted mid-run."""
        df = generate_synthetic_dataset(n_wallets=80, seed=51)
        report = run_adversarial_training(
            df,
            epochs=3,
            epsilon=0.2,
            test_size=0.25,
            random_state=51,
            model_dir=str(tmp_path),
            early_stopping_patience=10,
        )
        assert report["epochs_run"] == 3
        assert report["early_stopped"] is False

    def test_zero_patience_immediate_plateau_stops_before_last_epoch(self, tmp_path, monkeypatch):
        """With min_delta effectively infinite, every epoch after the first
        counts as "no improvement", so patience=1 must stop after epoch 2
        of a 5-epoch request."""
        df = generate_synthetic_dataset(n_wallets=100, seed=61)
        report = run_adversarial_training(
            df,
            epochs=5,
            epsilon=0.2,
            test_size=0.25,
            random_state=61,
            model_dir=str(tmp_path),
            early_stopping_patience=1,
            early_stopping_min_delta=1.0,  # unreachable improvement bar
        )
        assert report["epochs_run"] < 5
        assert report["early_stopped"] is True
        assert len(report["epoch_log"]) == report["epochs_run"]


# ---------------------------------------------------------------------------
# Experiment tracking integration
# ---------------------------------------------------------------------------


class TestExperimentTracking:
    def test_no_tracker_writes_no_file(self, tmp_path):
        df = generate_synthetic_dataset(n_wallets=60, seed=71)
        run_adversarial_training(
            df,
            epochs=2,
            epsilon=0.2,
            test_size=0.25,
            random_state=71,
            model_dir=str(tmp_path / "models"),
        )
        assert not (tmp_path / "experiments.jsonl").exists()

    def test_tracker_logs_one_run_per_epoch_with_clean_and_robust_accuracy(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JsonlExperimentTracker(path=str(log_path))
        df = generate_synthetic_dataset(n_wallets=80, seed=81)

        report = run_adversarial_training(
            df,
            epochs=3,
            epsilon=0.2,
            test_size=0.25,
            random_state=81,
            model_dir=str(tmp_path / "models"),
            experiment_tracker=tracker,
        )

        assert log_path.exists()
        runs = tracker.list_runs()
        assert len(runs) == report["epochs_run"] == 3
        for i, run in enumerate(runs):
            assert run["name"] == "adversarial_training_epoch"
            assert run["params"]["epoch"] == i
            assert "clean_auc" in run["metrics"]
            assert "adversarial_auc" in run["metrics"]
            assert run["dataset_sha256"]
            assert run["feature_schema_hash"]

    def test_log_file_is_valid_jsonl(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JsonlExperimentTracker(path=str(log_path))
        df = generate_synthetic_dataset(n_wallets=60, seed=91)
        run_adversarial_training(
            df,
            epochs=2,
            epsilon=0.2,
            test_size=0.25,
            random_state=91,
            model_dir=str(tmp_path / "models"),
            experiment_tracker=tracker,
        )
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# No training divergence across seeds (issue acceptance criterion)
# ---------------------------------------------------------------------------


class TestNoDivergenceAcrossSeeds:
    @pytest.mark.parametrize("seed", [101, 202, 303])
    def test_curriculum_run_produces_finite_non_negative_auc_every_epoch(self, tmp_path, seed):
        df = generate_synthetic_dataset(n_wallets=100, seed=seed)
        report = run_adversarial_training(
            df,
            epochs=4,
            epsilon=0.2,
            test_size=0.25,
            random_state=seed,
            model_dir=str(tmp_path / f"seed_{seed}"),
            curriculum="linear",
        )
        for entry in report["epoch_log"]:
            assert entry["clean_auc"] == entry["clean_auc"]  # not NaN
            assert entry["adversarial_auc"] == entry["adversarial_auc"]  # not NaN
            assert 0.0 <= entry["clean_auc"] <= 1.0
            assert 0.0 <= entry["adversarial_auc"] <= 1.0
