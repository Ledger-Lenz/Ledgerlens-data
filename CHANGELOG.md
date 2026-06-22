# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Adversarial robustness evaluation framework (`detection/adversarial/`):
  FGSM and PGD attacks (Madry et al., 2018) against the wash-trade ensemble
  using finite-difference gradients on the continuous ensemble score, an
  `AdversarialEvaluator` reporting attack success rate, per-feature minimum
  L-infinity epsilon, and most-vulnerable features, and an
  `adversarial_augmentation_gain` adversarial-training loop measuring the
  AUC-ROC robustness gain.
- `scripts/run_adversarial_eval.py` CLI that produces a JSON robustness report
  (with usage docs in `scripts/README.md`).
- Test coverage for the adversarial attacks, evaluator, and augmentation loop
  (`tests/test_adversarial.py`).

## [0.2.0] - 2026-06-13

### Added
- MIT LICENSE.
- Project tooling: `pyproject.toml` (ruff/black/mypy/pytest config), `Makefile`,
  pre-commit hooks, and CI workflow (lint + test on Python 3.11/3.12).
- `Dockerfile` / `.dockerignore` for containerized runs.
- `CONTRIBUTING.md` with local dev setup and PR guidelines.
- GitHub issue templates (bug report, feature request) and a pull request
  template.
- Structured logging (`utils/logging.py`) and a retry/backoff helper
  (`utils/retry.py`) for Horizon API calls.
- Persistence layer for `RiskScore` records (`detection/persistence.py`,
  `detection/risk_score_store.py`) backed by SQLAlchemy and `RISK_SCORE_DB_URL`.
- Order-book event ingestion (`ingestion/orderbook_loader.py`) and a real
  `order_cancellation_rate` feature.
- Wallet funding-graph features: `funding_source_similarity` and
  `network_centrality` (`detection/wallet_graph.py`).
- Soroban contract client (`integrations/contract_client.py`) for
  `submit_score` / `get_score` against `ledgerlens-score`.
- Synthetic labelled dataset generator (`scripts/generate_synthetic_dataset.py`,
  with usage docs in `scripts/README.md`) and a `model_training.py` CLI for
  local training/demo runs.
- Ensemble SHAP aggregation (`ShapExplainer.explain_ensemble`) and explainer
  caching.
- Test coverage for persistence, order-book ingestion, wallet graph features,
  the contract client, the training CLI, and ensemble inference/SHAP.

### Changed
- `run_pipeline.py` now loads order-book events, persists scored wallets,
  and supports `--no-orderbook`, `--no-persist`, and `--submit-onchain`
  flags.
- `model_inference.py`'s ensemble combination is now a configurable
  `_combine_probabilities` helper, and `confidence` reflects inter-model
  agreement rather than mirroring `score`.

### Fixed
- `RiskScorer.score` and `ShapExplainer` now coerce feature rows to numeric
  dtypes before calling models/explainers, fixing failures with XGBoost and
  newer SHAP versions.
