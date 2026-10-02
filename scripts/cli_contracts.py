"""Declarative CLI contracts for Ledgerlens-data's operational scripts.

``scripts/`` has ~49 standalone entry points used for on-call/operational
workflows (scoring a wallet on demand, replaying a Kafka topic, running a
backtest, managing the annotation queue, ...). Each defines its own
``argparse`` parser, and nothing previously documented -- outside of
``--help`` output buried in each file -- which flags an operator can rely
on, which are required, and which scripts even exist for a given task.

This module is the single source of truth for that contract: one
:class:`CliContract` per operationally-important script, describing its
required/optional flags. It is validated against the real ``argparse``
definitions by ``scripts/check_cli_contracts.py`` (``make
check-cli-contracts``), so this file can't silently drift from the actual
CLI -- adding a flag to a script without updating its contract here (or
vice versa) fails CI with a diagnostic pointing at the exact line.

Adding a new operational script to the contract
-------------------------------------------------
1. Add a :class:`CliContract` entry to ``CONTRACTS`` below, keyed by the
   script's filename under ``scripts/``.
2. Run ``python scripts/check_cli_contracts.py`` -- it will report any
   mismatch between what you declared and the script's actual
   ``add_argument`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CliArgument:
    """One flag or positional argument in a script's contract.

    ``name`` is the canonical long form (e.g. ``"--wallet"``) or, for a
    positional argument, its dest name (e.g. ``"wallet"``) with no leading
    dashes.
    """

    name: str
    required: bool = False
    description: str = ""

    @property
    def is_positional(self) -> bool:
        return not self.name.startswith("-")


@dataclass(frozen=True)
class CliContract:
    script: str
    command: str
    description: str
    arguments: tuple[CliArgument, ...]

    def argument_names(self) -> set[str]:
        return {a.name for a in self.arguments}

    def required_argument_names(self) -> set[str]:
        return {a.name for a in self.arguments if a.required}


CONTRACTS: dict[str, CliContract] = {
    "score_wallet.py": CliContract(
        script="score_wallet.py",
        command="python -m scripts.score_wallet",
        description="Score one wallet, or many, on demand.",
        arguments=(
            CliArgument("--wallet", description="Stellar wallet public key (G...)"),
            CliArgument(
                "--wallets-file", description="Path to a file of newline-delimited wallets"
            ),
            CliArgument("--workers", description="Parallel worker count"),
            CliArgument("--pair", required=True, description="Trading pair to score against"),
            CliArgument("--since", description="Only consider trades since this timestamp"),
            CliArgument("--no-orderbook", description="Skip orderbook-derived features"),
            CliArgument("--json", description="Emit machine-readable JSON output"),
            CliArgument("--quiet", description="Suppress progress output"),
            CliArgument("--causal", description="Include causal attribution in the result"),
            CliArgument(
                "--what-if-remove", description="Counterfactual: rescore without this wallet"
            ),
            CliArgument("--log-level", description="Logging verbosity"),
        ),
    ),
    "backtest.py": CliContract(
        script="backtest.py",
        command="python -m scripts.backtest",
        description="Backtest the fraud-detection model over a historical window.",
        arguments=(
            CliArgument("--start", required=True, description="Start date, ISO format"),
            CliArgument("--end", required=True, description="End date, ISO format"),
            CliArgument("--model-path", description="Path to the model artifact to backtest"),
            CliArgument("--ground-truth", description="Path to labelled ground-truth data"),
            CliArgument("--output", description="Where to write the backtest report"),
            CliArgument("--threshold", description="Score threshold for a positive prediction"),
            CliArgument("--step-hours", description="Evaluation step size in hours"),
            CliArgument("--force-refresh", description="Ignore cached intermediate results"),
            CliArgument("--sliding-window", description="Use a sliding-window evaluation"),
            CliArgument("--window-days", description="Sliding window size in days"),
            CliArgument("--step-days", description="Sliding window step size in days"),
            CliArgument("--random-baseline", description="Also compute a random baseline"),
            CliArgument(
                "--random-baseline-simulations", description="Number of random baseline simulations"
            ),
        ),
    ),
    "manage_queue.py": CliContract(
        script="manage_queue.py",
        command="python -m scripts.manage_queue",
        description="Inspect and operate on the human annotation queue.",
        arguments=(
            CliArgument("--queue", description="Path to the annotation queue file"),
            CliArgument("--status", description="Filter queue entries by status"),
            CliArgument("--limit", description="Max entries to list"),
            CliArgument("wallet", description="Wallet ID to annotate or skip"),
            CliArgument("label", description="Label to apply when annotating"),
            CliArgument("--comment", description="Optional annotation comment"),
            CliArgument("--annotator-id", description="Identifier of the annotator"),
            CliArgument("--reason", description="Optional reason when skipping a wallet"),
            CliArgument("--output", required=True, description="Path to write an export"),
        ),
    ),
    "kafka_workers.py": CliContract(
        script="kafka_workers.py",
        command="python -m scripts.kafka_workers",
        description="Scale the Kafka consumer worker pool.",
        arguments=(
            CliArgument("--num-workers", required=True, description="Number of worker processes"),
            CliArgument("--topic", description="Kafka topic to consume"),
            CliArgument("--group", description="Kafka consumer group id"),
            CliArgument("--bootstrap-servers", description="Kafka bootstrap servers"),
        ),
    ),
    "replay_stream.py": CliContract(
        script="replay_stream.py",
        command="python -m scripts.replay_stream",
        description="Replay a historical window of a Kafka topic for reprocessing.",
        arguments=(
            CliArgument("--topic", description="Kafka topic to replay"),
            CliArgument("--bootstrap-servers", description="Kafka bootstrap servers"),
            CliArgument("--group", description="Consumer group id used for the replay"),
            CliArgument("--from-timestamp", description="Start of the replay window"),
            CliArgument("--to-timestamp", description="End of the replay window"),
            CliArgument("--resume", description="Resume a previously interrupted replay"),
            CliArgument("--dry-run", description="Preview the replay without publishing"),
            CliArgument("--confirm", description="Skip the interactive confirmation prompt"),
        ),
    ),
    "run_active_learning.py": CliContract(
        script="run_active_learning.py",
        command="python -m scripts.run_active_learning",
        description="Run one active-learning query-selection round.",
        arguments=(
            CliArgument("--pool", required=True, description="Path to the unscored wallet pool"),
            CliArgument("--strategy", description="Query strategy to use"),
            CliArgument("--batch-size", description="Number of wallets to select"),
            CliArgument("--queue", description="Path to the annotation queue to append to"),
            CliArgument("--model-dir", description="Directory containing model artifacts"),
            CliArgument("--asset-pair", description="Restrict selection to one asset pair"),
            CliArgument("--update", description="Update the pool in place after selection"),
            CliArgument("--historical", description="Include historical wallets in the pool"),
            CliArgument(
                "--force-continue", description="Continue past non-fatal validation warnings"
            ),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Operational CLI (cli/main.py) — structured-output contracts (Issue #959)
#
# These contracts describe the *subcommands* of the ``ledgerlens-ops`` CLI
# (``python -m cli.main``).  They are checked by a dedicated section in
# ``scripts/check_cli_contracts.py`` that parses ``cli/main.py`` rather than
# a ``scripts/`` file.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CliSubcommandContract:
    """Contract for one subcommand of a non-scripts CLI module."""

    module: str            # dotted module path, e.g. "cli.main"
    subcommand: str        # subcommand name, e.g. "healthcheck"
    description: str
    arguments: tuple[CliArgument, ...]
    json_schema_version: str = "1.0"
    json_schema_fields: tuple[str, ...] = ()

    def argument_names(self) -> set[str]:
        return {a.name for a in self.arguments}


CLI_SUBCOMMAND_CONTRACTS: dict[str, CliSubcommandContract] = {
    "cli.main:healthcheck": CliSubcommandContract(
        module="cli.main",
        subcommand="healthcheck",
        description="Run diagnostic health checks on environment variables and streaming config.",
        arguments=(
            CliArgument("--json", description="Emit machine-readable JSON output"),
        ),
        json_schema_version="1.0",
        json_schema_fields=(
            "schema_version",
            "overall_status",
            "checks.environment.status",
            "checks.environment.details",
            "checks.environment.missing",
            "checks.streaming.status",
            "checks.streaming.backend",
        ),
    ),
    "cli.main:validate-artifacts": CliSubcommandContract(
        module="cli.main",
        subcommand="validate-artifacts",
        description="Validate local model and schema artifacts.",
        arguments=(
            CliArgument("--dir", description="Path to artifacts folder"),
            CliArgument("--json", description="Emit machine-readable JSON output"),
        ),
        json_schema_version="1.0",
        json_schema_fields=(
            "schema_version",
            "status",
            "artifacts_dir",
            "version",
            "schema_hash",
            "error",
        ),
    ),
}


__all__ = ["CliArgument", "CliContract", "CONTRACTS", "CliSubcommandContract", "CLI_SUBCOMMAND_CONTRACTS"]
