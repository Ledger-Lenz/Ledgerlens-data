# CLI Command Contracts for Operational Workflows

## Problem

`scripts/` holds ~49 standalone entry points used for real operational
workflows: scoring a wallet on demand (`score_wallet.py`), replaying a
Kafka topic during an incident (`replay_stream.py`), scaling consumer
workers (`kafka_workers.py`), managing the human annotation queue
(`manage_queue.py`), running a backtest (`backtest.py`), kicking off an
active-learning round (`run_active_learning.py`). Each script owns its own
`argparse` parser with no shared, reviewable definition of what an on-call
operator can rely on -- a required flag silently becoming optional (or vice
versa), or a flag being renamed, was previously only discoverable by
reading the script's source or running `--help`.

## Design

- **`scripts/cli_contracts.py`** declares one `CliContract` per
  operationally-important script: its flags/positionals, which are
  required, and a short description of each -- effectively a machine-checked
  version of the `--help` text, doubling as operator-facing documentation.
- **`scripts/check_cli_contracts.py`** parses the real script with `ast`
  (no execution -- several of these scripts import Kafka clients or ML
  frameworks that shouldn't be required just to lint the CLI surface) and
  extracts every `add_argument(...)` call's alias(es) and `required` flag,
  then diffs it against the declared contract:
  - a contract entry with no matching `add_argument` call -> **missing**
    (renamed/removed without updating the contract)
  - an `add_argument` call not covered by any contract entry -> **undeclared**
    (a new flag shipped without documenting it as part of the operational
    surface)
  - a `required=` mismatch between contract and source -> **required mismatch**

## Validation

```
python scripts/check_cli_contracts.py                     # all contracted scripts
python scripts/check_cli_contracts.py --script backtest.py
make check-cli-contracts                                   # same, via Makefile
pytest tests/test_cli_contracts.py -q                       # unit tests
```

`tests/test_cli_contracts.py` exercises extraction and diffing logic
against synthetic scripts (single/multi-alias arguments, positionals,
missing/undeclared/required-mismatch diagnostics), plus a real-contract
smoke test that all six contracted scripts (`score_wallet.py`,
`backtest.py`, `manage_queue.py`, `kafka_workers.py`, `replay_stream.py`,
`run_active_learning.py`) currently match `scripts/cli_contracts.py`
exactly. Both the standalone script and the pytest run are wired into CI.

## Tradeoffs / follow-up

- Contracts are declared for the 6 scripts with the widest operational
  blast radius (incident response, scoring, backtesting, queue
  management), not all 49 files in `scripts/`. Extending coverage is
  additive: add a `CliContract` entry and rerun the checker.
- Extraction is script-wide rather than per-subcommand -- `manage_queue.py`
  uses subparsers (`list` / `annotate` / `skip` / `export`), and the
  contract treats their flags as one flat set. This was a deliberate
  scope tradeoff to keep the contract model simple; a future iteration
  could add a `subcommand` field to `CliArgument` if per-subcommand
  precision becomes necessary.
- Like the API-compatibility check (`docs/api_compatibility.md`), this is
  static analysis only: it validates the *shape* of the CLI contract, not
  runtime behavior (e.g. it won't catch a flag whose value is silently
  ignored). Runtime coverage for individual scripts already exists in
  `tests/test_score_wallet.py`, `tests/test_backtest.py`, etc.

---

## Dry-run mode (Issue #960)

All state-mutating CLI commands must support a `--dry-run` flag.  When
active, the command:

1. Displays a formatted plan enumerating *exactly* which writes would occur
   (file paths, table names, row counts, date ranges).
2. Exits **without** performing any mutation — zero side effects guaranteed.
3. Exits with code `0`.

**Recommendation: always run `--dry-run` first before executing any
state-mutating command in production.**

```bash
python -m scripts.restore --dry-run           # preview restore plan
python -m scripts.backfill_amm_trades --dry-run --pool-ids <id>
ledgerlens-ops validate-artifacts --dry-run
```

The `cli.dry_run` module provides:
- `add_dry_run_argument(parser)` — consistent flag registration
- `DryRunPlan` / `DryRunAction` — structured plan description
- `check_dry_run(args, plan)` — print & return True if dry-run active

Tests: `tests/test_cli_dry_run.py`

---

## Interactive confirmation & blast-radius summary (Issue #962)

Destructive commands present a **blast-radius summary** before prompting
for explicit confirmation.  The summary includes:

- Operation name
- Affected record counts (by table)
- Date range of affected data
- Affected tenants / environments
- Any additional context (backup timestamp, target database URL, etc.)

The operator must type exactly `yes` to proceed; any other input cancels.

```bash
python -m scripts.restore     # shows blast-radius, prompts for confirmation
```

**Non-interactive override (`--yes`):**

```bash
python -m scripts.restore --yes   # ⚠️  skips confirmation — CI/automation only
# or
LEDGERLENS_YES=1 python -m scripts.restore
```

> **Warning:** `--yes` / `LEDGERLENS_YES=1` bypasses the human safety gate
> entirely.  Only use it in CI pipelines or automation scripts where the
> blast radius has already been reviewed.  **Never** use it as a shortcut
> during ad-hoc production operations.

The `cli.confirmation` module provides:
- `BlastRadiusSummary` — structured blast-radius description
- `confirm_destructive_action(summary, non_interactive=False)` — prompt logic

Tests: `tests/test_cli_confirmation.py`

---

## CLI audit logging (Issue #961)

Every state-mutating CLI command produces an audit-log entry when running
in a production-configured environment (`LEDGERLENS_ENV=production`).  The
entry is appended to the same NDJSON trail as forensic-report entries
(`AUDIT_LOG_PATH`, default `data/audit_trail.ndjson`).

Each entry captures:
- `event_type`: `"cli_command"`
- `command`: command name
- `actor`: from `LEDGERLENS_ACTOR`, `USER`, or `"unknown"`
- `args`: redacted argument dict (secrets replaced with `"[REDACTED]"`)
- `outcome`: `"success"` or `"failure"`
- `error`: error message on failure
- `timestamp`: UTC ISO-8601

**Activating production auditing:**

```bash
export LEDGERLENS_ENV=production
python -m scripts.restore --yes         # audit entry written on exit
```

**Force auditing in non-production (staging/CI):**

```bash
export AUDIT_ALL_ENVIRONMENTS=1
```

**Secret redaction:** any argument key matching `secret`, `password`,
`token`, `key`, `credential`, `passphrase`, or `private` (case-insensitive)
has its value replaced with `"[REDACTED]"` in the audit entry.

The `cli.audit` module provides:
- `audit_cli_command(command, args)` — context manager
- `cli_audit_hook(command)` — decorator
- `is_production_env()` — production detection
- `redact_secrets(args)` — secret scrubbing

Tests: `tests/test_cli_audit.py`
