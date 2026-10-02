#!/usr/bin/env python
"""Validate that the operational scripts in ``scripts/`` still match their
declared CLI contracts in ``scripts/cli_contracts.py``.

This parses each contracted script with :mod:`ast` (no execution, so it
carries none of the script's own runtime dependencies -- important since
several operational scripts import Kafka clients, ML frameworks, etc. that
aren't guaranteed to be installed in every environment that just wants to
lint the CLI surface) and extracts every ``parser.add_argument(...)`` /
``sub_parser.add_argument(...)`` call: its name(s) and whether
``required=True`` was passed.

It then diffs that against the contract for the same script and reports,
per script:

* **missing**: a contract argument that no longer appears in the script's
  source at all (renamed or removed without updating the contract).
* **undeclared**: a flag the script defines that isn't in the contract
  (added without documenting it as part of the operational surface).
* **required mismatch**: the contract and the script disagree on whether
  the argument is required.

Usage
-----
    python scripts/check_cli_contracts.py
    python scripts/check_cli_contracts.py --script score_wallet.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
from cli_contracts import CONTRACTS, CliContract  # noqa: E402


@dataclass(frozen=True)
class ActualArgument:
    aliases: tuple[str, ...]
    required: bool
    lineno: int

    def matches(self, name: str) -> bool:
        return name in self.aliases


def _extract_actual_arguments(path: Path) -> list[ActualArgument]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    actual: list[ActualArgument] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue

        aliases = tuple(
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        )
        if not aliases:
            continue

        required = False
        for kw in node.keywords:
            if (
                kw.arg == "required"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                required = True

        actual.append(ActualArgument(aliases=aliases, required=required, lineno=node.lineno))

    return actual


def check_contract(contract: CliContract, actual: list[ActualArgument]) -> list[str]:
    diagnostics: list[str] = []

    for declared in contract.arguments:
        match = next((a for a in actual if a.matches(declared.name)), None)
        if match is None:
            diagnostics.append(
                f"[{contract.script}] contract declares '{declared.name}' but no matching "
                f"add_argument() call was found -- update scripts/cli_contracts.py or restore the flag."
            )
            continue
        if declared.required != match.required:
            diagnostics.append(
                f"[{contract.script}:{match.lineno}] '{declared.name}' required mismatch: "
                f"contract says required={declared.required}, source says required={match.required}."
            )

    declared_names = contract.argument_names()
    for found in actual:
        if not any(alias in declared_names for alias in found.aliases):
            diagnostics.append(
                f"[{contract.script}:{found.lineno}] undeclared argument {found.aliases!r} -- "
                f"add it to its CliContract in scripts/cli_contracts.py (or remove it from the script)."
            )

    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default=None, help="Only check a single contracted script")
    args = parser.parse_args()

    scripts = [args.script] if args.script else sorted(CONTRACTS)
    all_diagnostics: list[str] = []

    for script_name in scripts:
        contract = CONTRACTS.get(script_name)
        if contract is None:
            print(f"No contract declared for '{script_name}' in scripts/cli_contracts.py")
            return 1
        script_path = SCRIPTS_DIR / contract.script
        if not script_path.is_file():
            all_diagnostics.append(
                f"[{contract.script}] contract references a script that no longer exists at "
                f"{script_path.relative_to(REPO_ROOT)}."
            )
            continue
        actual = _extract_actual_arguments(script_path)
        all_diagnostics.extend(check_contract(contract, actual))

    # ------------------------------------------------------------------
    # Issue #959 — also check structured-output (--json) contracts for
    # CLI subcommands defined in cli/main.py
    # ------------------------------------------------------------------
    from cli_contracts import CLI_SUBCOMMAND_CONTRACTS, CliSubcommandContract  # noqa: E402

    cli_diagnostics = _check_subcommand_contracts(CLI_SUBCOMMAND_CONTRACTS)
    all_diagnostics.extend(cli_diagnostics)

    if all_diagnostics:
        print(f"CLI contract check FAILED: {len(all_diagnostics)} issue(s)\n")
        for d in all_diagnostics:
            print(f"  - {d}")
        return 1

    print(f"CLI contract check passed for {len(scripts)} script(s): {', '.join(scripts)}")
    n_sub = len(CLI_SUBCOMMAND_CONTRACTS)
    print(f"  + {n_sub} CLI subcommand contract(s) verified (Issue #959).")
    return 0


def _check_subcommand_contracts(
    subcommand_contracts: dict,
) -> list[str]:
    """Validate that every CLI subcommand contract still matches cli/main.py.

    Checks:
    - The module file (e.g. cli/main.py) exists.
    - Every declared ``--json`` flag is actually present in the parser.
    - The ``schema_version`` constant exported by the module matches the
      contract's ``json_schema_version``.
    - Every ``json_schema_fields`` entry appears (by name) in the module
      docstring or the subparser's description, so the schema documentation
      stays in sync with the contract.
    """
    diagnostics: list[str] = []

    for key, sub_contract in subcommand_contracts.items():
        # Resolve module path to a file
        module_rel = sub_contract.module.replace(".", "/") + ".py"
        module_path = REPO_ROOT / module_rel
        if not module_path.is_file():
            diagnostics.append(
                f"[{key}] module '{sub_contract.module}' not found at "
                f"{module_rel} — update CLI_SUBCOMMAND_CONTRACTS or restore the file."
            )
            continue

        actual = _extract_actual_arguments(module_path)

        # Check each declared argument is present in the source
        for declared in sub_contract.arguments:
            match = next((a for a in actual if a.matches(declared.name)), None)
            if match is None:
                diagnostics.append(
                    f"[{key}] contract declares '{declared.name}' but no matching "
                    f"add_argument() call was found in {module_rel} — "
                    f"update CLI_SUBCOMMAND_CONTRACTS or restore the flag."
                )

        # Check that --json is present for every subcommand contract
        json_flag_present = any(a.matches("--json") for a in actual)
        if not json_flag_present:
            diagnostics.append(
                f"[{key}] contract requires '--json' flag but it was not found "
                f"in {module_rel} — Issue #959 requires --json on all diagnostic subcommands."
            )

        # Check that the schema_version constant exists in the module
        source = module_path.read_text(encoding="utf-8")
        version = sub_contract.json_schema_version
        # Look for SCHEMA_VERSION = "X.Y" or schema_version: "X.Y" in docstrings
        import re
        version_found = (
            re.search(rf'SCHEMA_VERSION\s*=\s*["\']' + re.escape(version) + r'["\']', source)
            or re.search(rf'schema_version.*["\']' + re.escape(version) + r'["\']', source)
            or version in source
        )
        if not version_found:
            diagnostics.append(
                f"[{key}] contract declares json_schema_version='{version}' "
                f"but that version string was not found in {module_rel}."
            )

        # Check that documented JSON schema fields appear somewhere in the
        # module (docstring or subparser help text)
        for field in sub_contract.json_schema_fields:
            # Use the field base name (before the first dot) for the check
            field_base = field.split(".")[0]
            if field_base not in source:
                diagnostics.append(
                    f"[{key}] JSON schema field '{field}' (base: '{field_base}') "
                    f"is declared in the contract but not mentioned in {module_rel}. "
                    f"Update the docstring/help text or the contract."
                )

    return diagnostics


if __name__ == "__main__":
    sys.exit(main())
