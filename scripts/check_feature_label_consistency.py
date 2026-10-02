"""CI check: verify consistency between feature_labels.py and feature_dictionary.md.

Issue #946 — feature_labels.py and data/feature_dictionary.md risk drifting out
of sync as features are added or renamed. This script fails CI when:

1. A feature key in ``reporting.feature_labels.FEATURE_LABELS`` has no
   corresponding entry in ``data/feature_dictionary.md``.
2. A feature key in ``reporting.feature_labels.FEATURE_LABELS`` is not produced
   by any ``detection/feature_engineering.py`` feature-builder call (optional
   cross-check, enabled with ``--check-code``).

The inverse direction (features in the dictionary but not labelled) is reported
as a **warning** only — dictionary entries often cover more features than the
subset surfaced in narrative text.

Usage::

    python scripts/check_feature_label_consistency.py
    python scripts/check_feature_label_consistency.py --check-code
    python scripts/check_feature_label_consistency.py --strict  # warns→errors

Exit codes:
    0  All feature label keys are present in the feature dictionary.
    1  One or more feature label keys are missing from the feature dictionary.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_LABELS_PATH = REPO_ROOT / "reporting" / "feature_labels.py"
FEATURE_DICTIONARY_PATH = REPO_ROOT / "data" / "feature_dictionary.md"
FEATURE_ENGINEERING_PATH = REPO_ROOT / "detection" / "feature_engineering.py"

# Matches a feature name (backtick-wrapped) inside the feature_dictionary.md.
# Feature names appear as:
#   ### 1.1 · `benford_chi_square_{h}h`
# or as inline code: `counterparty_concentration_ratio`
# We also need to match template names like `benford_mad_{h}h`.
_DICT_FEATURE_RE = re.compile(r"`([a-z][a-z0-9_{}/]*)`")


def load_feature_labels() -> dict[str, str]:
    """Parse FEATURE_LABELS from feature_labels.py without importing it.

    Uses the AST so the check works even when the reporting package cannot be
    imported (e.g. in a minimal CI environment without all optional deps).
    Handles both plain ``Assign`` and annotated ``AnnAssign`` forms.
    """
    source = FEATURE_LABELS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FEATURE_LABELS_PATH))
    for node in ast.walk(tree):
        # Plain assignment: FEATURE_LABELS = {...}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FEATURE_LABELS":
                    if isinstance(node.value, ast.Dict):
                        return _extract_dict_keys(node.value)
        # Annotated assignment: FEATURE_LABELS: dict[str, str] = {...}
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "FEATURE_LABELS":
                if node.value is not None and isinstance(node.value, ast.Dict):
                    return _extract_dict_keys(node.value)
    raise RuntimeError(
        f"Could not find FEATURE_LABELS dict assignment in {FEATURE_LABELS_PATH}"
    )


def _extract_dict_keys(dict_node: ast.Dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in zip(dict_node.keys, dict_node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            label = value.value if isinstance(value, ast.Constant) else str(value)
            result[key.value] = label
    return result


def extract_dict_entries(dictionary_path: Path) -> set[str]:
    """Return every feature name mentioned in the feature_dictionary.md.

    We look for backtick-wrapped identifiers that look like feature names
    (all-lowercase, underscores, no special chars). Template names like
    ``benford_chi_square_{h}h`` are expanded into their concrete variants
    automatically.
    """
    text = dictionary_path.read_text(encoding="utf-8")
    raw: set[str] = set()
    for match in _DICT_FEATURE_RE.finditer(text):
        raw.add(match.group(1))

    # Expand template names — e.g. `benford_chi_square_{h}h` → real feature
    # names. We detect templates by the presence of '{' in the name; the
    # windows are hard-coded here to match the engine defaults.
    expanded: set[str] = set()
    windows = [1, 4, 24, 168, 720]
    for name in raw:
        if "{h}" in name:
            for w in windows:
                expanded.add(name.replace("{h}", str(w)))
        else:
            expanded.add(name)
    return expanded


def extract_produced_features(engineering_path: Path) -> set[str]:
    """Heuristically extract feature column names from feature_engineering.py.

    Looks for string literals that look like feature names in assignment
    statements like ``features["benford_mad_1h"] = ...`` or dict literals.
    This is an approximation — generated column names (e.g. looped over
    windows) are expanded using the standard window list.
    """
    source = engineering_path.read_text(encoding="utf-8")
    # Subscript assignments: result["feature_name"] = ...
    subscript_re = re.compile(r'\[\s*["\']([a-z][a-z0-9_]*)["\']')
    names = set(subscript_re.findall(source))

    # Also capture string constants that look like column names
    col_re = re.compile(r'["\']([a-z][a-z0-9_]{3,})["\']')
    for match in col_re.finditer(source):
        candidate = match.group(1)
        # Filter out obvious non-feature strings (paths, log messages, etc.)
        if "_" in candidate and not any(
            kw in candidate for kw in ("import", "error", "warning", "http", "://")
        ):
            names.add(candidate)

    return names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify consistency between reporting/feature_labels.py and "
            "data/feature_dictionary.md (Issue #946)."
        )
    )
    parser.add_argument(
        "--check-code",
        action="store_true",
        help=(
            "Also verify that every labelled feature is produced by "
            "detection/feature_engineering.py (heuristic; best-effort)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat dictionary-only warnings as errors (fail CI).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    try:
        feature_labels = load_feature_labels()
    except Exception as exc:
        print(f"ERROR: Failed to parse feature_labels.py: {exc}")
        return 1

    try:
        dict_entries = extract_dict_entries(FEATURE_DICTIONARY_PATH)
    except Exception as exc:
        print(f"ERROR: Failed to read feature_dictionary.md: {exc}")
        return 1

    label_keys: set[str] = set(feature_labels)

    # ------------------------------------------------------------------
    # Check 1: every label key must have a dictionary entry
    # ------------------------------------------------------------------
    missing_from_dict = label_keys - dict_entries
    if missing_from_dict:
        print(
            f"FAIL: {len(missing_from_dict)} feature(s) in FEATURE_LABELS are missing "
            "from data/feature_dictionary.md:"
        )
        for name in sorted(missing_from_dict):
            label = feature_labels[name]
            print(f"  - {name!r}  (label: {label!r})")
        print()
        print("Fix: add an entry for each feature to data/feature_dictionary.md.")
        print(
            "See the contributor checklist in CONTRIBUTING.md section "
            "'Adding a new feature'."
        )
        return 1

    print(
        f"OK: all {len(label_keys)} feature label keys are present in "
        "data/feature_dictionary.md."
    )

    # ------------------------------------------------------------------
    # Check 2 (warning): dictionary entries with no label
    # ------------------------------------------------------------------
    dict_only = dict_entries - label_keys
    # Trim to names that look like real ML features (contain underscore,
    # long enough, no template placeholders)
    dict_only_filtered = {
        n for n in dict_only if "_" in n and len(n) > 5 and "{" not in n
    }
    if dict_only_filtered:
        level = "FAIL" if args.strict else "WARN"
        print(
            f"{level}: {len(dict_only_filtered)} feature(s) appear in "
            "data/feature_dictionary.md but have no label in FEATURE_LABELS "
            "(they fall back to the de-slugified name in reports):"
        )
        for name in sorted(dict_only_filtered):
            print(f"  - {name!r}")
        if args.strict:
            print()
            print(
                "Fix: add a plain-English label for each feature to "
                "reporting/feature_labels.FEATURE_LABELS."
            )
            return 1
        print(
            "  (This is a warning only. Use --strict to promote to a CI failure.)"
        )

    # ------------------------------------------------------------------
    # Check 3 (optional): label keys present in feature_engineering.py
    # ------------------------------------------------------------------
    if args.check_code:
        try:
            produced = extract_produced_features(FEATURE_ENGINEERING_PATH)
        except Exception as exc:
            print(f"WARN: could not parse feature_engineering.py: {exc}")
            produced = set()

        if produced:
            missing_from_code = {k for k in label_keys if k not in produced}
            if missing_from_code:
                print(
                    f"\nWARN: {len(missing_from_code)} feature label key(s) were not "
                    "found in detection/feature_engineering.py "
                    "(heuristic check — may include false positives):"
                )
                for name in sorted(missing_from_code):
                    print(f"  - {name!r}")
            else:
                print(
                    "OK: all feature label keys found in detection/feature_engineering.py."
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
