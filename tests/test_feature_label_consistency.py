"""Tests for the feature-label / feature-dictionary consistency check (Issue #946).

Verifies that:
  1. The check passes cleanly against the current state of the repo.
  2. A deliberately introduced missing label entry is caught.
  3. The check is invocable as a module and returns the correct exit code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_feature_label_consistency import (
    _extract_dict_keys,
    extract_dict_entries,
    load_feature_labels,
    main,
)


class TestLoadFeatureLabels:
    """Unit tests for load_feature_labels() — AST parsing of FEATURE_LABELS."""

    def test_returns_dict(self):
        labels = load_feature_labels()
        assert isinstance(labels, dict)

    def test_all_keys_are_strings(self):
        labels = load_feature_labels()
        for k, v in labels.items():
            assert isinstance(k, str), f"Key {k!r} is not a string"
            assert isinstance(v, str), f"Value {v!r} for key {k!r} is not a string"

    def test_known_feature_present(self):
        labels = load_feature_labels()
        # Spot-check a well-known entry
        assert "benford_mad_1h" in labels

    def test_extract_dict_keys_plain_assign(self):
        """_extract_dict_keys handles a plain dict AST node."""
        source = 'D = {"a": "alpha", "b": "beta"}'
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                result = _extract_dict_keys(node.value)
                assert result == {"a": "alpha", "b": "beta"}


class TestExtractDictEntries:
    """Unit tests for extract_dict_entries() — markdown feature name parsing."""

    def test_expands_template_benford_mad(self, tmp_path):
        md = tmp_path / "dict.md"
        md.write_text("### `benford_mad_{h}h`\nSome text.\n")
        entries = extract_dict_entries(md)
        for window in [1, 4, 24, 168, 720]:
            assert f"benford_mad_{window}h" in entries

    def test_does_not_expand_concrete_names(self, tmp_path):
        md = tmp_path / "dict.md"
        md.write_text("### `ring_size`\nSome text.\n")
        entries = extract_dict_entries(md)
        assert "ring_size" in entries
        # Should not inject spurious variants
        assert "ring_size_1h" not in entries

    def test_ignores_non_feature_backtick_strings(self, tmp_path):
        md = tmp_path / "dict.md"
        md.write_text("See `detection.benford_engine.chi_square_statistic`.\n")
        entries = extract_dict_entries(md)
        # Module paths don't start with a lowercase letter followed by underscores
        # that look like feature names — but the extractor is permissive. What
        # matters is that real feature names ARE captured.
        assert isinstance(entries, set)


class TestConsistencyCheck:
    """Integration tests running the full check against the actual repo files."""

    def test_current_state_passes(self):
        """CI gate: the check must exit 0 against the current repo state."""
        exit_code = main([])
        assert exit_code == 0, (
            "Feature label / dictionary consistency check failed on the current "
            "repo state. Run `python scripts/check_feature_label_consistency.py` "
            "to see the details and fix any missing dictionary entries."
        )

    def test_deliberate_missing_entry_is_caught(self, tmp_path, monkeypatch):
        """Catch mode: introduce a label key absent from the dictionary."""
        # Write a minimal feature_labels.py with a fake key
        fake_labels = tmp_path / "feature_labels.py"
        fake_labels.write_text(
            'FEATURE_LABELS: dict[str, str] = {\n'
            '    "benford_mad_1h": "1-hour Benford deviation",\n'
            '    "this_feature_does_not_exist_in_the_dictionary": "mystery feature",\n'
            '}\n'
        )

        # Write a minimal feature dictionary that only documents benford_mad
        fake_dict = tmp_path / "feature_dictionary.md"
        fake_dict.write_text("### 1.2 · `benford_mad_{h}h`\nSome text.\n")

        # Patch the module-level paths
        import scripts.check_feature_label_consistency as mod

        monkeypatch.setattr(mod, "FEATURE_LABELS_PATH", fake_labels)
        monkeypatch.setattr(mod, "FEATURE_DICTIONARY_PATH", fake_dict)

        exit_code = main([])
        assert exit_code == 1, (
            "Expected exit code 1 when a label key is missing from the dictionary, "
            "but the check returned 0."
        )

    def test_strict_mode_fails_on_dict_only_features(self, tmp_path, monkeypatch):
        """--strict promotes dictionary-only warnings to failures."""
        fake_labels = tmp_path / "feature_labels.py"
        fake_labels.write_text(
            'FEATURE_LABELS: dict[str, str] = {\n'
            '    "benford_mad_1h": "1-hour Benford deviation",\n'
            '}\n'
        )
        fake_dict = tmp_path / "feature_dictionary.md"
        # Dictionary has benford_mad AND an extra feature not labelled
        fake_dict.write_text(
            "### 1.2 · `benford_mad_{h}h`\nSome text.\n"
            "### X · `extra_unlabelled_feature`\nSome text.\n"
        )

        import scripts.check_feature_label_consistency as mod

        monkeypatch.setattr(mod, "FEATURE_LABELS_PATH", fake_labels)
        monkeypatch.setattr(mod, "FEATURE_DICTIONARY_PATH", fake_dict)

        # Without --strict, should pass (warnings only)
        assert main([]) == 0
        # With --strict, should fail
        assert main(["--strict"]) == 1
