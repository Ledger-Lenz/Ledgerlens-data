"""Tests for Issue #192: DoWhy Causal Graph Validation Against Domain Expert Prior Knowledge.

Covers:
  - CausalPriorConstraints.load() validates YAML schema on load.
  - Unknown variables cause a startup ValueError.
  - Forbidden edge is not present in the learned graph after constraints applied.
  - Required edge is present in the learned graph regardless of data evidence.
  - warn_soft_conflicts emits warnings for forbidden-edge conflicts.
  - WashTradeCausalDiscovery.fit() correctly integrates priors.
  - causal_priors.yaml loads and validates successfully.
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from detection.causal_discovery import CausalPriorConstraints, WashTradeCausalDiscovery, _Constraint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Minimal numeric DataFrame with known feature names."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "account_age_days": rng.uniform(1, 1000, n),
            "trading_volume": rng.exponential(100, n),
            "counterparty_concentration": rng.uniform(0, 1, n),
            "round_trip_frequency": rng.uniform(0, 1, n),
            "funding_source_similarity": rng.uniform(0, 1, n),
            "benford_mad_24h": rng.uniform(0, 0.05, n),
            "label": rng.integers(0, 2, n),
        }
    )
    return df


def _priors_from_dict(d: dict) -> CausalPriorConstraints:
    return CausalPriorConstraints._from_dict(d, source="<test>")


# ---------------------------------------------------------------------------
# CausalPriorConstraints — schema validation
# ---------------------------------------------------------------------------


class TestCausalPriorConstraintsSchemaValidation:
    def test_valid_dict_parses_correctly(self):
        raw = {
            "constraints": [
                {"cause": "A", "effect": "B", "kind": "required"},
                {"cause": "C", "effect": "D", "kind": "forbidden"},
            ]
        }
        priors = _priors_from_dict(raw)
        assert len(priors) == 2
        assert priors.required == [("A", "B")]
        assert priors.forbidden == [("C", "D")]

    def test_missing_constraints_key_raises(self):
        with pytest.raises(ValueError, match="missing the required 'constraints' key"):
            _priors_from_dict({"not_constraints": []})

    def test_non_list_constraints_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _priors_from_dict({"constraints": "foo"})

    def test_non_mapping_item_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _priors_from_dict({"constraints": ["not a dict"]})

    def test_missing_cause_raises(self):
        with pytest.raises(ValueError, match="missing required key 'cause'"):
            _priors_from_dict({"constraints": [{"effect": "B", "kind": "required"}]})

    def test_missing_effect_raises(self):
        with pytest.raises(ValueError, match="missing required key 'effect'"):
            _priors_from_dict({"constraints": [{"cause": "A", "kind": "required"}]})

    def test_missing_kind_raises(self):
        with pytest.raises(ValueError, match="missing required key 'kind'"):
            _priors_from_dict({"constraints": [{"cause": "A", "effect": "B"}]})

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            _priors_from_dict({"constraints": [{"cause": "A", "effect": "B", "kind": "maybe"}]})

    def test_non_string_cause_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            _priors_from_dict({"constraints": [{"cause": 123, "effect": "B", "kind": "required"}]})

    def test_non_dict_root_raises(self):
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            _priors_from_dict(["not", "a", "dict"])

    def test_empty_constraints_list_is_valid(self):
        priors = _priors_from_dict({"constraints": []})
        assert len(priors) == 0

    def test_kind_is_case_insensitive(self):
        """Kind matching should be case-insensitive (REQUIRED / Forbidden etc)."""
        raw = {
            "constraints": [
                {"cause": "A", "effect": "B", "kind": "REQUIRED"},
                {"cause": "C", "effect": "D", "kind": "Forbidden"},
            ]
        }
        priors = _priors_from_dict(raw)
        assert priors.required == [("A", "B")]
        assert priors.forbidden == [("C", "D")]


# ---------------------------------------------------------------------------
# CausalPriorConstraints — variable validation
# ---------------------------------------------------------------------------


class TestCausalPriorConstraintsVariableValidation:
    def test_all_known_variables_passes(self):
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "required"}]}
        )
        priors.validate(["A", "B", "C"])  # should not raise

    def test_unknown_cause_raises(self):
        priors = _priors_from_dict(
            {"constraints": [{"cause": "UNKNOWN_VAR", "effect": "B", "kind": "forbidden"}]}
        )
        with pytest.raises(ValueError, match="UNKNOWN_VAR"):
            priors.validate(["B", "C"])

    def test_unknown_effect_raises(self):
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "GHOST", "kind": "required"}]}
        )
        with pytest.raises(ValueError, match="GHOST"):
            priors.validate(["A", "B"])

    def test_multiple_unknown_variables_all_listed(self):
        priors = _priors_from_dict(
            {
                "constraints": [
                    {"cause": "X1", "effect": "X2", "kind": "required"},
                    {"cause": "X3", "effect": "label", "kind": "forbidden"},
                ]
            }
        )
        with pytest.raises(ValueError, match="X1"):
            priors.validate(["label"])


# ---------------------------------------------------------------------------
# CausalPriorConstraints — graph enforcement
# ---------------------------------------------------------------------------


class TestCausalPriorConstraintsGraphEnforcement:
    def _dag_with_edges(self, edges: list[tuple[str, str]]) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_edges_from(edges)
        return g

    # Forbidden edge tests
    def test_forbidden_edge_removed_from_dag(self):
        """After apply(), the forbidden edge must not exist in the returned graph."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "forbidden"}]}
        )
        dag = self._dag_with_edges([("A", "B"), ("B", "C")])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = priors.apply(dag)

        assert not result.has_edge("A", "B"), "Forbidden edge A→B must be removed"
        assert result.has_edge("B", "C"), "Unrelated edge B→C must be preserved"

    def test_forbidden_edge_not_present_is_noop(self):
        """apply() on a graph without the forbidden edge leaves it unchanged."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "forbidden"}]}
        )
        dag = self._dag_with_edges([("C", "D")])
        result = priors.apply(dag)
        assert list(result.edges()) == [("C", "D")]

    def test_forbidden_edge_triggers_warning(self):
        """apply() must emit a UserWarning when removing a forbidden edge."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "forbidden"}]}
        )
        dag = self._dag_with_edges([("A", "B")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            priors.apply(dag)

        assert any("forbidden" in str(w.message).lower() for w in caught), (
            "A UserWarning mentioning 'forbidden' must be emitted"
        )

    # Required edge tests
    def test_required_edge_inserted_when_absent(self):
        """apply() must insert a required edge that PC omitted."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "required"}]}
        )
        dag = self._dag_with_edges([("C", "D")])
        result = priors.apply(dag)
        assert result.has_edge("A", "B"), "Required edge A→B must be inserted"

    def test_required_edge_reversed_when_wrong_direction(self):
        """apply() must flip a reversed required edge to the correct direction."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "required"}]}
        )
        dag = self._dag_with_edges([("B", "A")])  # reversed!
        result = priors.apply(dag)
        assert result.has_edge("A", "B"), "Required edge must be in correct direction"
        assert not result.has_edge("B", "A"), "Reversed edge must be removed"

    def test_required_edge_already_present_is_noop(self):
        """apply() must leave an already-correct required edge unchanged."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "required"}]}
        )
        dag = self._dag_with_edges([("A", "B"), ("C", "D")])
        result = priors.apply(dag)
        assert result.has_edge("A", "B")
        assert result.has_edge("C", "D")
        assert result.number_of_edges() == 2

    def test_input_dag_not_mutated(self):
        """apply() must return a new DiGraph and not mutate the input."""
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "forbidden"}]}
        )
        dag = self._dag_with_edges([("A", "B")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = priors.apply(dag)
        assert dag.has_edge("A", "B"), "Original DAG must not be mutated"
        assert not result.has_edge("A", "B"), "Result must not have forbidden edge"

    # warn_soft_conflicts
    def test_warn_soft_conflicts_returns_messages(self):
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "forbidden"}]}
        )
        dag = self._dag_with_edges([("A", "B")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            messages = priors.warn_soft_conflicts(dag)

        assert len(messages) == 1
        assert "A" in messages[0] and "B" in messages[0]

    def test_warn_soft_conflicts_no_conflict_returns_empty(self):
        priors = _priors_from_dict(
            {"constraints": [{"cause": "A", "effect": "B", "kind": "forbidden"}]}
        )
        dag = self._dag_with_edges([("C", "D")])
        messages = priors.warn_soft_conflicts(dag)
        assert messages == []


# ---------------------------------------------------------------------------
# WashTradeCausalDiscovery integration with priors
# ---------------------------------------------------------------------------


class TestWashTradeCausalDiscoveryWithPriors:
    def test_forbidden_edge_absent_after_fit(self):
        """After fit() with a forbidden constraint, the DAG must not contain the edge."""
        df = _make_simple_df(n=300, seed=1)
        # Inject a strong spurious correlation so PC would naturally learn the edge
        # trading_volume → account_age_days (which we declare forbidden)
        df["account_age_days"] = df["trading_volume"] * 0.8 + np.random.default_rng(1).normal(0, 5, len(df))

        priors = _priors_from_dict(
            {
                "constraints": [
                    {"cause": "trading_volume", "effect": "account_age_days", "kind": "forbidden"}
                ]
            }
        )
        priors.validate(df.columns.tolist())

        discoverer = WashTradeCausalDiscovery()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dag = discoverer.fit(df, alpha=0.05, priors=priors)

        assert not dag.has_edge("trading_volume", "account_age_days"), (
            "Forbidden edge trading_volume→account_age_days must not exist in learned DAG"
        )

    def test_required_edge_present_after_fit(self):
        """After fit() with a required constraint, the edge must appear in the DAG."""
        df = _make_simple_df(n=300, seed=2)
        # Use independent columns so PC would NOT learn the edge naturally
        priors = _priors_from_dict(
            {
                "constraints": [
                    {
                        "cause": "funding_source_similarity",
                        "effect": "round_trip_frequency",
                        "kind": "required",
                    }
                ]
            }
        )
        priors.validate(df.columns.tolist())

        discoverer = WashTradeCausalDiscovery()
        dag = discoverer.fit(df, alpha=0.05, priors=priors)

        assert dag.has_edge("funding_source_similarity", "round_trip_frequency"), (
            "Required edge funding_source_similarity→round_trip_frequency must be "
            "present in the learned DAG regardless of data evidence"
        )

    def test_unknown_variable_in_priors_raises_before_pc_runs(self):
        """Validation must happen before PC, so an unknown variable raises ValueError."""
        df = _make_simple_df(n=100, seed=3)
        priors = _priors_from_dict(
            {"constraints": [{"cause": "NONEXISTENT_FEATURE", "effect": "label", "kind": "required"}]}
        )
        discoverer = WashTradeCausalDiscovery()
        with pytest.raises(ValueError, match="NONEXISTENT_FEATURE"):
            discoverer.fit(df, alpha=0.05, priors=priors)

    def test_fit_without_priors_still_works(self):
        """Backward-compatibility: fit() without priors must not raise."""
        df = _make_simple_df(n=100, seed=4)
        discoverer = WashTradeCausalDiscovery()
        dag = discoverer.fit(df, alpha=0.05)
        assert isinstance(dag, nx.DiGraph)


# ---------------------------------------------------------------------------
# causal_priors.yaml loads and validates
# ---------------------------------------------------------------------------


class TestCausalPriorsYaml:
    def test_yaml_file_exists(self):
        import os
        assert os.path.exists("data/causal_priors.yaml"), (
            "data/causal_priors.yaml must exist"
        )

    def test_yaml_file_loads_without_error(self):
        priors = CausalPriorConstraints.load("data/causal_priors.yaml")
        assert len(priors) > 0, "causal_priors.yaml must contain at least one constraint"

    def test_yaml_file_has_both_kinds(self):
        priors = CausalPriorConstraints.load("data/causal_priors.yaml")
        assert len(priors.required) > 0, "causal_priors.yaml must have at least one required edge"
        assert len(priors.forbidden) > 0, "causal_priors.yaml must have at least one forbidden edge"

    def test_yaml_features_exist_in_synthetic_dataset(self):
        """All variables in causal_priors.yaml must be column names in the synthetic dataset."""
        from scripts.generate_synthetic_dataset import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_wallets=20, seed=0)
        feature_cols = [c for c in df.columns if c not in {"wallet", "label"}]

        priors = CausalPriorConstraints.load("data/causal_priors.yaml")
        known = set(feature_cols) | {"label"}
        # Collect all variables referenced
        all_vars: set[str] = set()
        for cause, effect in priors.required + priors.forbidden:
            all_vars.add(cause)
            all_vars.add(effect)

        unknown = all_vars - known
        assert not unknown, (
            f"causal_priors.yaml references variables not in the synthetic dataset: "
            f"{sorted(unknown)}"
        )

    def test_yaml_file_schema_rejects_injection(self, tmp_path):
        """A YAML file containing a Python-object tag must be rejected by safe_load."""
        bad_yaml = tmp_path / "bad_priors.yaml"
        bad_yaml.write_text(
            "constraints:\n"
            "  - !!python/object/apply:os.system ['echo hacked']\n"
        )
        # yaml.safe_load raises on !!python/... tags
        with pytest.raises(Exception):
            CausalPriorConstraints.load(str(bad_yaml))
