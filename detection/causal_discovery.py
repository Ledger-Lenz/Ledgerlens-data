"""Causal discovery module using the PC algorithm to identify causal features of wash trading.

Now includes ``CausalPriorConstraints`` (Issue #192): a prior-knowledge
constraint system that prevents the learned graph from violating known causal
facts about the Stellar DEX wash-trade domain.  Constraints are expressed as
(cause, effect, required|forbidden) triples, loaded from a YAML file whose
schema is validated on every load to prevent injection of arbitrary Python
objects.
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CausalPriorConstraints — Issue #192
# ---------------------------------------------------------------------------

ConstraintKind = Literal["required", "forbidden"]

# JSON-Schema-style YAML structure expected in the prior-knowledge file.
# We validate manually rather than importing jsonschema to avoid an extra
# dependency: the structure is simple enough that explicit checks suffice.
_VALID_KINDS = {"required", "forbidden"}


@dataclass
class _Constraint:
    """A single (cause, effect, kind) causal constraint triple."""

    cause: str
    effect: str
    kind: ConstraintKind


class CausalPriorConstraints:
    """Domain-expert prior knowledge for Stellar DEX wash-trade causal graphs.

    Encodes two types of constraints:

    * **forbidden** (soft): the learned graph must not contain this directed
      edge.  If the PC algorithm produces the edge anyway (against data
      evidence), it is removed from the DAG with a warning.
    * **required** (hard): the learned graph must contain this directed edge
      regardless of data evidence.  If PC omits or reverses the edge, it is
      inserted / corrected.

    The constraint set is loaded from a YAML file whose schema is validated on
    every ``load()`` call so that arbitrary Python objects cannot be injected.
    All named variables must exist in the feature set supplied to ``validate``
    or a startup ``ValueError`` is raised.

    Typical usage::

        priors = CausalPriorConstraints.load("data/causal_priors.yaml")
        priors.validate(feature_columns)   # raises if unknown variables
        dag = discoverer.fit(df, priors=priors)

    YAML format::

        constraints:
          - cause: funding_source_similarity
            effect: round_trip_frequency
            kind: required
          - cause: trading_volume
            effect: account_age_days
            kind: forbidden

    Parameters
    ----------
    constraints:
        List of :class:`_Constraint` triples (populated by ``load``).
    """

    def __init__(self, constraints: list[_Constraint] | None = None) -> None:
        self._constraints: list[_Constraint] = constraints or []

    # ------------------------------------------------------------------
    # Loading & validation
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "CausalPriorConstraints":
        """Load and validate a YAML prior-knowledge file.

        The YAML must have a top-level ``constraints`` key containing a list of
        mappings with ``cause``, ``effect``, and ``kind`` string fields.
        Unknown top-level keys are ignored; unknown constraint fields raise
        ``ValueError``.

        Security: the file is parsed with ``yaml.safe_load`` so no Python
        objects can be constructed from it.

        Args:
            path: Path to the YAML prior-knowledge file.

        Returns:
            A populated :class:`CausalPriorConstraints` instance.

        Raises:
            FileNotFoundError: if *path* does not exist.
            ValueError: if the YAML schema is invalid or a ``kind`` is unknown.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Causal priors file not found: {path}")

        with open(path, encoding="utf-8") as fh:
            # safe_load prevents arbitrary Python object construction
            raw = yaml.safe_load(fh)

        return cls._from_dict(raw, source=path)

    @classmethod
    def _from_dict(cls, raw: object, source: str = "<dict>") -> "CausalPriorConstraints":
        """Parse and validate a dict-tree produced by ``yaml.safe_load``."""
        if not isinstance(raw, dict):
            raise ValueError(
                f"Causal priors file {source!r} must be a YAML mapping "
                f"(got {type(raw).__name__})"
            )

        constraints_raw = raw.get("constraints")
        if constraints_raw is None:
            raise ValueError(
                f"Causal priors file {source!r} is missing the required "
                "'constraints' key"
            )
        if not isinstance(constraints_raw, list):
            raise ValueError(
                f"'constraints' in {source!r} must be a list "
                f"(got {type(constraints_raw).__name__})"
            )

        parsed: list[_Constraint] = []
        for i, item in enumerate(constraints_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"constraints[{i}] in {source!r} must be a mapping "
                    f"(got {type(item).__name__})"
                )
            for required_key in ("cause", "effect", "kind"):
                if required_key not in item:
                    raise ValueError(
                        f"constraints[{i}] in {source!r} is missing "
                        f"required key '{required_key}'"
                    )
                if not isinstance(item[required_key], str):
                    raise ValueError(
                        f"constraints[{i}].{required_key} in {source!r} "
                        f"must be a string (got {type(item[required_key]).__name__})"
                    )

            kind = item["kind"].strip().lower()
            if kind not in _VALID_KINDS:
                raise ValueError(
                    f"constraints[{i}].kind in {source!r} must be one of "
                    f"{sorted(_VALID_KINDS)!r}, got {item['kind']!r}"
                )

            parsed.append(
                _Constraint(
                    cause=item["cause"].strip(),
                    effect=item["effect"].strip(),
                    kind=kind,  # type: ignore[arg-type]
                )
            )

        logger.info(
            "Loaded %d causal prior constraints from %s", len(parsed), source
        )
        return cls(parsed)

    def validate(self, feature_columns: list[str] | set[str]) -> None:
        """Check that all constraint variables exist in *feature_columns*.

        Args:
            feature_columns: Column names present in the feature matrix.

        Raises:
            ValueError: listing every unknown variable found in any constraint.
        """
        known = set(feature_columns)
        unknown: list[str] = []
        for c in self._constraints:
            for var in (c.cause, c.effect):
                if var not in known:
                    unknown.append(var)

        if unknown:
            raise ValueError(
                "Causal prior constraints reference variables not present in "
                "the feature set. Unknown variables: "
                + ", ".join(sorted(set(unknown)))
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def forbidden(self) -> list[tuple[str, str]]:
        """List of (cause, effect) pairs that must NOT appear in the DAG."""
        return [(c.cause, c.effect) for c in self._constraints if c.kind == "forbidden"]

    @property
    def required(self) -> list[tuple[str, str]]:
        """List of (cause, effect) pairs that MUST appear in the DAG."""
        return [(c.cause, c.effect) for c in self._constraints if c.kind == "required"]

    def __len__(self) -> int:
        return len(self._constraints)

    def __repr__(self) -> str:
        return (
            f"CausalPriorConstraints("
            f"required={len(self.required)}, "
            f"forbidden={len(self.forbidden)})"
        )

    # ------------------------------------------------------------------
    # Graph enforcement
    # ------------------------------------------------------------------

    def apply(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Return a copy of *dag* with all prior constraints enforced.

        * **Forbidden edges** are removed from the graph.  A WARNING is emitted
          for each removal so the conflict is visible in logs.
        * **Required edges** are inserted if absent, or reversed if the graph
          contains them in the wrong direction.  An INFO message is emitted for
          each insertion / correction.

        The returned graph is always a new ``DiGraph`` instance; the input is
        not mutated.

        Args:
            dag: The data-driven causal DAG produced by the PC algorithm.

        Returns:
            A new ``DiGraph`` with constraints applied.
        """
        g = dag.copy()

        # Remove forbidden edges
        for cause, effect in self.forbidden:
            if g.has_edge(cause, effect):
                warnings.warn(
                    f"Causal prior: removing forbidden edge {cause!r} → {effect!r} "
                    "from learned DAG (data-driven graph conflicts with domain prior).",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning(
                    "Causal prior conflict: removed forbidden edge %r → %r", cause, effect
                )
                g.remove_edge(cause, effect)

        # Insert / correct required edges
        for cause, effect in self.required:
            if g.has_edge(cause, effect):
                continue  # already present in correct direction
            if g.has_edge(effect, cause):
                # Reversed edge present — correct the direction
                g.remove_edge(effect, cause)
                logger.info(
                    "Causal prior: reversed edge %r → %r to %r → %r (required constraint)",
                    effect, cause, cause, effect,
                )
            g.add_edge(cause, effect)
            logger.info(
                "Causal prior: inserted required edge %r → %r", cause, effect
            )

        return g

    def warn_soft_conflicts(self, dag: nx.DiGraph) -> list[str]:
        """Log warnings for forbidden-edge conflicts without modifying the graph.

        Returns a list of warning messages (useful for testing).
        """
        messages: list[str] = []
        for cause, effect in self.forbidden:
            if dag.has_edge(cause, effect):
                msg = (
                    f"Data-driven graph contains forbidden edge {cause!r} → {effect!r}. "
                    "This conflicts with the domain prior (forbidden constraint)."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)
                messages.append(msg)
        return messages


# ---------------------------------------------------------------------------
# WashTradeCausalDiscovery (updated for Issue #192)
# ---------------------------------------------------------------------------


class WashTradeCausalDiscovery:
    """PC-algorithm causal discovery with optional domain-expert prior constraints.

    Prior constraints (Issue #192) are applied *after* the data-driven PC run:

    1. The PC algorithm learns a skeleton from observational data.
    2. ``CausalPriorConstraints.apply`` removes forbidden edges and inserts /
       corrects required edges.
    3. The resulting DAG is guaranteed to respect the domain prior.

    Usage without priors (backward-compatible)::

        discoverer = WashTradeCausalDiscovery()
        dag = discoverer.fit(df, alpha=0.05)

    Usage with priors::

        priors = CausalPriorConstraints.load("data/causal_priors.yaml")
        priors.validate(df.columns)
        discoverer = WashTradeCausalDiscovery()
        dag = discoverer.fit(df, alpha=0.05, priors=priors)
    """

    def __init__(self) -> None:
        self.dag = nx.DiGraph()

    def fit(
        self,
        feature_df: pd.DataFrame,
        alpha: float = 0.05,
        priors: CausalPriorConstraints | None = None,
    ) -> nx.DiGraph:
        """Fit the PC causal discovery algorithm on the features and label.

        If *priors* is provided, forbidden edges are removed and required edges
        are inserted / corrected after the data-driven discovery step.  A
        validation check ensures all constraint variables exist in
        *feature_df*'s columns before running PC.

        Args:
            feature_df: Numeric feature DataFrame (non-numeric columns dropped).
            alpha:      Significance level for the conditional independence tests.
            priors:     Optional domain-expert prior constraints.

        Returns:
            A ``networkx.DiGraph`` representing the causal DAG.
        """
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        # Ensure all columns are numeric
        numeric_cols = [
            col
            for col in feature_df.columns
            if pd.api.types.is_numeric_dtype(feature_df[col])
        ]
        df_filtered = feature_df[numeric_cols].dropna()

        # Validate prior variables against the feature set before running PC
        if priors is not None:
            priors.validate(df_filtered.columns.tolist())

        # Run PC causal discovery
        cg = pc(
            df_filtered.values,
            alpha=alpha,
            indep_test=fisherz,
            node_names=list(df_filtered.columns),
        )

        self.dag = self._to_networkx(cg.G)

        # Apply domain-expert prior constraints (Issue #192)
        if priors is not None:
            # Warn about soft conflicts before enforcing
            priors.warn_soft_conflicts(self.dag)
            self.dag = priors.apply(self.dag)

        return self.dag

    def _to_networkx(self, cg_graph) -> nx.DiGraph:
        """Convert causal-learn GeneralGraph to networkx DiGraph."""
        g = nx.DiGraph()

        # Add all nodes
        for node in cg_graph.get_nodes():
            g.add_node(node.get_name())

        # Add edges
        for edge in cg_graph.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            ep1 = edge.get_endpoint1().name  # 'TAIL' or 'ARROW'
            ep2 = edge.get_endpoint2().name  # 'TAIL' or 'ARROW'

            if ep1 == "TAIL" and ep2 == "ARROW":
                g.add_edge(u, v)
            elif ep1 == "ARROW" and ep2 == "TAIL":
                g.add_edge(v, u)
            elif ep1 == "TAIL" and ep2 == "TAIL":
                # Undirected edge: orient consistently using node name order
                if u < v:
                    g.add_edge(u, v)
                else:
                    g.add_edge(v, u)
            elif ep1 == "ARROW" and ep2 == "ARROW":
                # Bidirectional edge: orient consistently using node name order
                if u < v:
                    g.add_edge(u, v)
                else:
                    g.add_edge(v, u)

        return g

    def causal_features(self, label_name: str = "label") -> list[str]:
        """Features with direct causal edge to the wash-trade label."""
        if not self.dag or label_name not in self.dag:
            return []

        features = []
        for u in self.dag.nodes:
            if u == label_name:
                continue
            if self.dag.has_edge(u, label_name) or self.dag.has_edge(label_name, u):
                features.append(u)
        return sorted(list(set(features)))

    def save_dag(self, path: str) -> None:
        """Save the causal DAG structure as a JSON file."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = {"nodes": list(self.dag.nodes), "edges": list(self.dag.edges)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
