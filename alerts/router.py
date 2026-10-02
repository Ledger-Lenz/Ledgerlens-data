"""Configurable alert routing for detection outcomes.

``streaming/alert_dispatcher.py`` delivers every alert above the risk
threshold to a single, process-wide channel (stdout, one webhook URL, or
one websocket client) chosen at ``AlertDispatcher`` construction time.
``alerts/deduplicator.py`` groups correlated alerts from multiple
detectors into one enriched alert, but still hands that alert to whatever
single destination the caller wired up.

Neither module lets an operator say, for example, "route anything above
risk 90 to PagerDuty *and* the audit webhook, route consensus-escalated
alerts to a dedicated Slack channel, and send everything else to the
default webhook" without writing new dispatch code. This module adds that
as a declarative, rule-based routing layer that sits between alert
production (dedup / consensus escalation) and delivery (the existing
`AlertDispatcher` channels), so adding or changing a routing policy is a
config change, not a code change.

Core contract
─────────────
* :class:`RoutingRule` -- a match predicate (min risk score, detector
  names, asset-pair glob, tenant) plus a list of destination channels to
  route to when it matches.
* :class:`RouteDestination` -- one delivery target: a channel name (must
  correspond to a channel `AlertDispatcher`/a custom sender understands)
  and an opaque `target` (a webhook URL, a Slack channel id, ...).
* :class:`AlertRouter` -- evaluates an alert dict against an ordered list
  of rules and returns the union of matched destinations. Rules are
  evaluated in order; a rule may set `stop_on_match: true` to short-circuit
  evaluation of subsequent rules (useful for "route critical alerts here
  and *only* here" policies).
* :func:`load_routing_config` -- parses the YAML config shape documented
  in ``alerts/routing_config.yaml`` into a list of `RoutingRule`.

This module does not perform delivery itself -- it decides *where* an
alert should go. Actual delivery stays the responsibility of
`AlertDispatcher` (or any other sender); `AlertRouter.route()` output is
designed to be iterated and handed to per-channel senders.
"""

from __future__ import annotations

import fnmatch
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypedDict

import yaml

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Counter

    alert_routing_matches_total = Counter(
        "alert_routing_matches_total",
        "Total number of (rule, destination channel) matches produced by AlertRouter",
        ["rule_name", "channel"],
    )
    alert_routing_unmatched_total = Counter(
        "alert_routing_unmatched_total",
        "Total number of alerts that matched no routing rule and no default route",
    )
except Exception:  # pragma: no cover - prometheus optional in tests
    alert_routing_matches_total = None  # type: ignore[assignment]
    alert_routing_unmatched_total = None  # type: ignore[assignment]


class RoutingConfigError(ValueError):
    """Raised when a routing rule config is structurally invalid.

    Carries the offending rule name (or index, if unnamed) so a typo in
    ``alerts/routing_config.yaml`` fails fast with an actionable message
    at load time, rather than silently never matching at runtime.
    """

    def __init__(self, rule_identifier: str, reason: str):
        self.rule_identifier = rule_identifier
        self.reason = reason
        super().__init__(f"invalid routing rule {rule_identifier!r}: {reason}")


class Alert(TypedDict, total=False):
    """Shape of an alert dict as consumed by routing rules.

    All fields are optional (total=False) since alerts may be missing fields,
    and routing rules handle missing fields gracefully (unset rules match any alert).
    """

    wallet_address: str
    asset_pair: str
    detectors: list[str]
    risk_score: float
    tenant: str
    # Stream-level data-quality context (ingestion.data_quality.StreamQualityMonitor)
    source: str
    metric: str
    direction: str
    observed: float
    baseline: float
    magnitude: float
    detected_at: str
    message: str
    # Per-pair degradation context (issue #971). Populated by the per-pair
    # metric tracker when a single pair's model quality degrades while
    # aggregate metrics stay flat.
    metric: str
    degradation_magnitude: float


@dataclass(frozen=True)
class RouteDestination:
    """A single delivery target an alert can be routed to."""

    channel: str
    target: str
    severity_label: str = "default"

    def key(self) -> tuple[str, str]:
        return (self.channel, self.target)


@dataclass
class RoutingRule:
    """A match predicate plus the destinations to route matching alerts to.

    All match fields are optional; an unset field always matches (so a
    rule with no fields set at all matches every alert -- typically used
    as the final "catch-all" rule in a rule list).

    Attributes:
        name: human-readable identifier, used in diagnostics
            (`AlertRouter.explain`) and Prometheus labels.
        min_risk_score: alert's ``risk_score`` must be >= this value.
        detectors: alert's ``detectors`` list must intersect this set
            (i.e. at least one contributing detector matches).
        asset_pair_patterns: alert's ``asset_pair`` must match at least one
            `fnmatch`-style glob (e.g. ``"XLM:native/*"``).
        tenant: alert's ``tenant`` must equal this value exactly.
        destinations: where to route the alert when this rule matches.
        stop_on_match: if ``True``, `AlertRouter.route` stops evaluating
            subsequent rules once this one matches.
    """

    name: str
    destinations: list[RouteDestination]
    min_risk_score: float | None = None
    detectors: set[str] | None = None
    asset_pair_patterns: list[str] | None = None
    tenant: str | None = None
    stop_on_match: bool = False

    def matches(self, alert: Alert) -> bool:
        if self.min_risk_score is not None:
            if float(alert.get("risk_score", 0)) < self.min_risk_score:
                return False

        if self.detectors is not None:
            alert_detectors = set(alert.get("detectors") or [])
            if not (alert_detectors & self.detectors):
                return False

        if self.asset_pair_patterns is not None:
            asset_pair = str(alert.get("asset_pair", ""))
            if not any(fnmatch.fnmatch(asset_pair, p) for p in self.asset_pair_patterns):
                return False

        if self.tenant is not None:
            if alert.get("tenant") != self.tenant:
                return False

        return True


def _parse_destination(raw: dict[str, Any], rule_id: str) -> RouteDestination:
    try:
        return RouteDestination(
            channel=raw["channel"],
            target=raw["target"],
            severity_label=raw.get("severity_label", "default"),
        )
    except KeyError as exc:
        raise RoutingConfigError(rule_id, f"destination missing required key {exc}") from exc


def _parse_rule(raw: dict[str, Any], index: int) -> RoutingRule:
    rule_id = raw.get("name") or f"rule#{index}"

    if "destinations" not in raw or not raw["destinations"]:
        raise RoutingConfigError(rule_id, "rule must define at least one destination")

    destinations = [_parse_destination(d, rule_id) for d in raw["destinations"]]

    min_risk_score = raw.get("min_risk_score")
    if min_risk_score is not None:
        try:
            min_risk_score = float(min_risk_score)
        except (TypeError, ValueError) as exc:
            raise RoutingConfigError(rule_id, "min_risk_score must be numeric") from exc

    detectors = raw.get("detectors")
    if detectors is not None:
        if not isinstance(detectors, list):
            raise RoutingConfigError(rule_id, "detectors must be a list")
        detectors = set(detectors)

    asset_pair_patterns = raw.get("asset_pair_patterns")
    if asset_pair_patterns is not None and not isinstance(asset_pair_patterns, list):
        raise RoutingConfigError(rule_id, "asset_pair_patterns must be a list")

    return RoutingRule(
        name=rule_id,
        destinations=destinations,
        min_risk_score=min_risk_score,
        detectors=detectors,
        asset_pair_patterns=asset_pair_patterns,
        tenant=raw.get("tenant"),
        stop_on_match=bool(raw.get("stop_on_match", False)),
    )


class AlertRouter:
    """Evaluates alerts against an ordered list of routing rules.

    Thread-safe: rule lists are immutable after construction, and the
    Prometheus counters used for observability are themselves thread-safe.
    """

    def __init__(
        self,
        rules: Iterable[RoutingRule],
        default_destinations: Iterable[RouteDestination] | None = None,
    ):
        self._rules = list(rules)
        self._default_destinations = list(default_destinations or [])
        self._lock = threading.Lock()

    @property
    def rules(self) -> list[RoutingRule]:
        return list(self._rules)

    def route(self, alert: Alert) -> list[RouteDestination]:
        """Return the union of destinations for every matching rule.

        Rules are evaluated in order. If a matching rule sets
        ``stop_on_match``, evaluation stops immediately after collecting
        that rule's destinations. If no rule matches, the configured
        default destinations are returned (which may be empty).
        """
        matched: list[RouteDestination] = []
        seen: set[tuple[str, str]] = set()

        for rule in self._rules:
            if not rule.matches(alert):
                continue

            for destination in rule.destinations:
                if destination.key() in seen:
                    continue
                seen.add(destination.key())
                matched.append(destination)
                if alert_routing_matches_total is not None:
                    alert_routing_matches_total.labels(
                        rule_name=rule.name, channel=destination.channel
                    ).inc()

            if rule.stop_on_match:
                break

        if not matched:
            if alert_routing_unmatched_total is not None:
                alert_routing_unmatched_total.inc()
            return list(self._default_destinations)

        return matched

    def explain(self, alert: Alert) -> list[str]:
        """Return the names of rules that match ``alert`` (diagnostics)."""
        return [rule.name for rule in self._rules if rule.matches(alert)]


def load_routing_config(path: str) -> tuple[list[RoutingRule], list[RouteDestination]]:
    """Parse a routing config YAML file into rules and default destinations.

    Expected shape (see ``alerts/routing_config.yaml``)::

        rules:
          - name: critical
            min_risk_score: 90
            destinations:
              - channel: pagerduty
                target: https://events.pagerduty.com/...
        default_destinations:
          - channel: webhook
            target: https://example.com/alerts
    """
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise RoutingConfigError("<root>", "config root must be a mapping")

    raw_rules = raw.get("rules") or []
    if not isinstance(raw_rules, list):
        raise RoutingConfigError("<root>", "'rules' must be a list")

    rules = [_parse_rule(rule, index) for index, rule in enumerate(raw_rules)]

    raw_defaults = raw.get("default_destinations") or []
    if not isinstance(raw_defaults, list):
        raise RoutingConfigError("<root>", "'default_destinations' must be a list")

    defaults = [_parse_destination(d, "<default>") for d in raw_defaults]
    return rules, defaults


def build_per_pair_degradation_alert(
    asset_pair: str,
    metric: str,
    degradation_magnitude: float,
    *,
    detector: str = "per_pair_cusum",
    risk_score: float = 0.0,
    tenant: str | None = None,
) -> Alert:
    """Build a routing-ready alert for a single pair's model degradation.

    Issue #971: per-pair metric tracking detects when one pair's model
    quality degrades while aggregate metrics stay flat. The resulting
    alert must carry pair-level context -- the pair identifier, the
    affected metric, and the magnitude of the degradation -- so routing
    rules can target it (e.g. via ``asset_pair_patterns``) and downstream
    consumers can act on it.
    """
    alert: Alert = {
        "asset_pair": asset_pair,
        "detectors": [detector],
        "risk_score": float(risk_score),
        "metric": metric,
        "degradation_magnitude": float(degradation_magnitude),
    }
    if tenant is not None:
        alert["tenant"] = tenant
    return alert
