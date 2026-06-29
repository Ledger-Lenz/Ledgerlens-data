"""Plain-English labels for SHAP feature names, for use in narrative text."""

FEATURE_LABELS: dict[str, str] = {
    "benford_mad_1h": "1-hour Benford's Law deviation",
    "benford_mad_4h": "4-hour Benford's Law deviation",
    "benford_mad_24h": "24-hour Benford's Law deviation",
    "benford_mad_168h": "7-day Benford's Law deviation",
    "benford_mad_720h": "30-day Benford's Law deviation",
    "counterparty_concentration_ratio": "counterparty concentration",
    "round_trip_frequency": "round-trip trade frequency",
    "self_matching_rate": "self-matching trade rate",
    "wallet_graph_ring_size": "wash-trading ring size",
    "funding_depth": "shared funding depth",
    "trade_velocity_zscore": "trade velocity anomaly score",
}


def label_for(feature_name: str) -> str:
    """Plain-English label for a raw feature name, falling back to a
    de-slugified version of the name itself for unregistered features."""
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))
