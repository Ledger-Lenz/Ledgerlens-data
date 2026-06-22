"""Persist differential privacy training metrics to models/metrics.json."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from config import config


def record_dp_metrics(
    model_dir: str | None,
    component_name: str,
    metrics: dict[str, Any],
) -> str:
    """Merge DP metrics for *component_name* into ``metrics.json``.

    Creates the file if absent; preserves existing ensemble / GNN entries.
    """
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "metrics.json")

    payload: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as handle:
            payload = json.load(handle)

    dp_section = payload.setdefault("differential_privacy", {})
    dp_section[component_name] = {
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **metrics,
    }
    payload["differential_privacy"] = dp_section

    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

    return path
