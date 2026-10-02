"""Operator CLI to pin / release the RL alert threshold (Issue #898).

Writes the override file watched by ``ThresholdController(override_path=...)``
(default ``$RL_THRESHOLD_OVERRIDE_PATH`` or ``data/threshold_override.json``).
A pin takes precedence over the RL policy until explicitly released.

Usage::

    python scripts/threshold_override.py pin 80 [--asset XLM-USDC]
    python scripts/threshold_override.py release [--asset XLM-USDC]
    python scripts/threshold_override.py show
"""

from __future__ import annotations

import argparse
import json
import os

from streaming.rl_threshold_controller import DEFAULT_OVERRIDE_PATH


def _load(path: str) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pin or release the RL alert threshold.")
    p.add_argument("command", choices=["pin", "release", "show"])
    p.add_argument("value", nargs="?", type=float)
    p.add_argument("--asset", default="*", help="asset pair, or * for all (default)")
    p.add_argument("--path", default=DEFAULT_OVERRIDE_PATH)
    args = p.parse_args(argv)

    data = _load(args.path)
    if args.command == "show":
        print(json.dumps(data, indent=2))
        return 0
    if args.command == "pin":
        if args.value is None:
            p.error("pin requires a threshold value")
        data[args.asset] = args.value
    else:
        data.pop(args.asset, None)
    os.makedirs(os.path.dirname(args.path) or ".", exist_ok=True)
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
