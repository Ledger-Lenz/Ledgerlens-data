"""Configuration module.

Re-exports ``config`` and ``Config`` from the root ``config.py`` so that
``from config import config`` resolves correctly even though a ``config/``
package directory exists alongside the module file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_config_root", Path(__file__).parent.parent / "config.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

config = _mod.config
Config = _mod.Config

__all__ = ["config", "Config"]
