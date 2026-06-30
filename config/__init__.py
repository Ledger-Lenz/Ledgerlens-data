"""Configuration module.

Re-exports the ``config`` singleton and ``Config`` class from the top-level
``config.py`` so that ``from config import config`` resolves correctly
regardless of whether Python resolves ``config`` as this package or as the
sibling ``config.py`` module.

The ``config/`` directory exists for tenant-specific YAML files
(``config/tenants.yaml``) and the ``tenant_config.py`` helper; all global
environment-driven configuration lives in the top-level ``config.py``.
"""

import importlib
import os
import sys
from pathlib import Path

# When Python resolves 'config' it picks this package (config/) over the
# sibling config.py because packages take precedence.  We load config.py
# explicitly by file path so that ``from config import config`` works the same
# way everywhere in the codebase.
_root = Path(__file__).resolve().parent.parent
_config_py = str(_root / "config.py")

_spec = importlib.util.spec_from_file_location("_config_module", _config_py)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Re-export the names the rest of the codebase imports.
Config = _mod.Config
config = _mod.config

__all__ = ["Config", "config"]
