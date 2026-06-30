"""Configuration module.

Re-exports ``Config`` and ``config`` from the project-root ``config.py`` so
that ``from config import config`` and ``from config import Config`` continue
to work even though a ``config/`` sub-package also exists at the project root.
"""

import importlib.util
import os as _os

_root_config_path = _os.path.join(_os.path.dirname(__file__), "..", "config.py")
_spec = importlib.util.spec_from_file_location("_config_root", _root_config_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

Config = _mod.Config
config = _mod.config
