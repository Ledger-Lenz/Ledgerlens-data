"""Configuration module.

Re-exports the ``Config`` class and ``config`` singleton from the top-level
``config.py`` module so that ``from config import config`` works regardless
of whether Python resolves the ``config`` name to this package or the module.
"""

import importlib.util
import os
import sys


def _load_root_config():
    """Load the root-level config.py without triggering the package import."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, "config.py")
    spec = importlib.util.spec_from_file_location("_root_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_root_config_module = _load_root_config()
Config = _root_config_module.Config
config = _root_config_module.config

__all__ = ["Config", "config"]
