"""Configuration module.

This package re-exports the ``config`` singleton from the top-level
``config.py`` file so that ``from config import config`` works regardless
of whether Python resolves ``config`` as this package or the ``.py`` file.
"""
import importlib.util
import os
import sys

# Load config.py from the project root explicitly, bypassing the package shadow.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("_config_module", os.path.join(_root, "config.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export the public names so `from config import config` works.
config = _mod.config
Config = _mod.Config

__all__ = ["config", "Config"]
