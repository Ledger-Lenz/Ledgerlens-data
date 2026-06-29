"""Configuration module."""
# Re-export from parent config.py to handle module shadowing
import sys
import importlib.util

# Load config.py explicitly to avoid name collision with config/ directory
spec = importlib.util.spec_from_file_location("_config_module", "/workspaces/Ledgerlens-data/config.py")
_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_config_module)

Config = _config_module.Config
config = _config_module.config

__all__ = ["Config", "config"]
