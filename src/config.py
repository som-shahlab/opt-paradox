# src/config.py
# =============================================================================
# Configuration loader for ClinAgents
# 
# This module reads the `config.yaml` file located in the project root,
# parses its contents, and converts the nested dictionary structure into
# a Python object with attribute‐style access. All configuration values
# can then be accessed via the global `CONFIG` constant, e.g.:
#
#     from config import CONFIG
#     print(CONFIG.api_keys.openai)
#     print(CONFIG.endpoints.gpt.model_id)
#
# Using this loader ensures that configuration is centralized in YAML
# (human-editable) while your Python code works with a clean, namespaced
# object without repeating parsing logic.
# =============================================================================


import yaml
from pathlib import Path
from types import SimpleNamespace

# Path to the YAML configuration file
PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH  = PROJECT_ROOT / "config.yaml"


def _to_namespace(data):
    """
    Recursively convert nested dicts into SimpleNamespace objects.
    """
    if isinstance(data, dict):
        return SimpleNamespace(**{
            key: _to_namespace(value) for key, value in data.items()
        })
    return data


# Load and parse the YAML file once at import time
with open(_CONFIG_PATH, "r") as f:
    _raw_config = yaml.safe_load(f)

# Expose the configuration as a constant namespace
CONFIG = _to_namespace(_raw_config)
