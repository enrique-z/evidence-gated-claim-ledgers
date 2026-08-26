"""
Config loader for Cost Meter.

Reads configuration from a TOML file. Config schema:
  [budget]
  cap = 300.0  # Hard budget cap in USD

  [storage]
  ledger_path = "~/.cache/neurips-26/cost_meter.jsonl"  # Append-only ledger
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ImportError:
    # Python < 3.11
    import tomli as tomllib  # type: ignore


DEFAULT_CONFIG = {
    "budget": {"cap": 300.0},
    "storage": {"ledger_path": "~/.cache/neurips-26/cost_meter.jsonl"},
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to config.toml. If None, uses the default config path
                    relative to this module.

    Returns:
        Configuration dictionary with keys: 'budget', 'storage'.

    Raises:
        FileNotFoundError: If the specified config_path does not exist.
    """
    if config_path is None:
        # Use default config file in the same directory as this module
        config_path = Path(__file__).parent / "config.toml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Merge with defaults to ensure all required keys exist
    merged = dict(DEFAULT_CONFIG)
    if "budget" in config:
        merged["budget"].update(config["budget"])
    if "storage" in config:
        merged["storage"].update(config["storage"])

    return merged
