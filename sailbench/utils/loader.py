"""Load configuration from YAML files."""

import math
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML config file, with environment wind_dir converted to radians."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Convert wind_dir_deg to wind_dir (radians) in environment
    env = cfg.setdefault("environment", {})
    if "wind_dir_deg" in env and "wind_dir" not in env:
        env["wind_dir"] = math.radians(float(env["wind_dir_deg"]))

    return cfg
