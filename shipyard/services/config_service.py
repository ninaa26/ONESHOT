from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

CONFIG_PATH = Path("boat.yaml")
TEMPLATE_PATH = Path("configs/shipyard_template.yaml")


def save_config(config: dict[str, Any]) -> None:
    """Persist the current boat configuration to disk."""
    with CONFIG_PATH.open("w") as file:
        yaml.dump(config, file, sort_keys=False)


def load_config() -> dict[str, Any]:
    """Load a previously saved boat configuration, if it exists."""
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open() as file:
        loaded: dict[str, Any] | None = yaml.safe_load(file)
        return loaded or {}


def load_template_config() -> dict[str, Any]:
    """Load the baseline template configuration used by Shipyard."""
    if not TEMPLATE_PATH.exists():
        return {}
    with TEMPLATE_PATH.open() as file:
        template: dict[str, Any] | None = yaml.safe_load(file)
        return template or {}
