"""Pytest fixtures for sailbench."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_rudder import BasicRudder


@pytest.fixture
def config() -> dict[str, Any]:
    """Load the basic sailboat congifiguration for testing."""
    with Path("configs/basic_sailbot.yaml").open() as file:
        return dict(yaml.safe_load(file))


@pytest.fixture
def keel(config: dict[str, Any]) -> BasicKeel:
    """Generate a BasicKeel instance for testing."""
    keel_cfg = config["keel"]
    return BasicKeel(keel_cfg)

@pytest.fixture
def rudder(config: dict[str, Any]) -> BasicRudder:
    """Generate a BasicRudder instance for testing."""
    rudder_cfg = config["rudder"]
    return BasicRudder(rudder_cfg)