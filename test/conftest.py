"""Pytest fixtures for sailbench."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_sail import BasicSail
from sailbench.tf.tf_tree import TFTree2D, Transform2D
from sailbench.foils.basic_rudder import BasicRudder


@pytest.fixture
def config() -> dict[str, Any]:
    """Load the basic sailboat congifiguration for testing."""
    with Path("configs/basic_sailbot.yaml").open() as file:
        return dict(yaml.safe_load(file))


@pytest.fixture
def tf_tree() -> TFTree2D:
    """Generate a simple transform tree for testing."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world", transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
    tf.add_frame(name="keel", parent="boat", transform=Transform2D(x=0.0, y=-0.5, c=1.0, s=0.0))
    tf.add_frame(name="fluid", parent="boat", transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
    tf.add_frame(name="rudder", parent="boat", transform = Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
    return tf


@pytest.fixture
def keel(config: dict[str, Any]) -> BasicKeel:
    """Generate a BasicKeel instance for testing."""
    keel_cfg = config["keel"]
    return BasicKeel(keel_cfg)


@pytest.fixture
def sail(config: dict[str, Any]) -> BasicSail:
    """Create a basic sail with fixed wind."""
    sail_cfg = config["sail"]
    return BasicSail(sail_cfg)
@pytest.fixture
def rudder(config: dict[str, Any]) -> BasicRudder:
    """Generate a BasicRudder instance for testing."""
    rudder_cfg = config["rudder"]
    return BasicRudder(rudder_cfg)
