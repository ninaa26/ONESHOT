"""Compose a simulated sailboat."""

from pathlib import Path

import numpy as np
import yaml

from sailbench.foils.basic_keel import BasicKeel

CONFIG_PATH = "configs/"


class SailboatHub:
    """A hub to manage sailboat simulation components."""

    def __init__(self, config_file: str) -> None:
        """Initialize the SailboatHub with configuration from a YAML file."""
        with Path(CONFIG_PATH + config_file).open() as file:
            cfg = yaml.safe_load(file)

        self.simulation_cfg = cfg["simulation"]
        self.boat_cfg = cfg["boat"]
        self.keel_cfg = cfg["keel"]
        self.rudder_cfg = cfg["rudder"]
        self.sail_cfg = cfg["sail"]

        self.boat_factory()

    def boat_factory(self) -> None:
        """Instantiate boat components from configs."""
        self.keel = BasicKeel(self.keel_cfg)
        self.components = [self.keel]

    # --- Physics core ----------------------------------------
    def _forces(self, state: np.ndarray) -> tuple[float, float, float]:
        """Compute total body-frame forces and yaw moment."""
        fx_total = 0.0
        fy_total = 0.0
        mz_total = 0.0

        for component in self.components:
            fx, fy = component.compute(state)

            # Sum forces
            fx_total += fx
            fy_total += fy

            # Moment about CG (2D cross product)
            mz = component.p["x_pos"] * fy - component.p["y_pos"] * fx
            mz_total += mz

        return fx_total, fy_total, mz_total
