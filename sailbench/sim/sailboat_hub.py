"""Compose a simulated sailboat."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from sailbench.foils.basic_keel import BasicKeel
from sailbench.models.model import State

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

    def step(self, state: State, dt: float, solver: Callable) -> State:
        """Sail the boat."""

        def dynamics(state: State) -> np.ndarray:
            # --- forces from all components ---
            fx, fy, mz = self._forces(state)

            m = self.boat_cfg["mass"]
            iz = self.boat_cfg["inertia_z"]

            c = state.psi[0]  # Heading cosine term
            s = state.psi[1]  # Heading sine term

            # --- body-frame accelerations ---
            du = fx / m + state.r * state.v
            dv = fy / m - state.r * state.u
            dr = mz / iz

            # --- world-frame position rates ---
            dx = state.u * c - state.v * s
            dy = state.u * s + state.v * c

            # --- heading representation rates ---
            dc = -state.r * s
            ds = state.r * c

            return np.array([dx, dy, dc, ds, du, dv, dr])

        # --- integrate ---
        next_arr = solver(dynamics, state.to_array(), dt)

        # --- rebuild state ---
        return State.from_array(next_arr)

    # --- Physics core ----------------------------------------
    def _forces(self, state: State) -> tuple[float, float, float]:
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
