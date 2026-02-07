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

        def dynamics(arr: np.ndarray) -> np.ndarray:
            """State derivative; arr = [x, y, c, s, u, v, r]."""
            state_vec = State.from_array(arr)
            fx, fy, mz = self._forces(state_vec)

            m = self.boat_cfg.get("mass", self.boat_cfg.get("m", 27.0))
            iz = self.boat_cfg.get("inertia_z", self.boat_cfg.get("Iz", 10.0))

            c, s = arr[2], arr[3]  # Heading cosine, sine
            u, v, r = arr[4], arr[5], arr[6]

            # --- body-frame accelerations ---
            du = fx / m + r * v
            dv = fy / m - r * u
            dr = mz / iz

            # --- world-frame position rates ---
            dx = u * c - v * s
            dy = u * s + v * c

            # --- heading representation rates ---
            dc = -r * s
            ds = r * c

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

            # Moment about CG (2D cross product; x_pos = arm along boat, y_pos = lateral offset)
            x_pos = component.p.get("x_pos", component.p.get("x_k", component.p.get("x_r", 0.0)))
            y_pos = component.p.get("y_pos", 0.0)
            mz = x_pos * fy - y_pos * fx
            mz_total += mz

        return fx_total, fy_total, mz_total
