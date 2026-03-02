"""Compose a simulated sailboat."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from sailbench.dynamics.linear_hydro import LinearHydroModel
from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_rudder import BasicRudder
from sailbench.foils.basic_sail import BasicSail
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

CONFIG_PATH = "configs/"


class SailboatHub:
    """A hub to manage sailboat simulation components."""

    def __init__(self, config_file: str) -> None:
        """Initialize the SailboatHub with configuration from a YAML file."""
        with Path(CONFIG_PATH + config_file).open() as file:
            cfg = yaml.safe_load(file)

        self.simulation_cfg = cfg["simulation"]
        self.boat_cfg = cfg["boat"]
        self.hull_cfg = cfg["hull"]
        self.keel_cfg = cfg["keel"]
        self.rudder_cfg = cfg["rudder"]
        self.sail_cfg = cfg["sail"]

        self.tf = TFTree2D()

        self.boat_factory()

    def boat_factory(self) -> None:
        """Instantiate boat components from configs."""
       # self.keel = BasicKeel(self.keel_cfg)
        self.sail = BasicSail(self.sail_cfg)
        self.rudder = BasicRudder(self.rudder_cfg)
        self.hull = LinearHydroModel(self.hull_cfg)
        self.keel = BasicKeel(self.keel_cfg)
        self.components = [self.rudder,self.keel, self.hull, self.sail]  # order matters for force summation (e.g. keel before sail)
        self.m = self.boat_cfg.get("mass", self.boat_cfg.get("m", 27.0))
        self.iz = self.boat_cfg.get("inertia_z", self.boat_cfg.get("Iz", 25.0))

        # TODO: Change starting position and heading from config
        self.tf.add_frame(
            name="boat",
            parent="world",
            transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0),  # boat frame starts aligned with world frame
        )

        # How the boat is traveling through the water (local track)
        self.tf.add_frame(
            name="fluid",
            parent="boat",
            transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0),  # fluid frame starts aligned with boat frame
        )

        # Instantiate component frames in tf tree
        # Keel is fixed
        self.tf.add_frame(
            name="keel",
            parent="boat",
            transform=Transform2D(
                x=self.keel_cfg.get("x_pos", 0.0),
                y=self.keel_cfg.get("y_pos", 0.0),
                c=1,  # Keel is inline with boat axis
                s=0,
            ),
        )

        # Set up rudder frame
        self.tf.add_frame(
            name="rudder",
            parent="boat",
            transform=Transform2D(
                x=self.rudder_cfg.get("x_pos", 0.0),
                y=self.rudder_cfg.get("y_pos", 0.0),
                c=1,
                s=0,
            ),
        )

        # Sail is updated per step
        self.tf.add_frame(
            name="sail",
            parent="boat",
            transform=Transform2D(x=self.sail_cfg.get("x_pos", 0.0), y=self.sail_cfg.get("y_pos", 0.0), c=1.0, s=0.0),
        )

    def step(
        self,
        state: State,
        dt: float,
        solver: Callable,
        sail_angle: float = 0.0,
        rudder_angle: float = 0.0,
    ) -> State:
        """Sail the boat."""

        def dynamics(arr: np.ndarray) -> np.ndarray:
            """State derivative; arr = [x, y, c, s, u, v, r]."""
            state_vec = State.from_array(arr)
            fx, fy, mz = self._forces(state_vec)

            c, s = arr[2], arr[3]  # Heading cosine, sine
            u, v, r = arr[4], arr[5], arr[6]

            # --- body-frame accelerations ---
            du = fx / self.m + r * v
            dv = fy / self.m - r * u
            dr = mz / self.iz

            # --- world-frame position rates (transform body velocity to world) ---
            dx = u * c - v * s
            dy = u * s + v * c

            # --- heading representation rates ---
            dc = -r * s
            ds = r * c

            return np.array([dx, dy, dc, ds, du, dv, dr])

        # --- integrate ---
        next_arr = solver(dynamics, state.to_array(), dt)

        # --- update tf tree with dynamic components ---
        self.tf.add_frame(
            name="boat",
            parent="world",
            transform=Transform2D(x=next_arr[0], y=next_arr[1], c=next_arr[2], s=next_arr[3]),
        )

        # update sail frame with new sail angle
        self.tf.add_frame(
            name="sail",
            parent="boat",
            transform=Transform2D(
                x=self.sail_cfg.get("x_pos", 0.0),
                y=self.sail_cfg.get("y_pos", 0.0),
                c=np.cos(sail_angle),
                s=np.sin(sail_angle),
            ),
        )

        # (u, v) are body-frame; flow direction is opposite to velocity
        u, v = next_arr[4], next_arr[5]
        norm = np.hypot(u, v)
        if norm > 1e-6:
            c, s = -u / norm, -v / norm
        else:
            c, s = 1.0, 0.0
        self.tf.add_frame(
            name="fluid",
            parent="boat",
            transform=Transform2D(x=0.0, y=0.0, c=c, s=s),
        )

        self.tf.add_frame(
            name="rudder",
            parent="boat",
            transform=Transform2D(
                x=self.rudder_cfg.get("x_pos", 0.0),
                y=self.rudder_cfg.get("y_pos", 0.0),
                c=np.cos(np.radians(rudder_angle)),
                s=np.sin(np.radians(rudder_angle)),
            ),
        )

        # --- rebuild state ---
        return State.from_array(next_arr)

    # --- Physics core ----------------------------------------
    def _forces(self, state: State) -> tuple[float, float, float]:
        """Compute total body-frame forces and yaw moment."""
        fx_total = 0.0
        fy_total = 0.0
        mz_total = 0.0

        for component in self.components:
            result = np.atleast_1d(component.compute(state, self.tf))
            fx, fy = float(result[0]), float(result[1])
            mz_direct = float(result[2]) if len(result) > 2 else 0.0

            # Moment about CG (2D cross product; x_pos = arm along boat, y_pos = lateral offset)
            x_pos = component.p.get("x_pos", 0.0)
            y_pos = component.p.get("y_pos", 0.0)
            mz = x_pos * fy - y_pos * fx + mz_direct

            # Sum forces
            fx_total += fx
            fy_total += fy
            mz_total += mz

        return fx_total, fy_total, mz_total
