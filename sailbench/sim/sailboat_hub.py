"""Compose a simulated sailboat."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.dynamics.displacement_hull import DisplacementHull
from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_rudder import BasicRudder
from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.orc_sail import ORCSail
from sailbench.foils.thin_foil import ThinFoil
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
        self.physics = str(cfg.get("physics", "legacy")).lower()

        self.tf = TFTree2D()
        # Last computed sail force in boat frame (Fx, Fy) for diagnostics / UI.
        self.last_sail_force: tuple[float, float] = (0.0, 0.0)
        # Last resolved sail angle in radians after sheet-limit + wind logic.
        self.last_sail_angle_rad: float = 0.0
        # Per-component forces (boat frame) for visualization.
        self.last_forces: dict[str, tuple[float, float]] = {}

        # Rudder "servo" state (for manual controllability).
        self._rudder_angle_deg: float = 0.0

        self.boat_factory()

    def boat_factory(self) -> None:
        """Instantiate boat components from configs.

        ``physics: v2`` in the config selects the ORC-based sail, finite-span
        appendages and displacement hull, plus added mass. Anything else keeps
        the legacy NeuralFoil models so existing configs and trained policies
        are untouched.
        """
        self.m = self.boat_cfg.get("mass", self.boat_cfg.get("m", 27.0))
        self.iz = self.boat_cfg.get("inertia_z", self.boat_cfg.get("Iz", 25.0))

        if self.physics == "v2":
            self.hull_cfg.setdefault("mass", self.m)
            self.sail = ORCSail(self.sail_cfg)
            self.rudder = ThinFoil(self.rudder_cfg, frame="rudder")
            self.hull = DisplacementHull(self.hull_cfg)
            self.keel = ThinFoil(self.keel_cfg, frame="keel")
            m_surge, m_sway, i_yaw = self.hull.added_mass()
        else:
            self.sail = BasicSail(self.sail_cfg)
            self.rudder = BasicRudder(self.rudder_cfg)
            self.hull = BasicHullModel(self.hull_cfg)
            self.keel = BasicKeel(self.keel_cfg)
            m_surge = m_sway = i_yaw = 0.0

        self.components = [self.hull, self.keel, self.sail, self.rudder]

        # Effective (rigid + added) mass. Added mass in sway and yaw is the
        # same order as the rigid-body values for a slender hull; omitting it
        # makes the boat slide and turn several times too fast.
        self.m_surge = self.m + m_surge
        self.m_sway = self.m + m_sway
        self.iz_eff = self.iz + i_yaw

        # TODO: Change starting position and heading from config
        self.tf.add_frame(
            name="boat",
            parent="world",
            transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0),  # boat frame starts aligned with world frame
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
        """Sail the boat.

        `sail_angle` is treated as sheet limit (max |sail angle| from centerline),
        not as a rigid commanded sail angle.
        """

        sheet_limit_rad = float(np.abs(sail_angle))

        # The rudder servo is discrete-time actuator state, not part of the
        # continuous dynamics, so it advances exactly once per step.
        self._advance_rudder_servo(rudder_angle_deg=rudder_angle, dt=dt)

        def dynamics(arr: np.ndarray) -> np.ndarray:
            """State derivative; arr = [x, y, c, s, u, v, r]."""
            state_vec = State.from_array(arr)

            # Re-resolve the boat/fluid/sail frames at *this* stage's state.
            # Without this the force model reads a tf tree left over from the
            # previous step, which silently drops the integrator to 1st order.
            self._set_kinematic_frames(state_vec, sheet_limit_rad)

            fx, fy, mz = self._forces(state_vec)

            c, s = arr[2], arr[3]  # Heading cosine, sine
            u, v, r = arr[4], arr[5], arr[6]

            # --- body-frame accelerations (Fossen rigid-body + added mass) ---
            # With no added mass these reduce exactly to the previous form.
            du = (fx + self.m_sway * r * v) / self.m_surge
            dv = (fy - self.m_surge * r * u) / self.m_sway
            # Munk moment: a slender body with unequal surge/sway added mass
            # gets a real destabilising yaw moment when moving at an angle.
            dr = (mz + (self.m_surge - self.m_sway) * u * v) / self.iz_eff

            # --- world-frame position rates (transform body velocity to world) ---
            dx = u * c - v * s
            dy = u * s + v * c

            # --- heading representation rates ---
            dc = -r * s
            ds = r * c

            return np.array([dx, dy, dc, ds, du, dv, dr])

        # --- integrate ---
        next_arr = solver(dynamics, state.to_array(), dt)
        next_state = State.from_array(next_arr)

        # --- leave the tf tree consistent with the state we return ---
        self._set_kinematic_frames(next_state, sheet_limit_rad)

        # --- rebuild state ---
        return next_state

    def _update_dynamic_frames(
        self,
        state: State,
        sheet_limit_rad: float,
        rudder_angle_deg: float,
        dt: float,
    ) -> None:
        """Update every dynamic frame at once (kinematics + rudder servo)."""
        self._set_kinematic_frames(state, sheet_limit_rad)
        self._advance_rudder_servo(rudder_angle_deg, dt)

    def _set_kinematic_frames(self, state: State, sheet_limit_rad: float) -> None:
        """Set the boat and sail frames from an instantaneous state.

        Pure function of the state: safe to call inside an integrator stage.
        """
        self.tf.add_frame(
            name="boat",
            parent="world",
            transform=Transform2D(x=state.x, y=state.y, c=state.psi[0], s=state.psi[1]),
        )

        sail_angle = self._resolve_sail_angle_from_sheet(state=state, sheet_limit_rad=sheet_limit_rad)
        self.last_sail_angle_rad = sail_angle
        self.tf.add_frame(
            name="sail",
            parent="boat",
            transform=Transform2D(
                x=self.sail_cfg.get("x_pos", 0.0),
                y=self.sail_cfg.get("y_pos", 0.0),
                c=float(np.cos(sail_angle)),
                s=float(np.sin(sail_angle)),
            ),
        )

    def _advance_rudder_servo(self, rudder_angle_deg: float, dt: float) -> None:
        """Advance the rudder servo model by one step and set its frame.

        This is discrete actuator state, so it must be advanced once per step,
        never once per integrator stage.
        """
        # Rudder: smooth + auto-center for easier manual control.
        cmd_deg = float(rudder_angle_deg)
        deadband_deg = float(self.rudder_cfg.get("deadband_deg", 1.5))
        center_tau_s = float(self.rudder_cfg.get("center_tau_s", 0.6))
        max_rate_deg_s = float(self.rudder_cfg.get("max_rate_deg_s", 120.0))

        if abs(cmd_deg) <= deadband_deg:
            cmd_deg = 0.0

        if center_tau_s > 0.0 and cmd_deg == 0.0:
            # Exponential return-to-center when you "let go".
            alpha = float(np.clip(dt / center_tau_s, 0.0, 1.0))
            self._rudder_angle_deg = (1.0 - alpha) * self._rudder_angle_deg
        else:
            # Rate-limit toward commanded angle.
            max_step = max_rate_deg_s * float(dt)
            err = cmd_deg - self._rudder_angle_deg
            self._rudder_angle_deg += float(np.clip(err, -max_step, max_step))
        rudder_rad = float(np.radians(self._rudder_angle_deg))
        self.tf.add_frame(
            name="rudder",
            parent="boat",
            transform=Transform2D(
                x=self.rudder_cfg.get("x_pos", 0.0),
                y=self.rudder_cfg.get("y_pos", 0.0),
                c=float(np.cos(rudder_rad)),
                s=float(np.sin(rudder_rad)),
            ),
        )

    def _resolve_sail_angle_from_sheet(self, state: State, sheet_limit_rad: float) -> float:
        """Resolve sail angle from apparent wind side and geometric sheet angle.

        The sail free-spins with apparent wind, constrained by sheet limit.
        Luffing/depower remains in the aerodynamic sail model.
        """

        # Compute wind vector in world frame
        wind_speed = float(self.sail_cfg.get("wind_speed", 0.0))
        wind_angle_deg = float(self.sail_cfg.get("wind_dir_deg", 0.0))
        wind_rad = float(np.radians(wind_angle_deg))
        wind_world = wind_speed * np.array([np.cos(wind_rad), np.sin(wind_rad)], dtype=float)

        # Compute boat velocity in world frame
        v_boat_world = self.tf.vector_to_frame(np.array([state.u, state.v], dtype=float), "boat", "world")
        apparent_wind_world = wind_world - v_boat_world
        apparent_wind_boat = self.tf.vector_to_frame(apparent_wind_world, "world", "boat")
        awa = float(np.arctan2(-apparent_wind_boat[1], -apparent_wind_boat[0]))

        # Pure geometric sheeting:
        # - free sail follows |AWA| (weather-vane behavior)
        # - sheet is a geometric stop at |sheet_limit_rad|
        # - side follows apparent-wind side, with hysteresis when centered
        sheet_limit = float(np.clip(np.abs(sheet_limit_rad), 0.0, 0.5 * np.pi))
        if sheet_limit <= 0.0:
            return 0.0
        free_mag = float(np.clip(np.abs(awa), 0.0, 0.5 * np.pi))
        boom_mag = min(free_mag, sheet_limit)

        wind_side = float(np.sign(apparent_wind_boat[1]))
        if wind_side == 0.0:
            wind_side = float(np.sign(self.last_sail_angle_rad))
            if wind_side == 0.0:
                wind_side = -1.0

        # Coordinate convention: positive boat-frame Y maps to opposite visual-Z side.
        return float(-wind_side * boom_mag)

    # --- Physics core ----------------------------------------
    def _forces(self, state: State) -> tuple[float, float, float]:
        """Compute total body-frame forces and yaw moment."""
        fx_total = 0.0
        fy_total = 0.0
        mz_total = 0.0
        self.last_forces = {}

        # Helpful for debugging runaway forces.
        u, v, r = float(state.u), float(state.v), float(state.r)
        speed = float(np.hypot(u, v))

        for component in self.components:
            result = np.atleast_1d(component.compute(state, self.tf))
            fx, fy = float(result[0]), float(result[1])
            mz_direct = float(result[2]) if len(result) > 2 else 0.0

            # Track sail contribution for visualization.
            if component is self.sail:
                self.last_sail_force = (fx, fy)

            # Track per-component forces for visualization.
            name = (
                "hull"
                if component is self.hull
                else "keel"
                if component is self.keel
                else "rudder"
                if component is self.rudder
                else "sail"
            )
            self.last_forces[name] = (fx, fy)

            # Moment about CG (2D cross product; x_pos = arm along boat, y_pos = lateral offset)
            x_pos = component.p.get("x_pos", 0.0)
            y_pos = component.p.get("y_pos", 0.0)
            mz = x_pos * fy - y_pos * fx + mz_direct

            # Sum forces
            fx_total += fx
            fy_total += fy
            mz_total += mz

        self.last_forces["total"] = (fx_total, fy_total)
        return fx_total, fy_total, mz_total
