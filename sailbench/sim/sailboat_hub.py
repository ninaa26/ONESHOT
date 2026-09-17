"""Compose a simulated sailboat."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from sailbench.dynamics.windage import Windage
from sailbench.models.model import State
from sailbench.sim.actuator import Actuator
from sailbench.sim.part_config import compose, model_name
from sailbench.sim.registry import lookup
from sailbench.tf.tf_tree import TFTree2D, Transform2D

# Importing these two packages is what registers the models in them, and so what
# decides which names a config may select. The hub no longer keeps a table of its
# own: a model is offered because it exists and said so, not because this file
# was remembered to be edited.
import sailbench.dynamics  # isort:skip
import sailbench.foils  # noqa: F401  imported for the registrations  isort:skip

CONFIG_PATH = "configs/"

# The parts whose section names a model in the registry. The hull is one of them:
# `BasicHullModel` and the two hydro models are alternatives for the same slot.
MODELLED_PARTS: tuple[str, ...] = ("sail", "keel", "rudder", "hull")


class SailboatHub:
    """A hub to manage sailboat simulation components."""

    def __init__(self, config_file: str, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        """Initialize the SailboatHub with configuration from a YAML file.

        `overrides` is layered on top of the file, one mapping per section
        (`{"sail": {"model": "orc_main"}}`), so a caller can swap a component's
        model or a parameter without editing the YAML. The web shipyard uses
        this; the YAML stays the boat's record.

        Overrides land before composition, which is what lets one of them pick a
        different model and get that model's own block rather than the block the
        file happened to select.
        """
        with Path(CONFIG_PATH + config_file).open() as file:
            cfg = yaml.safe_load(file)
        for section, values in (overrides or {}).items():
            cfg.setdefault(section, {}).update(values)

        # Each modelled section is reduced to the parameters of the one model it
        # selects. A section that has not been given a `models` mapping is
        # returned as it stands, so this is a no-op for every config written
        # before the shape existed.
        for part in MODELLED_PARTS:
            cfg[part] = compose(part, cfg[part])

        self.simulation_cfg = cfg["simulation"]
        self.boat_cfg = cfg["boat"]
        self.hull_cfg = cfg["hull"]
        self.keel_cfg = cfg["keel"]
        self.rudder_cfg = cfg["rudder"]
        self.sail_cfg = cfg["sail"]
        self.environment_cfg = cfg.get("environment", {})
        self.windage_cfg = cfg.get("windage", {})
        # Keys each component states for itself, captured before anything is
        # injected. Everything else in a component's parameters came from the
        # `environment` block and may be refreshed from it.
        self._component_own_keys = {
            id(section): set(section)
            for section in (self.hull_cfg, self.keel_cfg, self.rudder_cfg, self.sail_cfg, self.windage_cfg)
        }

        self.tf = TFTree2D()
        # Last computed sail force in boat frame (Fx, Fy) for diagnostics / UI.
        self.last_sail_force: tuple[float, float] = (0.0, 0.0)
        # Last resolved sail angle in radians after sheet-limit + wind logic.
        self.last_sail_angle_rad: float = 0.0
        # Per-component forces (boat frame) for visualization.
        self.last_forces: dict[str, tuple[float, float]] = {}

        # Actuators are built in boat_factory, from the component configs.

        self.boat_factory()

    def boat_factory(self) -> None:
        """Instantiate boat components from configs."""
        self._apply_environment()

        # Rudder and sheet are the same kind of actuator with different
        # constants. Legacy defaults are preserved for configs that have not
        # stated tau_s, so their behaviour is unchanged.
        self.rudder_actuator = Actuator(
            tau_s=float(self.rudder_cfg.get("tau_s", 0.0)),
            max_rate=float(self.rudder_cfg.get("max_rate_deg_s", 120.0)),
            deadband=float(self.rudder_cfg.get("deadband_deg", 1.5)),
            center_tau_s=float(self.rudder_cfg.get("center_tau_s", 0.6)),
        )
        # The sheet moves in radians. Unset, tau is 0 and there is no rate limit,
        # which is the previous behaviour: the sail tracked its command instantly.
        self.sail_actuator = Actuator(
            tau_s=float(self.sail_cfg.get("tau_s", 0.0)),
            max_rate=np.radians(float(self.sail_cfg.get("max_rate_deg_s", 0.0))),
        )

        self.sail = self._model_for("sail", self.sail_cfg)(self.sail_cfg)
        self.rudder = self._model_for("rudder", self.rudder_cfg)(self.rudder_cfg)
        self.hull = self._model_for("hull", self.hull_cfg)(self.hull_cfg)
        self.keel = self._model_for("keel", self.keel_cfg)(self.keel_cfg)
        self.components = [self.hull, self.keel, self.sail, self.rudder]


        # Above-water drag is its own component, not part of the sail: the sail
        # is a trimmable lifting surface, the mast and topsides are bluff bodies.
        # Gated on a drag area rather than on the section existing, because
        # _apply_environment populates every section it is handed -- an empty
        # `windage` dict comes back non-empty and would build a do-nothing model.
        has_windage = any(
            float(self.windage_cfg.get(key, 0.0)) > 0.0 for key in ("frontal_area_m2", "drag_area_m2")
        )
        self.windage = Windage(self.windage_cfg) if has_windage else None
        if self.windage is not None:
            self.components.append(self.windage)
            # Sync at construction as well as per step. Without this the windage
            # model carries no wind until the first step(), so anything calling
            # compute() directly -- a test, an analysis script, the first frame
            # of the web UI -- silently gets zero windage.
            self._sync_wind()
        self.m = self.boat_cfg.get("mass", self.boat_cfg.get("m", 27.0))
        self.iz = self.boat_cfg.get("inertia_z", self.boat_cfg.get("Iz", 25.0))

        # Added mass: the water the hull drags along with it. Resolved once, from
        # the hull's own geometry. Zero unless the hull model offers it and is
        # configured for it, in which case the equations below reduce to the
        # rigid-body ones exactly.
        self.hull_cfg.setdefault("mass", self.m)
        # Only a hull model with measured stations to integrate offers added mass;
        # the linear and quadratic hydro models have no geometry to derive it from.
        added: tuple[float, float, float] = (
            self.hull.added_mass() if hasattr(self.hull, "added_mass") else (0.0, 0.0, 0.0)
        )
        self.a_surge, self.a_sway, self.a_yaw = added

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

    @staticmethod
    def _model_for(part: str, cfg: Mapping[str, Any]) -> type:
        """Return the class registered for the model this section names; default `basic`."""
        return cast("type", lookup(part, model_name(cfg)))

    def _apply_environment(self) -> None:
        """Offer the shared `environment` values to every component as defaults.

        Components used to default their own fluid density under their own key
        name (`rho`, `water_density`, `air_density`), none of which any config
        set, so every one of them silently fell back and the configured value did
        nothing.

        The hub layers the `environment` block underneath each component's own
        parameters rather than assigning named keys, so it stays ignorant of
        which component wants which quantity: a foil asks for `rho_air`, a hull
        for `rho_water`, and a component that needs neither sees no change. A
        value set in the component's own section still wins, so a component can
        override the shared one.
        """
        for cfg in (self.hull_cfg, self.keel_cfg, self.rudder_cfg, self.sail_cfg, self.windage_cfg):
            own = self._component_own_keys.get(id(cfg), set())
            for key, value in self.environment_cfg.items():
                # setdefault would be wrong here: after the first pass the injected
                # value is indistinguishable from one the component stated itself,
                # so a later change to `environment` would silently not apply.
                if key not in own:
                    cfg[key] = value

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
        self._sync_wind()

        # Advance the actuators BEFORE integrating, so a command is in effect
        # during the step that issued it. Advancing them afterwards left the boat
        # sailing each step on the previous step's rudder: a one-step input lag,
        # which shrinks linearly with dt and so pins the whole integration to
        # first order no matter how good the solver is.
        #
        # Both are held constant across the step, which is what makes the lag
        # exact rather than merely small.
        sheet_cmd = float(np.abs(sail_angle))
        rudder_cmd = float(rudder_angle)

        def dynamics(arr: np.ndarray, t_offset: float) -> np.ndarray:
            """State derivative; arr = [x, y, c, s, u, v, r]."""
            state_vec = State.from_array(arr)

            # Sample the actuators where they will actually be at this stage.
            # Freezing them at their start-of-step values is a first-order
            # approximation of a surface that is still moving, and measurement
            # showed that was setting the accuracy of the whole trajectory.
            sheet = self.sail_actuator.position_at(sheet_cmd, t_offset)
            rudder = self.rudder_actuator.position_at(rudder_cmd, t_offset)

            # The sail resolves its force through the boat and sail frames, and
            # the rudder through the rudder frame. Left alone, those frames hold
            # the heading from the end of the previous step, so every RK4 stage
            # evaluates its forces against a stale attitude and the integrator
            # collapses to first order -- four force evaluations per step buying
            # Euler-grade accuracy. Re-deriving them from this stage's state is
            # what makes the fourth-order behaviour real.
            self._set_kinematic_frames(state_vec, abs(sheet), rudder)

            fx, fy, mz = self._forces(state_vec)

            c, s = arr[2], arr[3]  # Heading cosine, sine
            u, v, r = arr[4], arr[5], arr[6]

            # --- body-frame accelerations, rigid body plus added mass ---
            # Fossen's 3-DOF form. Added mass appears three times and each one
            # matters: in the inertia that resists acceleration, in the Coriolis
            # terms (a turning boat carries its entrained water round with it),
            # and in the Munk moment.
            #
            # The Munk moment, -(A22 - A11) u v, is destabilising: a hull moving
            # at a drift angle is pushed to increase it. That is a real property
            # of a slender body in a fluid, and leaving it out gives the hull a
            # directional stability it does not have -- which is exactly the sort
            # of thing a policy will learn to lean on.
            m_surge = self.m + self.a_surge
            m_sway = self.m + self.a_sway
            i_yaw = self.iz + self.a_yaw

            du = (fx + m_sway * v * r) / m_surge
            dv = (fy - m_surge * u * r) / m_sway
            dr = (mz - (self.a_sway - self.a_surge) * u * v) / i_yaw

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

        # --- commit the actuators, then leave the tf tree consistent ---
        self._advance_actuators(sheet_cmd, rudder_cmd, dt)
        self._set_kinematic_frames(next_state, abs(self.sail_actuator.position))

        # --- rebuild state ---
        return next_state

    def _sync_wind(self) -> None:
        """Keep every above-water component on the same wind as the sail.

        Wind lives in the sail's config section rather than in `environment`,
        and callers change it by writing there -- the web runner does exactly
        that. Copying it across each step means a second aerodynamic component
        cannot quietly run on the wind from whenever it was constructed. If wind
        ever moves into `environment`, the layering in _apply_environment covers
        this and the method goes away.
        """
        if self.windage is None:
            return
        for key in ("wind_speed", "wind_dir_deg"):
            if key in self.sail_cfg:
                self.windage_cfg[key] = self.sail_cfg[key]

    def _set_kinematic_frames(
        self, state: State, sheet_limit_rad: float, rudder_angle_deg: float | None = None,
    ) -> None:
        """Set the boat, sail and rudder frames.

        A pure function of the state and the actuator positions handed to it, so
        it is safe to call inside an integrator stage. The actuators themselves
        are never advanced here -- they are sampled, which is what keeps one
        step's worth of slew from being applied once per RK4 stage.

        `rudder_angle_deg` defaults to wherever the rudder actuator currently is.
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

        rudder_deg = self.rudder_actuator.position if rudder_angle_deg is None else float(rudder_angle_deg)
        self._place_rudder_frame(rudder_deg)

    def _advance_actuators(self, sail_angle_rad: float, rudder_angle_deg: float, dt: float) -> None:
        """Advance both actuators one step and set the rudder frame.

        Actuator state, not a function of the boat state: call exactly once per
        step, outside the integrator.
        """
        self.sail_actuator.advance(float(np.abs(sail_angle_rad)), dt)
        self._place_rudder_frame(self.rudder_actuator.advance(float(rudder_angle_deg), dt))

    def _place_rudder_frame(self, rudder_angle_deg: float) -> None:
        """Put the rudder frame at a given deflection."""
        rudder_rad = float(np.radians(rudder_angle_deg))
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
                else "windage"
                if component is self.windage
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
