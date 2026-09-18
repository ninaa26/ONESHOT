"""That each component is actually wired to the physics it is supposed to use.

Every test here closes a gap found by mutation: the underlying kernels were
covered, but nothing checked that the components call them. Reverting the keel's
yaw-rate inflow, either foil's stall blending, or the hub's use of sway added
mass all left the suite green.
"""

import math

import numpy as np
import pytest

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step
from sailbench.tf.tf_tree import TFTree2D


def hub_at(u: float = 1.5, v: float = 0.0, r: float = 0.0, sheet: float = 0.3) -> tuple:
    """A configured hub with its frames set for the given state."""
    hub = SailboatHub("flingo_floty.yaml")
    state = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, u, v, r]))
    hub._set_kinematic_frames(state, sheet)
    return hub, state


class TestKeelSeesYawRate:
    """The keel sits forward of the centre of rotation, so a turn is sideslip."""

    def force(self, r: float) -> np.ndarray:
        """Keel force in the boat frame at yaw rate `r`."""
        hub, _ = hub_at()
        state = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1.5, 0.0, r]))
        return np.asarray(hub.keel.compute(state, hub.tf), dtype=float)

    def test_no_side_force_going_straight(self) -> None:
        """Baseline: no sideslip, no lift."""
        assert self.force(0.0)[1] == pytest.approx(0.0, abs=1e-9)

    def test_turning_alone_generates_side_force(self) -> None:
        """Yaw rate alone must load the keel; ignoring it gave zero yaw damping."""
        assert abs(self.force(0.5)[1]) > 10.0

    def test_side_force_opposes_the_turn(self) -> None:
        """A keel forward of the centre resists rotation, not assists it."""
        assert self.force(0.5)[1] < 0.0
        assert self.force(-0.5)[1] > 0.0

    def test_is_antisymmetric_in_yaw_rate(self) -> None:
        """Turning the other way mirrors the force."""
        assert self.force(0.5)[1] == pytest.approx(-self.force(-0.5)[1])


class TestFoilsUseStallBlending:
    """The blend kernel is tested elsewhere; this is that the foils call it."""

    def test_keel_becomes_a_flat_plate_broadside(self) -> None:
        """At 90 degrees of sideslip the keel must give the plate result.

        0.5 * rho * CN * area * v^2, with CN the keel's own plate normal force.
        Without blending the keel reads the section table at 90 degrees instead,
        which is an extrapolation. The coefficient is taken from the model
        rather than written in here, because it is a function of aspect ratio,
        not a constant -- that is what :meth:`Foil.plate_normal_force` settles.
        """
        hub, _ = hub_at()
        state = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1e-4, 1.0, 0.0]))
        force = np.asarray(hub.keel.compute(state, hub.tf), dtype=float)
        cn = hub.keel.plate_normal_force()
        plate = 0.5 * 1000.0 * cn * float(hub.keel.p["area"]) * 1.0**2
        assert abs(force[1]) == pytest.approx(plate, rel=1e-3)

    def test_keel_plate_coefficient_follows_its_aspect_ratio(self) -> None:
        """And that coefficient is the Hoerner value for this keel, not the 2-D one.

        The keel is AR 8 effective, so CN is about 1.25. It was charged 2.0.
        """
        hub, _ = hub_at()
        assert hub.keel.effective_aspect_ratio() == pytest.approx(8.0, rel=1e-3)
        assert hub.keel.plate_normal_force() == pytest.approx(1.254, abs=0.01)

    def rudder_force(self, deflection_deg: float) -> np.ndarray:
        """Rudder force in the boat frame at a given deflection."""
        hub, state = hub_at()
        hub.rudder_actuator.position = deflection_deg
        hub._place_rudder_frame(deflection_deg)
        return np.asarray(hub.rudder.compute(state, hub.tf), dtype=float)

    def test_rudder_makes_no_lift_broadside(self) -> None:
        """Held square across the flow, the rudder is a plate: pure drag.

        A symmetric plate at 90 degrees has no lift by symmetry, and blending
        gives exactly that. The section table extrapolated to 90 degrees does
        not -- it still returns CL 0.064 -- so this separates the blend being
        applied from the raw polar being read.
        """
        force = self.rudder_force(90.0)
        # 1e-3 N, against 1.99 N from the unblended table: six orders of margin.
        assert force[1] == pytest.approx(0.0, abs=1e-3), "broadside rudder is making lift"
        assert force[0] < -1.0, "broadside rudder should be almost pure drag"

    def test_rudder_keeps_responding_far_past_stall(self) -> None:
        """Deflection must still change the force at 45 and 80 degrees.

        This is what separates blending from clamping. The clamps pin the angle
        of attack at aoa_limit_deg and the coefficients at cl_max/cd_max, so
        every deflection past about 30 degrees returns an identical force -- the
        rudder stops being a control surface and the derivative goes flat.
        """
        mid, far = self.rudder_force(45.0), self.rudder_force(80.0)
        assert abs(far[0] - mid[0]) > 1.0, "rudder drag frozen past stall"
        assert abs(far[1] - mid[1]) > 1.0, "rudder side force frozen past stall"

    def test_rudder_drag_grows_monotonically_with_deflection(self) -> None:
        """More deflection, more drag, all the way out."""
        drags = [self.rudder_force(a)[0] for a in (10.0, 30.0, 45.0, 60.0, 80.0)]
        assert drags == sorted(drags, reverse=True), "rudder drag stopped growing"


class TestHubUsesAddedMass:
    """Added mass has to reach the equations of motion, not just be computed."""

    class _Constant:
        """A stand-in component with a fixed force, so the algebra is checkable."""

        def __init__(self, fx: float, fy: float) -> None:
            self.p = {"x_pos": 0.0, "y_pos": 0.0}
            self._f = np.array([fx, fy], dtype=float)

        def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
            """Return the fixed force."""
            return self._f

    def accel(self, fx: float, fy: float, u: float, v: float) -> tuple[float, float]:
        """Surge and sway acceleration over one very short step."""
        hub = SailboatHub("flingo_floty.yaml")
        hub.components = [self._Constant(fx, fy)]
        dt = 1e-6
        st0 = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, u, v, 0.0]))
        st1 = hub.step(st0, dt, rk4_step, 0.0, 0.0)
        return (st1.u - u) / dt, (st1.v - v) / dt

    def test_sway_uses_the_added_mass(self) -> None:
        """A lateral force accelerates the boat as fy / (m + A22), not fy / m.

        For this hull that is nearly a factor of two.
        """
        hub = SailboatHub("flingo_floty.yaml")
        _, dv = self.accel(0.0, 50.0, 0.0, 0.0)
        assert dv == pytest.approx(50.0 / (hub.m + hub.a_sway), rel=1e-4)
        assert dv != pytest.approx(50.0 / hub.m, rel=1e-2)

    def test_surge_uses_the_added_mass(self) -> None:
        """Surge added mass is small, but it is applied."""
        hub = SailboatHub("flingo_floty.yaml")
        du, _ = self.accel(50.0, 0.0, 0.0, 0.0)
        assert du == pytest.approx(50.0 / (hub.m + hub.a_surge), rel=1e-4)

    def test_surge_coriolis_carries_the_entrained_water(self) -> None:
        """The v*r term scales with (m + A22), not m: turning carries it round."""
        hub = SailboatHub("flingo_floty.yaml")
        u, v, r = 1.5, 0.2, 0.4
        st0 = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, u, v, r]))
        hub.components = [self._Constant(0.0, 0.0)]
        dt = 1e-6
        st1 = hub.step(st0, dt, rk4_step, 0.0, 0.0)
        du = (st1.u - u) / dt
        expected = (hub.m + hub.a_sway) * v * r / (hub.m + hub.a_surge)
        assert du == pytest.approx(expected, rel=1e-3)


class TestEnvironmentReachesComponents:
    """The `environment` block must actually govern, not coincide with defaults."""

    def test_water_density_changes_keel_force(self) -> None:
        """Salt water must move the numbers; the old silent fallback did not."""
        forces = []
        for rho in (1000.0, 1200.0):
            hub = SailboatHub("flingo_floty.yaml")
            hub.environment_cfg["rho_water"] = rho
            hub.boat_factory()
            state = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1.5, 0.15, 0.0]))
            hub._set_kinematic_frames(state, 0.3)
            forces.append(abs(float(np.asarray(hub.keel.compute(state, hub.tf), dtype=float)[1])))
        assert forces[1] == pytest.approx(1.2 * forces[0], rel=1e-6)

    def test_air_density_changes_sail_force(self) -> None:
        """Same for the rig."""
        forces = []
        for rho in (1.225, 2.450):
            hub = SailboatHub("flingo_floty.yaml")
            hub.environment_cfg["rho_air"] = rho
            hub.boat_factory()
            heading = math.radians(90.0) - math.pi + math.radians(45.0)
            state = State.from_array(
                np.array([0.0, 0.0, math.cos(heading), math.sin(heading), 1.5, 0.0, 0.0])
            )
            hub._set_kinematic_frames(state, math.radians(20.0))
            forces.append(float(np.asarray(hub.sail.compute(state, hub.tf), dtype=float)[0]))
        assert forces[1] == pytest.approx(2.0 * forces[0], rel=1e-6)

    def test_viscosity_reaches_the_hull(self) -> None:
        """nu_water is the one environment value that differs from its default."""
        hub = SailboatHub("flingo_floty.yaml")
        assert hub.hull_cfg["nu_water"] == hub.environment_cfg["nu_water"]
        configured = hub.hull._friction_coefficient(1.7, 1.372)
        hub.hull.p["nu_water"] = 1.19e-6
        assert hub.hull._friction_coefficient(1.7, 1.372) != pytest.approx(configured, rel=1e-6)
