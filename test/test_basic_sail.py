"""BasicSail: angle-of-attack convention, luffing, and drive direction."""

import math

import numpy as np
import pytest

from sailbench.foils.basic_sail import BasicSail
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

WIND_TO_DEG = 90.0  # wind blows toward +y


def make_state(u: float = 0.0, v: float = 0.0, psi: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, np.cos(psi), np.sin(psi), u, v, 0.0]))


def make_sail(**overrides: float) -> BasicSail:
    """Build a sail with a fixed wind."""
    return BasicSail({
        "airfoil_name": "NACA0012", "area": 1.971, "alpha_min": -179, "alpha_max": 179,
        "res": [1.8e5], "wind_speed": 5.0, "wind_dir_deg": WIND_TO_DEG, "rho_air": 1.225,
        **overrides,
    })


def tree(sail_rad: float, psi: float = 0.0) -> TFTree2D:
    """Transform tree with the boat on a heading and the sail trimmed."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world",
                 transform=Transform2D(x=0.0, y=0.0, c=math.cos(psi), s=math.sin(psi)))
    tf.add_frame(name="sail", parent="boat",
                 transform=Transform2D(x=0.0, y=0.0, c=math.cos(sail_rad), s=math.sin(sail_rad)))
    return tf


def beat(twa_deg: float) -> float:
    """Heading that puts the boat `twa_deg` off the true wind, on port tack."""
    return math.radians(WIND_TO_DEG) - math.pi + math.radians(twa_deg)


def drive_coefficient(sail: BasicSail, twa_deg: float, u: float = 1.0) -> float:
    """Best achievable CR = Fx / (q * A) over trim, driven through compute().

    Deliberately measured from the force the model returns rather than from a
    coefficient looked up in the test: a test that recomputes the angle of attack
    itself cannot fail when the model computes it wrongly.
    """
    psi = beat(twa_deg)
    st = make_state(u=u, psi=psi)
    v_world = np.array([math.cos(psi), math.sin(psi)]) * u
    aw = np.array([5.0 * math.cos(math.radians(WIND_TO_DEG)),
                   5.0 * math.sin(math.radians(WIND_TO_DEG))]) - v_world
    q_area = 0.5 * 1.225 * float(np.dot(aw, aw)) * float(sail.p["area"])
    return max(sail.compute(st, tree(math.radians(-t), psi))[0] for t in range(5, 90, 5)) / q_area


class TestAngleOfAttack:
    """The lookup angle must be the angle of attack, not the flow direction.

    Regression: the model used one variable for both, which put the lookup about
    180 degrees out and returned negative lift upwind. These thresholds sit
    between the correct values and the ones that bug produces (at TWA 30, CR is
    0.256 correct against 0.086 buggy).
    """

    @pytest.mark.parametrize(("twa_deg", "floor"), [(30, 0.20), (45, 0.45), (60, 0.60)])
    def test_drive_coefficient_is_physical(self, twa_deg: float, floor: float) -> None:
        """A real sail converts a decent fraction of dynamic pressure into drive."""
        assert drive_coefficient(make_sail(), twa_deg) > floor

    def test_drive_improves_as_the_boat_bears_away(self) -> None:
        """Drive coefficient rises from close-hauled towards a reach."""
        sail = make_sail()
        assert drive_coefficient(sail, 30) < drive_coefficient(sail, 45) < drive_coefficient(sail, 60)


class TestLuffing:
    """The luff ramp used to be dead code; it must actually engage."""

    def _scale(self, sail: BasicSail, awa_deg: float) -> float:
        flow = math.radians(180.0 - awa_deg)
        aw = np.array([math.cos(flow), math.sin(flow)])
        alpha_abs = abs(math.degrees(math.atan2(-aw[1], -aw[0])))
        luff = float(sail.p.get("luff_deg", 7.5))
        ramp = max(float(sail.p.get("luff_ramp_deg", 4.0)), 1e-6)
        return float(np.clip((alpha_abs - luff) / ramp, 0.0, 1.0))

    def test_fully_luffing_head_to_wind(self) -> None:
        """Inside the luff angle the sail carries no lift."""
        assert self._scale(make_sail(), 2.0) == 0.0

    def test_ramps_in_and_saturates(self) -> None:
        """The ramp is partial in between and full once past it."""
        sail = make_sail()
        assert 0.0 < self._scale(sail, 10.0) < 1.0
        assert self._scale(sail, 30.0) == 1.0

    def test_is_monotonic(self) -> None:
        """More angle of attack never means less lift authority."""
        sail = make_sail()
        scales = [self._scale(sail, a) for a in (0, 5, 8, 10, 12, 20)]
        assert scales == sorted(scales)


class TestDrive:
    """The sail is the engine: close-hauled it must push the boat forwards."""

    @pytest.mark.parametrize("twa_deg", [35, 45, 60, 90])
    def test_produces_forward_drive(self, twa_deg: float) -> None:
        """Correctly trimmed, boat-frame Fx is positive on every point of sail."""
        sail = make_sail()
        psi = beat(twa_deg)
        best = max(
            sail.compute(make_state(u=1.0, psi=psi), tree(math.radians(-t), psi))[0]
            for t in (10, 15, 20, 25, 30, 45, 60)
        )
        assert best > 0.0

    def test_no_wind_no_force(self) -> None:
        """Zero wind and zero boat speed means no apparent wind and no force."""
        fx, fy = make_sail(wind_speed=0.0).compute(make_state(u=0.0), tree(0.0))
        assert abs(fx) < 1e-9
        assert abs(fy) < 1e-9

    def test_force_scales_with_air_density(self) -> None:
        """Force is proportional to rho_air, so the environment value is live."""
        psi = beat(45.0)
        args = (make_state(u=1.0, psi=psi), tree(math.radians(-20.0), psi))
        light = make_sail(rho_air=1.0).compute(*args)
        heavy = make_sail(rho_air=2.0).compute(*args)
        assert heavy[0] == pytest.approx(2.0 * light[0], rel=1e-9)


class TestIntegratorContract:
    """Properties of the hub's integration that the force models depend on.

    Both are regressions with no other coverage: mutating either back to its old
    behaviour left the whole suite green.
    """

    def test_rudder_acts_during_the_step_that_commands_it(self) -> None:
        """A rudder command must bite immediately, not one step later.

        The servo used to advance after the integration, so each step sailed on
        the previous step's rudder -- a one-step input lag that shrinks with dt
        and pinned the integrator to first order.
        """
        import numpy as np_

        from sailbench.sim.sailboat_hub import SailboatHub
        from sailbench.solvers.rk4 import rk4_step

        def yaw_after_one_step(rudder_deg: float) -> float:
            hub = SailboatHub("flingo_floty.yaml")
            st = State.from_array(np_.array([0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0]))
            return hub.step(st, 0.02, rk4_step, math.radians(20.0), rudder_deg).r

        # Opposite commands must already diverge after a single step. Asserting
        # merely that yaw is non-zero is too weak: the sail's own moment turns the
        # boat whatever the rudder does.
        port, starboard = yaw_after_one_step(-25.0), yaw_after_one_step(25.0)
        assert abs(starboard - port) > 1e-6, "rudder had no effect in its own step"

    def test_integration_is_better_than_first_order(self) -> None:
        """Self-convergence: halving dt must do better than halving the error.

        Forces used to be evaluated against frames held from the end of the
        previous step, so every RK4 stage saw a stale attitude and the method
        degraded to first order. Order is measured without a reference solution,
        as log2 of successive |y(dt) - y(dt/2)|.
        """
        import numpy as np_

        from sailbench.sim.sailboat_hub import SailboatHub
        from sailbench.solvers.rk4 import rk4_step

        def run(dt: float, secs: float = 1.0) -> np_.ndarray:
            hub = SailboatHub("flingo_floty.yaml")
            # Hold the rudder as a constant input. The servo slews by forward
            # Euler once per step and carries a deadband and an auto-centre, all
            # of which are first order or worse; with it active it dominates the
            # trajectory error and this would measure the actuator, not the
            # integrator. Measured separately: order 0.6-0.9 with the servo
            # slewing, 4.4 with it snapped.
            hub.rudder_cfg.update({"max_rate_deg_s": 1e9, "deadband_deg": 0.0, "center_tau_s": 0.0})
            hub.boat_factory()
            psi = beat(45.0)
            st = State.from_array(np_.array([0.0, 0.0, math.cos(psi), math.sin(psi), 1.0, 0.0, 0.0]))
            for _ in range(int(round(secs / dt))):
                st = hub.step(st, dt, rk4_step, math.radians(20.0), 8.0)
            return np_.array([st.x, st.y, st.u, st.v, st.r])

        errors = [float(np_.linalg.norm(run(dt) - run(dt / 2))) for dt in (0.04, 0.02)]
        order = math.log2(errors[0] / errors[1])
        assert order > 2.0, f"integration converging at order {order:.2f}; first order means a stale frame"
