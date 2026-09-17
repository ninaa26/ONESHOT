"""Added mass: the water the hull carries with it, and what it does to motion."""

import math

import numpy as np
import pytest

from sailbench.dynamics.basic_hull_model import BasicHullModel, MeasuredHullModel
from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

# Flingo's measured stations, sim frame.
SECTIONS = [
    {"x_m": 0.693, "draft_m": 0.031},
    {"x_m": 0.513, "draft_m": 0.067},
    {"x_m": 0.337, "draft_m": 0.092},
    {"x_m": 0.157, "draft_m": 0.108},
    {"x_m": -0.019, "draft_m": 0.108},
    {"x_m": -0.199, "draft_m": 0.093},
    {"x_m": -0.375, "draft_m": 0.062},
]
BASE = {"L": 1.372, "B": 0.492, "T": 0.110, "rho_water": 1000.0, "mass": 27.0}
MEASURED = {**BASE, "wetted_surface_m2": 0.571}


class TestStripTheory:
    """Integrating rho*pi*T^2 over the measured stations."""

    def test_basic_hull_has_none(self) -> None:
        """The basic hull reports no added mass."""
        assert BasicHullModel(BASE).added_mass() == (0.0, 0.0, 0.0)

    def test_a_single_station_is_a_config_error(self) -> None:
        """One station cannot be integrated, so the measured hull refuses it."""
        with pytest.raises(ValueError, match="sections"):
            MeasuredHullModel({**MEASURED, "sections": SECTIONS[:1]})

    def test_sway_added_mass_is_comparable_to_the_boat(self) -> None:
        """For a hull this beamy it is not a correction, it is a second boat."""
        _, a22, _ = MeasuredHullModel({**MEASURED, "sections": SECTIONS}).added_mass()
        assert 0.8 * BASE["mass"] < a22 < 1.2 * BASE["mass"]

    def test_matches_the_trapezoidal_integral(self) -> None:
        """A22 = integral of rho*pi*T^2 dx over the stations."""
        xs = np.array([s["x_m"] for s in reversed(SECTIONS)])
        ts = np.array([s["draft_m"] for s in reversed(SECTIONS)])
        expected = float(np.trapezoid(1000.0 * math.pi * ts**2, xs))
        _, a22, _ = MeasuredHullModel({**MEASURED, "sections": SECTIONS}).added_mass()
        assert a22 == pytest.approx(expected)

    def test_yaw_term_is_the_same_integral_weighted_by_x_squared(self) -> None:
        """A66 = integral of rho*pi*T^2 x^2 dx."""
        xs = np.array([s["x_m"] for s in reversed(SECTIONS)])
        ts = np.array([s["draft_m"] for s in reversed(SECTIONS)])
        expected = float(np.trapezoid(1000.0 * math.pi * ts**2 * xs**2, xs))
        _, _, a66 = MeasuredHullModel({**MEASURED, "sections": SECTIONS}).added_mass()
        assert a66 == pytest.approx(expected)

    def test_station_order_does_not_matter(self) -> None:
        """The table may be given bow-first or stern-first."""
        forward = MeasuredHullModel({**MEASURED, "sections": SECTIONS}).added_mass()
        reversed_ = MeasuredHullModel({**MEASURED, "sections": list(reversed(SECTIONS))}).added_mass()
        assert forward == pytest.approx(reversed_)

    def test_surge_is_a_small_fraction_of_displacement(self) -> None:
        """A slender hull moving along its axis disturbs very little water."""
        a11, a22, _ = MeasuredHullModel({**MEASURED, "sections": SECTIONS}).added_mass()
        assert a11 == pytest.approx(0.05 * BASE["mass"])
        assert a11 < 0.1 * a22

    def test_deeper_sections_carry_more_water(self) -> None:
        """T^2 means draft dominates."""
        deep = [{"x_m": s["x_m"], "draft_m": 2.0 * s["draft_m"]} for s in SECTIONS]
        _, shallow_a22, _ = MeasuredHullModel({**MEASURED, "sections": SECTIONS}).added_mass()
        _, deep_a22, _ = MeasuredHullModel({**MEASURED, "sections": deep}).added_mass()
        assert deep_a22 == pytest.approx(4.0 * shallow_a22)


class TestEquationsOfMotion:
    """What the hub does with it."""

    def test_flingo_picks_it_up(self) -> None:
        """The configured boat resolves non-zero added mass at construction."""
        hub = SailboatHub("flingo_floty.yaml")
        assert hub.a_sway > 0.0
        assert hub.a_yaw > 0.0

    def test_a_hull_without_sections_reduces_to_rigid_body(self) -> None:
        """Configs that have not opted in keep the previous equations exactly."""
        hub = SailboatHub("basic_sailbot.yaml")
        assert (hub.a_surge, hub.a_sway, hub.a_yaw) == (0.0, 0.0, 0.0)

    def test_munk_moment_is_destabilising(self) -> None:
        """A hull at a drift angle is pushed to increase it, not reduce it.

        That is a real property of a slender body in a fluid, and leaving it out
        gives the hull a directional stability it does not have. Tested with the
        force models removed, so only the inertia terms act.
        """
        for v0 in (0.2, -0.2):
            hub = SailboatHub("flingo_floty.yaml")
            hub.components = []
            st = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1.5, v0, 0.0]))
            for _ in range(50):
                st = hub.step(st, 0.02, rk4_step, 0.0, 0.0)
            # bow turns away from the slip => r opposes v
            assert st.r * v0 < 0.0

    def test_no_munk_moment_without_drift(self) -> None:
        """Running straight, there is nothing to destabilise."""
        hub = SailboatHub("flingo_floty.yaml")
        hub.components = []
        st = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0]))
        for _ in range(50):
            st = hub.step(st, 0.02, rk4_step, 0.0, 0.0)
        assert abs(st.r) < 1e-12

    def test_steady_state_balances_including_the_munk_term(self) -> None:
        """A settled boat has zero acceleration in all three degrees of freedom.

        The component moments do NOT sum to zero once added mass is modelled;
        they sum to the Munk moment. Asserting the old invariant would now be
        wrong.
        """
        hub = SailboatHub("flingo_floty.yaml")
        heading = math.radians(90.0) - math.pi + math.radians(40.0)
        st = State.from_array(
            np.array([0.0, 0.0, math.cos(heading), math.sin(heading), 1.0, 0.0, 0.0])
        )
        integral = 0.0
        for _ in range(4000):
            psi = math.atan2(st.psi[1], st.psi[0])
            err = (heading - psi + math.pi) % (2 * math.pi) - math.pi
            integral += err * 0.02
            rudder = float(np.clip(-(40 * err + 8 * integral + 6 * (-st.r)), -35, 35))
            st = hub.step(st, 0.02, rk4_step, math.radians(10.0), rudder)

        fx, fy, mz = hub._forces(st)
        munk = (hub.a_sway - hub.a_surge) * st.u * st.v
        assert mz == pytest.approx(munk, abs=1e-3)
        assert (mz - munk) / (hub.iz + hub.a_yaw) == pytest.approx(0.0, abs=1e-5)
        assert (fx + (hub.m + hub.a_sway) * st.v * st.r) / (hub.m + hub.a_surge) == pytest.approx(0.0, abs=1e-5)
        assert (fy - (hub.m + hub.a_surge) * st.u * st.r) / (hub.m + hub.a_sway) == pytest.approx(0.0, abs=1e-5)
