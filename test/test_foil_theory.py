"""Pin the foil sign convention with physical anchors.

These are the tests that would have caught the bugs in the old models: a rudder
that produced forward thrust at zero deflection, and force vectors rotated by
twice the angle of attack.
"""

from __future__ import annotations

import numpy as np
import pytest

from sailbench.foils.thin_foil import ThinFoil
from sailbench.models.foil_theory import fold_incidence, foil_coefficients, foil_force
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

KEEL_CFG = {"area": 0.13, "span": 0.4, "x_pos": 0.0, "y_pos": 0.0}
RUDDER_CFG = {"area": 0.046, "span": 0.25, "x_pos": -0.46, "y_pos": 0.0}


def _tree(rudder_deg: float = 0.0) -> TFTree2D:
    tf = TFTree2D()
    tf.add_frame("boat", "world", Transform2D(0.0, 0.0, 1.0, 0.0))
    tf.add_frame("keel", "boat", Transform2D(0.0, 0.0, 1.0, 0.0))
    rr = np.radians(rudder_deg)
    tf.add_frame("rudder", "boat", Transform2D(-0.46, 0.0, np.cos(rr), np.sin(rr)))
    return tf


# --- coefficient model -------------------------------------------------


def test_zero_incidence_gives_no_lift_and_only_parasitic_drag() -> None:
    cl, cd = foil_coefficients(0.0, aspect_ratio=2.5, cd0=0.008)
    assert cl == pytest.approx(0.0, abs=1e-12)
    assert cd == pytest.approx(0.008, abs=1e-9)


def test_lift_is_odd_and_drag_is_even_in_incidence() -> None:
    for a in np.radians([3.0, 8.0, 15.0, 40.0, 80.0]):
        cl_p, cd_p = foil_coefficients(a, 2.5, 0.008)
        cl_m, cd_m = foil_coefficients(-a, 2.5, 0.008)
        assert cl_p == pytest.approx(-cl_m, abs=1e-12)
        assert cd_p == pytest.approx(cd_m, abs=1e-12)


def test_finite_span_slope_is_well_below_two_pi() -> None:
    """The old NeuralFoil path applied 2-D section lift with no AR correction."""
    a = np.radians(5.0)
    cl, _ = foil_coefficients(a, aspect_ratio=2.5, cd0=0.008)
    assert cl < 0.6 * (2.0 * np.pi * a)


def test_induced_drag_grows_with_lift() -> None:
    _, cd_small = foil_coefficients(np.radians(2.0), 2.5, 0.008)
    _, cd_big = foil_coefficients(np.radians(10.0), 2.5, 0.008)
    assert cd_big > cd_small


def test_lift_is_bounded_by_a_physical_maximum() -> None:
    """A bounded CL is the property that matters: the old path had none."""
    for ar in (1.0, 2.5, 6.0):
        cls = [abs(foil_coefficients(np.radians(d), ar, 0.008)[0]) for d in range(0, 91)]
        assert max(cls) < 1.25


def test_low_aspect_foils_are_less_efficient_than_high_aspect_ones() -> None:
    """The two properties a keel actually depends on: slope and induced drag."""
    a = np.radians(5.0)
    cl_low, cd_low = foil_coefficients(a, aspect_ratio=1.0, cd0=0.008)
    cl_high, cd_high = foil_coefficients(a, aspect_ratio=8.0, cd0=0.008)
    # Shallower lift slope...
    assert cl_low < cl_high
    # ...and worse lift-to-drag.
    assert cl_low / cd_low < cl_high / cd_high


def test_coefficients_stay_finite_over_the_full_circle() -> None:
    for d in range(-180, 181):
        cl, cd = foil_coefficients(np.radians(d), 2.5, 0.008)
        assert np.isfinite(cl) and np.isfinite(cd)
        assert abs(cl) < 2.0
        assert 0.0 <= cd < 3.0


def test_fold_incidence_mirrors_flow_from_astern() -> None:
    assert fold_incidence(np.radians(175.0)) == pytest.approx(np.radians(-5.0))
    assert fold_incidence(np.radians(-175.0)) == pytest.approx(np.radians(5.0))


# --- force kernel: energy and direction --------------------------------


def test_foil_force_never_injects_energy() -> None:
    """A passive foil cannot do net work on the flow."""
    for vx in (-2.0, -0.5, 0.5, 2.0):
        for vy in (-1.0, -0.2, 0.0, 0.2, 1.0):
            velocity = np.array([vx, vy])
            f = foil_force(-velocity, area=0.13, rho=1000.0, aspect_ratio=2.5, cd0=0.008)
            assert float(f @ velocity) <= 1e-9


# --- the two anchors that pin the sign convention ----------------------


def test_keel_resists_leeway() -> None:
    tf = _tree()
    keel = ThinFoil(KEEL_CFG, frame="keel")
    for v in (-0.4, -0.1, 0.1, 0.4):
        f = keel.compute(State(0.0, 0.0, (1.0, 0.0), 1.0, v, 0.0), tf)
        assert np.sign(f[1]) == -np.sign(v), f"leeway v={v} not resisted"


def test_keel_drag_opposes_forward_motion() -> None:
    tf = _tree()
    keel = ThinFoil(KEEL_CFG, frame="keel")
    f = keel.compute(State(0.0, 0.0, (1.0, 0.0), 1.0, 0.0, 0.0), tf)
    assert f[0] < 0.0


def test_rudder_produces_drag_not_thrust_at_zero_deflection() -> None:
    """Regression: the old rudder produced positive Fx with the helm centred."""
    tf = _tree(rudder_deg=0.0)
    rudder = ThinFoil(RUDDER_CFG, frame="rudder")
    f = rudder.compute(State(0.0, 0.0, (1.0, 0.0), 1.0, 0.0, 0.0), tf)
    assert f[0] < 0.0
    assert f[1] == pytest.approx(0.0, abs=1e-9)


def test_rudder_never_produces_thrust_at_any_deflection() -> None:
    rudder = ThinFoil(RUDDER_CFG, frame="rudder")
    for deg in range(-40, 41, 5):
        f = rudder.compute(State(0.0, 0.0, (1.0, 0.0), 1.0, 0.0, 0.0), _tree(deg))
        assert f[0] < 1e-9, f"rudder at {deg} deg pushes the boat forward"


def test_rudder_force_is_near_perpendicular_to_the_flow() -> None:
    """Regression: the old rudder rotated its force by exactly twice the AoA.

    The flow is along the boat's x axis, so the force must sit close to the
    boat's y axis, tilted aft only by the foil's own drag (L/D of order 5-15
    for a low-aspect rudder). The old model put it at 90 + 2*deflection.
    """
    rudder = ThinFoil(RUDDER_CFG, frame="rudder")
    for deg in (5.0, 10.0, 15.0):
        f = rudder.compute(State(0.0, 0.0, (1.0, 0.0), 1.0, 0.0, 0.0), _tree(deg))
        angle = np.degrees(np.arctan2(f[1], f[0]))
        assert 90.0 <= angle <= 105.0, f"rudder force at {angle:.1f} deg, expected ~90"
        # Specifically not the old double-rotation signature.
        assert abs(angle - (90.0 + 2.0 * deg)) > 1.0


def test_positive_rudder_angle_yaws_the_boat_to_starboard() -> None:
    """Matches the convention the existing polar script and RL policies assume."""
    rudder = ThinFoil(RUDDER_CFG, frame="rudder")
    f = rudder.compute(State(0.0, 0.0, (1.0, 0.0), 1.0, 0.0, 0.0), _tree(20.0))
    mz = RUDDER_CFG["x_pos"] * f[1]
    assert mz < 0.0
