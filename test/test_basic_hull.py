"""BasicHullModel resistance tests: skin friction, wave-making, wetted surface."""

import math

import numpy as np
import pytest

from sailbench.dynamics.basic_hull_model import GRAVITY, BasicHullModel, MeasuredHullModel
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D

BASE = {"L": 1.372, "B": 0.492, "T": 0.110, "rho_water": 1000.0}
SECTIONS = [{"x_m": 0.693, "draft_m": 0.031}, {"x_m": 0.157, "draft_m": 0.108}, {"x_m": -0.375, "draft_m": 0.062}]
MEASURED = {**BASE, "wetted_surface_m2": 0.571, "sections": SECTIONS}


def make_state(u: float = 0.0, v: float = 0.0, r: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, 1.0, 0.0, u, v, r]))


def surge(params: dict, u: float) -> float:
    """Surge force from a hull built with `params` at forward speed `u`."""
    return float(BasicHullModel({**BASE, **params}).compute(make_state(u=u), TFTree2D())[0])


class TestResiduaryResistance:
    """Wave-making resistance: the term that gives the hull a top speed."""

    def test_absent_by_default(self) -> None:
        """A config without c_wave keeps its previous resistance exactly."""
        assert surge({}, 2.0) == surge({"c_wave": 0.0}, 2.0)

    def test_opposes_motion_in_both_directions(self) -> None:
        """Wave drag resists travel whichever way the hull is moving."""
        assert surge({"c_wave": 0.01}, 2.0) < 0.0
        assert surge({"c_wave": 0.01}, -2.0) > 0.0

    def test_grows_faster_than_skin_friction(self) -> None:
        """Doubling speed must cost more than the quadratic friction term alone.

        This is the whole point of the term: friction alone scales as u^2 and so
        never stops the boat, which is why the hull used to pass hull speed.
        """
        friction_only = abs(surge({}, 2.0)) / abs(surge({}, 1.0))
        with_waves = abs(surge({"c_wave": 0.01}, 2.0)) / abs(surge({"c_wave": 0.01}, 1.0))
        assert with_waves > friction_only

    def test_negligible_well_below_hull_speed(self) -> None:
        """Below hull speed the quartic should barely matter."""
        v_hull = 0.4 * math.sqrt(GRAVITY * BASE["L"])
        slow = 0.3 * v_hull
        assert abs(surge({"c_wave": 0.01}, slow) - surge({}, slow)) < 0.02 * abs(surge({}, slow))

    def test_dominant_above_hull_speed(self) -> None:
        """Well past hull speed it should be the larger part of the resistance."""
        v_hull = 0.4 * math.sqrt(GRAVITY * BASE["L"])
        fast = 2.0 * v_hull
        assert abs(surge({"c_wave": 0.01}, fast)) > 2.0 * abs(surge({}, fast))


class TestFrictionLine:
    """Hughes skin friction on the measured hull versus the basic hull's flat coefficient."""

    def test_basic_is_flat(self) -> None:
        """The basic hull's friction coefficient is the old constant at any speed."""
        assert BasicHullModel(BASE)._friction_coefficient(1.5, BASE["L"]) == 0.004

    def test_falls_with_speed(self) -> None:
        """Cf drops as Reynolds number rises; a constant cannot do that."""
        hull = MeasuredHullModel(MEASURED)
        coeffs = [hull._friction_coefficient(u, BASE["L"]) for u in (0.5, 1.0, 2.0)]
        assert coeffs[0] > coeffs[1] > coeffs[2]

    def test_falls_back_when_barely_moving(self) -> None:
        """The line is invalid at tiny Reynolds numbers, so it must not be used."""
        hull = MeasuredHullModel(MEASURED)
        assert hull._friction_coefficient(1e-6, BASE["L"]) == 0.004


class TestHullSelection:
    """`basic` estimates from L, B, T; `measured` needs the measured geometry; neither guesses."""

    def test_basic_refuses_measured_keys(self) -> None:
        """Sections or a friction model handed to the basic hull is an error, not a silent upgrade."""
        with pytest.raises(ValueError, match="measured"):
            BasicHullModel({**BASE, "sections": SECTIONS})
        with pytest.raises(ValueError, match="measured"):
            BasicHullModel({**BASE, "friction_model": "hughes"})

    def test_measured_requires_its_geometry(self) -> None:
        """No wetted surface, or no sections, is a config error pointing at `basic`."""
        with pytest.raises(ValueError, match="wetted_surface_m2"):
            MeasuredHullModel({**BASE, "sections": SECTIONS})
        with pytest.raises(ValueError, match="sections"):
            MeasuredHullModel({**BASE, "wetted_surface_m2": 0.571})

    def test_measured_refuses_a_friction_model_key(self) -> None:
        """The measured hull is always Hughes; a key that pretends otherwise is refused."""
        with pytest.raises(ValueError, match="friction_model"):
            MeasuredHullModel({**MEASURED, "friction_model": "flat"})

    def test_c_wave_is_a_coefficient_on_both(self) -> None:
        """Wave-making is a resistance term either hull may carry, not a model choice."""
        assert BasicHullModel({**BASE, "c_wave": 0.01}).compute(make_state(u=2.0), TFTree2D())[0] < surge({}, 2.0)
        assert MeasuredHullModel({**MEASURED, "c_wave": 0.01})._residuary_resistance(2.0, 1000.0, 1.372, 0.11) < 0.0

    def test_hub_wires_both_by_model_type(self) -> None:
        """The hub's registry exposes both hulls, and each shipped config gets the one it names."""
        from sailbench.sim.sailboat_hub import HULL_MODELS, SailboatHub

        assert HULL_MODELS["basic"] is BasicHullModel
        assert HULL_MODELS["measured"] is MeasuredHullModel
        assert type(SailboatHub("basic_sailbot.yaml").hull) is BasicHullModel
        assert type(SailboatHub("flingo_floty.yaml").hull) is MeasuredHullModel


class TestWettedSurface:
    """A measured hull area should override the 1.7*L*(B+T) approximation."""

    def test_measured_area_scales_friction(self) -> None:
        """Friction drag is proportional to wetted area."""
        approx = 1.7 * BASE["L"] * (BASE["B"] + BASE["T"])
        half = surge({"wetted_surface_m2": approx / 2.0}, 1.5)
        assert math.isclose(half, surge({}, 1.5) / 2.0, rel_tol=1e-9)

    def test_absent_key_uses_the_approximation(self) -> None:
        """Without the key nothing changes."""
        approx = 1.7 * BASE["L"] * (BASE["B"] + BASE["T"])
        assert math.isclose(surge({"wetted_surface_m2": approx}, 1.5), surge({}, 1.5), rel_tol=1e-9)


class TestYawDamping:
    """Strip theory over the hull, and the coefficient it exposes."""

    def yaw_moment(self, params: dict, r: float) -> float:
        """Yaw moment from a hull built with `params` at yaw rate `r`."""
        return float(BasicHullModel({**BASE, **params}).compute(make_state(r=r), TFTree2D())[2])

    def test_matches_the_strip_theory_integral(self) -> None:
        """k_r = rho * Cd * T * L^4 / 64, from integrating x^2|x| over the hull."""
        expected = -1000.0 * 1.0 * BASE["T"] * BASE["L"] ** 4 / 64.0
        assert self.yaw_moment({}, 1.0) == pytest.approx(expected)

    def test_opposes_rotation_both_ways(self) -> None:
        """Damping resists whichever way the boat is turning."""
        assert self.yaw_moment({}, 1.0) < 0.0
        assert self.yaw_moment({}, -1.0) > 0.0

    def test_is_quadratic_in_yaw_rate(self) -> None:
        """r|r| means doubling the rate quadruples the moment."""
        assert self.yaw_moment({}, 2.0) == pytest.approx(4.0 * self.yaw_moment({}, 1.0))

    def test_zero_at_rest(self) -> None:
        """No rotation, no damping."""
        assert self.yaw_moment({}, 0.0) == pytest.approx(0.0)

    def test_scales_with_the_fourth_power_of_length(self) -> None:
        """The lever arm enters twice and the strip speed twice."""
        short = self.yaw_moment({"L": 1.0}, 1.0)
        long_ = self.yaw_moment({"L": 2.0}, 1.0)
        assert long_ == pytest.approx(16.0 * short)

    def test_yaw_coefficient_can_be_calibrated_alone(self) -> None:
        """Turning can be tuned without disturbing sway, which shares cross_flow_cd."""
        tuned = {"yaw_damping_cd": 4.0}
        assert self.yaw_moment(tuned, 1.0) == pytest.approx(4.0 * self.yaw_moment({}, 1.0))
        # sway untouched
        base_sway = float(BasicHullModel(BASE).compute(make_state(v=1.0), TFTree2D())[1])
        tuned_sway = float(BasicHullModel({**BASE, **tuned}).compute(make_state(v=1.0), TFTree2D())[1])
        assert tuned_sway == pytest.approx(base_sway)

    def test_defaults_to_the_cross_flow_coefficient(self) -> None:
        """Unset, strip theory's own answer: the two coefficients are equal."""
        assert self.yaw_moment({"cross_flow_cd": 2.0}, 1.0) == pytest.approx(
            self.yaw_moment({"cross_flow_cd": 2.0, "yaw_damping_cd": 2.0}, 1.0)
        )
