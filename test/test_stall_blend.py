"""Post-stall blending towards a flat plate, shared by every Foil."""

import math

import numpy as np
import pytest

from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class _Foil(Foil):
    """Concrete Foil; the blend under test is independent of compute()."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Not exercised."""
        raise NotImplementedError


def foil(**params: float) -> _Foil:
    """Build a foil with the given stall parameters."""
    return _Foil({"airfoil_name": "NACA0012", **params})


CN = 2.0


class TestSeparationFraction:
    """How the blend moves from attached flow to a plate."""

    def test_inert_when_unconfigured(self) -> None:
        """Without alpha_sep_deg the coefficients pass straight through."""
        f = foil()
        assert f.stall_blending is False
        assert f.blend_stall(math.radians(45.0), 0.8, 0.05) == (0.8, 0.05)

    def test_attached_at_zero_incidence(self) -> None:
        """At zero angle of attack nothing is separated."""
        cl, cd = foil(alpha_sep_deg=25.0).blend_stall(0.0, 0.9, 0.02)
        assert cl == pytest.approx(0.9)
        assert cd == pytest.approx(0.02)

    def test_barely_touches_small_angles(self) -> None:
        """Below stall the section polar should still dominate."""
        cl, _ = foil(alpha_sep_deg=25.0).blend_stall(math.radians(5.0), 0.5, 0.02)
        assert cl == pytest.approx(0.5, rel=0.05)

    def test_becomes_a_flat_plate_well_past_stall(self) -> None:
        """At 90 degrees the foil is a plate: no lift, normal drag only."""
        cl, cd = foil(alpha_sep_deg=25.0).blend_stall(math.radians(90.0), 0.5, 0.3)
        # Separation is asymptotic: at 90 deg with alpha_sep 25 the attached
        # fraction is exp(-12.96), about 2e-6, so this is "a plate" to well
        # inside any physical tolerance rather than exactly.
        assert cl == pytest.approx(0.0, abs=1e-4)
        assert cd == pytest.approx(CN, abs=1e-4)

    def test_is_continuous_across_stall(self) -> None:
        """No jumps anywhere -- the whole point of blending over clamping."""
        f = foil(alpha_sep_deg=25.0)
        angles = np.radians(np.arange(0.0, 90.0, 0.25))
        cls = [f.blend_stall(a, 0.8, 0.05)[0] for a in angles]
        steps = np.abs(np.diff(cls))
        assert steps.max() < 0.02, "coefficient jumps across the blend"

    def test_separation_grows_with_angle(self) -> None:
        """More incidence means more separated flow, never less."""
        f = foil(alpha_sep_deg=25.0)
        drags = [f.blend_stall(math.radians(a), 0.0, 0.0)[1] for a in (10, 20, 40, 60, 90)]
        assert drags == sorted(drags)

    def test_a_later_separation_angle_delays_the_blend(self) -> None:
        """alpha_sep sets where stall begins to bite."""
        early = foil(alpha_sep_deg=15.0).blend_stall(math.radians(20.0), 1.0, 0.02)[1]
        late = foil(alpha_sep_deg=35.0).blend_stall(math.radians(20.0), 1.0, 0.02)[1]
        assert early > late


class TestPlateLimit:
    """The separated end of the blend."""

    @pytest.mark.parametrize("alpha_deg", [30.0, 45.0, 60.0, 90.0])
    def test_matches_the_flat_plate_when_fully_separated(self, alpha_deg: float) -> None:
        """Forcing full separation must give exactly cn*sin*cos and cn*sin^2."""
        a = math.radians(alpha_deg)
        # alpha_sep tiny => separated fraction is 1 to machine precision
        cl, cd = foil(alpha_sep_deg=0.5).blend_stall(a, 0.0, 0.0)
        assert cl == pytest.approx(CN * math.sin(a) * math.cos(a))
        assert cd == pytest.approx(CN * math.sin(a) ** 2)

    def test_drag_is_never_negative(self) -> None:
        """A plate cannot pull the foil forwards."""
        f = foil(alpha_sep_deg=25.0)
        for a in np.radians(np.arange(-90.0, 90.0, 1.0)):
            assert f.blend_stall(float(a), 0.0, 0.01)[1] >= 0.0

    def test_is_antisymmetric_in_lift(self) -> None:
        """Mirroring the angle of attack mirrors the lift."""
        f = foil(alpha_sep_deg=25.0)
        up = f.blend_stall(math.radians(40.0), 0.7, 0.2)
        down = f.blend_stall(math.radians(-40.0), -0.7, 0.2)
        assert up[0] == pytest.approx(-down[0])
        assert up[1] == pytest.approx(down[1])

    def test_normal_force_coefficient_is_configurable(self) -> None:
        """cn_plate scales the separated end."""
        a = math.radians(90.0)
        assert foil(alpha_sep_deg=0.5, cn_plate=1.0).blend_stall(a, 0.0, 0.0)[1] == pytest.approx(1.0)


class TestPlateNormalForce:
    """CN of the fully separated foil depends on aspect ratio, not on a constant.

    2.0 is the two-dimensional value. A real plate lets flow escape round its
    tips, so Hoerner measures 1.18 at AR 1 and 1.50 at AR 20; the keel and
    rudder were charged 2.0 regardless, over-predicting every stalled force.
    """

    def test_defaults_to_two_d_without_a_span(self) -> None:
        """A foil that has not stated its span gets the only defensible default."""
        assert foil(alpha_sep_deg=25.0).plate_normal_force() == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("ar", "expected"),
        [(1.0, 1.128), (5.0, 1.20), (8.0, 1.254), (10.0, 1.29), (20.0, 1.47)],
    )
    def test_follows_hoerner(self, ar: float, expected: float) -> None:
        """CN = 1.11 + 0.018*AR, the Viterna fit to Hoerner's plate data."""
        f = foil(alpha_sep_deg=25.0, effective_aspect_ratio=ar)
        assert f.plate_normal_force() == pytest.approx(expected, abs=0.02)

    def test_never_exceeds_the_two_d_limit(self) -> None:
        """No aspect ratio makes a plate better than an infinite one."""
        assert foil(alpha_sep_deg=25.0, effective_aspect_ratio=500.0).plate_normal_force() <= 2.0

    def test_explicit_cn_plate_still_wins(self) -> None:
        """A config that states the number keeps control of it."""
        f = foil(alpha_sep_deg=25.0, effective_aspect_ratio=8.0, cn_plate=1.7)
        assert f.plate_normal_force() == pytest.approx(1.7)

    def test_broadside_drag_equals_the_normal_force(self) -> None:
        """At 90 degrees the plate is all drag, so CD must be exactly CN."""
        f = foil(alpha_sep_deg=0.5, effective_aspect_ratio=5.0)
        assert f.blend_stall(math.radians(90.0), 0.0, 0.0)[1] == pytest.approx(1.20, abs=0.01)

    def test_post_stall_lift_stays_under_the_attached_peak(self) -> None:
        """The bug this fixes: a stalled foil cannot out-lift its own attached peak.

        With CN pinned at 2.0 the rudder's blended CL reached 0.99 at 45 deg
        against an attached peak of 0.66 at 10 deg -- a second, higher peak in
        deep stall, which no foil does.
        """
        f = foil(alpha_sep_deg=25.0, span=0.478, area=0.0462)  # the Flingo rudder, AR 4.9
        attached_peak = max(f.blend_stall(float(a), *f.cl_cd(float(a), 1.2e5))[0]
                            for a in np.radians(np.arange(0.0, 13.0, 0.5)))
        stalled_peak = max(f.blend_stall(float(a), *f.cl_cd(float(a), 1.2e5))[0]
                           for a in np.radians(np.arange(20.0, 90.0, 0.5)))
        assert stalled_peak < attached_peak
