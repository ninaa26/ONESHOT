"""ORCSail: the ORC VPP coefficient envelope, trim depowering and force split."""

import math

import numpy as np
import pytest

from sailbench.foils.orc_sail import MAIN_AWA_DEG, MAIN_CL, ORCSail
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

WIND_TO_DEG = 90.0


def make_sail(**overrides: float) -> ORCSail:
    """Build an ORC sail with Flingo's measured rig."""
    return ORCSail({
        "area": 1.971, "heff": 2.592, "wind_speed": 5.0,
        "wind_dir_deg": WIND_TO_DEG, "rho_air": 1.225, **overrides,
    })


def make_state(u: float = 1.7, psi: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, np.cos(psi), np.sin(psi), u, 0.0, 0.0]))


def tree(boom_rad: float, psi: float = 0.0) -> TFTree2D:
    """Transform tree with the boat on a heading and the boom trimmed."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world",
                 transform=Transform2D(x=0.0, y=0.0, c=math.cos(psi), s=math.sin(psi)))
    tf.add_frame(name="sail", parent="boat",
                 transform=Transform2D(x=0.0, y=0.0, c=math.cos(boom_rad), s=math.sin(boom_rad)))
    return tf


def beat(twa_deg: float) -> float:
    """Heading `twa_deg` off the true wind, wind on the port side."""
    return math.radians(WIND_TO_DEG) - math.pi + math.radians(twa_deg)


class TestEnvelope:
    """ORC Table 5.1 lookup."""

    def test_no_lift_head_to_wind(self) -> None:
        """CL is zero at 0 degrees, so luffing is built into the table."""
        assert make_sail().envelope(0.0)[0] == 0.0

    def test_peak_lift_is_upwind(self) -> None:
        """A sail makes its most lift on a beat or close reach, not running."""
        sail = make_sail()
        assert sail.envelope(45.0)[0] > sail.envelope(135.0)[0]
        assert sail.envelope(28.0)[0] > sail.envelope(150.0)[0]

    def test_parasitic_drag_grows_downwind(self) -> None:
        """CD0 rises steeply as the sail squares off to the wind."""
        sail = make_sail()
        assert sail.envelope(150.0)[1] > sail.envelope(90.0)[1] > sail.envelope(28.0)[1]

    def test_matches_the_table_at_its_own_nodes(self) -> None:
        """Interpolation must reproduce the published values exactly."""
        sail = make_sail()
        for awa, cl in zip(MAIN_AWA_DEG, MAIN_CL, strict=True):
            assert sail.envelope(float(awa))[0] == pytest.approx(float(cl))

    def test_symmetric_in_wind_side(self) -> None:
        """The envelope depends on the magnitude of the apparent wind angle."""
        sail = make_sail()
        assert sail.envelope(-45.0) == sail.envelope(45.0)

    def test_stays_within_the_table_beyond_its_ends(self) -> None:
        """Past 180 degrees the lookup clamps rather than extrapolating."""
        sail = make_sail()
        assert sail.envelope(250.0) == sail.envelope(180.0)


class TestTrim:
    """The ORC `flat` depowering factor."""

    def test_luffing_at_zero_incidence(self) -> None:
        """An over-eased sail carries no lift at all."""
        assert make_sail().flat_from_trim(0.0) == 0.0

    def test_full_power_at_optimum(self) -> None:
        """flat reaches 1 at alpha_opt."""
        assert make_sail().flat_from_trim(math.radians(22.0)) == pytest.approx(1.0)

    def test_rises_monotonically_up_to_the_optimum(self) -> None:
        """Sheeting in from luffing must never lose power."""
        sail = make_sail()
        flats = [sail.flat_from_trim(math.radians(a)) for a in (0, 5, 10, 15, 22)]
        assert flats == sorted(flats)

    def test_over_sheeting_depowers_towards_the_floor(self) -> None:
        """Past the optimum the sail stalls back towards flat_stall_floor."""
        sail = make_sail()
        assert sail.flat_from_trim(math.radians(22.0)) > sail.flat_from_trim(math.radians(45.0))
        assert sail.flat_from_trim(math.radians(90.0)) == pytest.approx(sail.flat_floor)


class TestForces:
    """Drive/heel resolution in the boat frame."""

    def test_no_apparent_wind_no_force(self) -> None:
        """Zero wind with the boat stopped means no force."""
        f = make_sail(wind_speed=0.0).compute(make_state(u=0.0), tree(0.0))
        assert np.allclose(f, 0.0)

    @pytest.mark.parametrize("twa_deg", [35, 45, 60, 90, 135])
    def test_drives_the_boat_forward(self, twa_deg: float) -> None:
        """Correctly trimmed, the sail pushes the boat forwards."""
        sail = make_sail()
        psi = beat(twa_deg)
        best = max(sail.compute(make_state(psi=psi), tree(math.radians(t), psi))[0]
                   for t in range(5, 90, 5))
        assert best > 0.0

    def test_heel_force_opposes_the_wind_side(self) -> None:
        """Wind from one side pushes the boat toward the other."""
        sail = make_sail()
        psi = beat(45.0)
        assert sail.compute(make_state(psi=psi), tree(math.radians(20.0), psi))[1] > 0.0
        mirrored = math.radians(WIND_TO_DEG) + math.pi - math.radians(45.0)
        assert sail.compute(make_state(psi=mirrored), tree(math.radians(-20.0), mirrored))[1] < 0.0

    def test_force_scales_with_air_density(self) -> None:
        """Force is proportional to rho_air."""
        psi = beat(45.0)
        args = (make_state(psi=psi), tree(math.radians(20.0), psi))
        light = make_sail(rho_air=1.0).compute(*args)
        heavy = make_sail(rho_air=2.0).compute(*args)
        assert heavy[0] == pytest.approx(2.0 * light[0])

    def test_taller_rig_pays_less_induced_drag(self) -> None:
        """Induced drag goes as area / (pi * heff^2), so height buys efficiency."""
        psi = beat(40.0)
        args = (make_state(psi=psi), tree(math.radians(18.0), psi))
        short = make_sail(heff=1.5).compute(*args)[0]
        tall = make_sail(heff=3.5).compute(*args)[0]
        assert tall > short

    def test_luffed_sail_makes_no_lift(self) -> None:
        """A boom eased past the apparent wind angle produces no lift."""
        sail = make_sail()
        psi = beat(30.0)
        sail.compute(make_state(psi=psi), tree(math.radians(80.0), psi))
        assert sail.last_flat == 0.0
        assert sail.last_cl == 0.0
