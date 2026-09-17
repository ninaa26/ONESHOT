import numpy as np
import pytest

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D


def make_state(u: float = 0.0, v: float = 0.0, r: float = 0.0, psi: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(
        np.array(
            [
                0.0,         # x
                0.0,         # y
                np.cos(psi), # cos(psi)
                np.sin(psi), # sin(psi)
                u,
                v,
                r,
            ]
        )
    )


class TestSail:
    """Sail model physics tests."""

    def test_zero_wind_zero_force(self, sail: HybridSail, tf_tree: TFTree2D) -> None:
        """Sail should produce zero force with zero wind."""
        sail.p["wind_speed"] = 0.0
        state = make_state()

        fx, fy = sail.compute(state, tf_tree)

        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_headwind_produces_drag(self, sail: HybridSail, tf_tree: TFTree2D) -> None:
        """Sail should produce drag with headwind aligned to boat."""
        sail.p["wind_speed"] = 10.0
        # wind_dir_deg = direction wind blows TO; 180° = west = headwind for boat facing east
        sail.p["wind_dir_deg"] = 180.0
        state = make_state()

        fx, fy = sail.compute(state, tf_tree)
        assert fx < -1.0  # Drag should be negative (oppose forward)

    def test_beam_wind_produces_lift(self, sail: HybridSail, tf_tree: TFTree2D) -> None:
        """Sail should produce lateral force with beam wind."""
        sail.p["wind_speed"] = 10.0
        sail.p["wind_dir_deg"] = 90.0
        state = make_state()

        fx, fy = sail.compute(state, tf_tree)
        assert np.hypot(fx, fy) > 1.0  # Significant total force for beam wind

    @pytest.mark.parametrize(
        ("wind_dir_deg", "u", "v"),
        [
            (45.0, 0.0, 0.0),
            (135.0, 0.0, 0.0),
            (90.0, 2.0, 0.0),
        ],
    )
    def test_wind_angles_produce_nonzero_force(
        self, sail: HybridSail, tf_tree: TFTree2D, wind_dir_deg: float, u: float, v: float
    ) -> None:
        """Sail should produce nonzero force for various wind and boat states."""
        sail.p["wind_speed"] = 10.0
        sail.p["wind_dir_deg"] = wind_dir_deg
        state = make_state(u=u, v=v)

        fx, fy = sail.compute(state, tf_tree)

        # Apparent wind should be nonzero in these cases
        assert abs(fx) > 0.1 or abs(fy) > 0.1

    def test_forward_motion_reduces_apparent_wind(self, sail: HybridSail, tf_tree: TFTree2D) -> None:
        """Boat forward speed should reduce apparent wind and sail force."""
        sail.p["wind_speed"] = 10.0
        # wind_dir_deg=0 = wind blows east = tailwind; sail force decreases as boat speeds up
        sail.p["wind_dir_deg"] = 0.0
        state_still = make_state()
        state_moving = make_state(u=8.0)

        fx0, fy0 = sail.compute(state_still, tf_tree)
        fx1, fy1 = sail.compute(state_moving, tf_tree)

        assert abs(fx1) < abs(fx0)
        assert abs(fy1) < abs(fy0)

    def test_sail_angle_changes_force(self, sail: HybridSail, tf_tree: TFTree2D) -> None:
        """Changing sail angle should change force direction."""
        sail.p["wind_speed"] = 10.0
        sail.p["wind_dir_deg"] = 5.0  # Moderate angle so aoas stay in polar range [-25,25]
        state = make_state()

        # Sail at 0°
        fx0, fy0 = sail.compute(state, tf_tree)

        # Update sail frame to 30°
        c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
        tf_tree.add_frame(name="sail", parent="boat", transform=Transform2D(0.0, 0.0, c, s))
        fx1, fy1 = sail.compute(state, tf_tree)

        assert not np.isclose(fx0, fx1)
        assert not np.isclose(fy0, fy1)


def set_sail_angle(tf_tree: TFTree2D, angle_deg: float) -> None:
    """Rotate the sail frame relative to the boat (positive = boom swings to starboard)."""
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    tf_tree.add_frame(name="sail", parent="boat", transform=Transform2D(0.0, 0.0, c, s))


class TestBasicSail:
    """NeuralFoil sail: angle of attack must be measured from the chord, not the mast."""

    def test_aoa_measured_from_chord(self, basic_sail: BasicSail, tf_tree: TFTree2D) -> None:
        """Close-hauled with the boom sheeted in, the foil should see a small AoA, not ~180°."""
        basic_sail.p["wind_speed"] = 5.0
        basic_sail.p["wind_dir_deg"] = 225.0  # blows toward SW: wind from ahead-and-port for a boat facing east
        set_sail_angle(tf_tree, 30.0)  # boom 30° to starboard
        fed: list[float] = []
        orig = basic_sail.cl_cd
        basic_sail.cl_cd = lambda alpha_rad, re: fed.append(alpha_rad) or orig(alpha_rad, re)
        state = make_state()

        basic_sail.compute(state, tf_tree)

        # Apparent wind is at -135° in the boat frame, -165° in the sail frame;
        # relative to the chord (sail -x) that is a 15° angle of attack.
        assert len(fed) == 1
        assert np.isclose(np.degrees(fed[0]), 15.0, atol=0.5)

    def test_beam_reach_drives_forward(self, basic_sail: BasicSail, tf_tree: TFTree2D) -> None:
        """Sail force on a beam reach should push the boat forward and to leeward."""
        basic_sail.p["wind_speed"] = 5.0
        basic_sail.p["wind_dir_deg"] = 90.0  # blows toward north: wind from starboard for a boat facing east
        set_sail_angle(tf_tree, -45.0)  # boom eased 45° to port (leeward)
        state = make_state()

        fx, fy = basic_sail.compute(state, tf_tree)

        assert fx > 1.0  # drive
        assert fy > 1.0  # heeling/side force, to leeward (+y = port)

    def test_luff_zeroes_lift_below_threshold(self, basic_sail: BasicSail, tf_tree: TFTree2D) -> None:
        """Below luff_deg the sail flogs: only drag, along the apparent wind, remains."""
        basic_sail.p["wind_speed"] = 5.0
        basic_sail.p["wind_dir_deg"] = 180.0  # headwind for a boat facing east
        basic_sail.p["luff_deg"] = 14.0
        basic_sail.p["luff_ramp_deg"] = 4.0
        state = make_state()

        set_sail_angle(tf_tree, 10.0)  # 10° AoA: inside the luff band
        fx_luff, fy_luff = basic_sail.compute(state, tf_tree)
        set_sail_angle(tf_tree, 25.0)  # 25° AoA: fully powered
        fx_full, fy_full = basic_sail.compute(state, tf_tree)

        # Luffing: pure drag straight downwind, no lateral lift component.
        assert fx_luff < 0.0
        assert abs(fy_luff) < 1e-9
        # Powered: a clear lateral lift component appears.
        assert abs(fy_full) > 1.0
