import numpy as np
import pytest

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
        # Dead astern with the sail centred, lateral force is identically zero at
        # both speeds, so this can only be a non-strict bound.
        assert abs(fy1) <= abs(fy0)
        assert np.hypot(fx1, fy1) < np.hypot(fx0, fy0)

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
