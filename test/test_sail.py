import numpy as np
import pytest

from sailbench.foils.basic_sail import BasicSail
from sailbench.models.model import State


def make_state(
    u: float = 0.0,
    v: float = 0.0,
    r: float = 0.0,
    psi: float = 0.0,
) -> State:
    """Generate a test boat state."""
    return State.from_array(
        np.array(
            [
                0.0,                 # x
                0.0,                 # y
                np.cos(psi),         # cos(psi)
                np.sin(psi),         # sin(psi)
                u,                   # surge
                v,                   # sway
                r,                   # yaw rate
            ]
        )
    )

class TestSail:
    """Sail model physics tests."""

    def test_zero_wind_zero_force(self, sail: BasicSail) -> None:
        """No wind should produce no force."""
        params = {
            "wind_speed": 0.0,
            "wind_dir_deg": 0.0,
            "area": 5.0,
        }
        sail.p = params
        state = make_state()

        fx, fy = sail.compute(state)

        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_headwind_produces_drag(self, sail: BasicSail) -> None:
        """Headwind aligned with boat should produce drag."""
        state = make_state()

        fx, fy = sail.compute(state, sail_angle=0.0)

        assert fx < -1.0        # Drag opposes forward direction
        assert abs(fy) < 1.0    # Minimal lift expected

    def test_beam_wind_produces_lift(self, sail: BasicSail) -> None:
        """Side wind should generate lateral force."""
        sail.p["wind_dir_deg"] = 90.0  # wind from +y

        state = make_state()
        fx, fy = sail.compute(state, sail_angle=0.0)

        assert abs(fy) > abs(fx)
        assert abs(fy) > 1.0

    @pytest.mark.parametrize(
        ("wind_dir", "expected_sign"),
        [
            (45.0, 1.0),
            (315.0, -1.0),
        ],
    )
    def test_lift_direction_changes_with_wind(
        self,
        sail: BasicSail,
        wind_dir: float,
        expected_sign: float,
    ) -> None:
        """Lift direction should flip with wind angle."""
        sail.p["wind_dir_deg"] = wind_dir
        state = make_state()

        fx, fy = sail.compute(state, sail_angle=0.0)

        assert np.sign(fy) == expected_sign

    def test_forward_motion_reduces_apparent_wind(self, sail: BasicSail) -> None:
        """Boat forward speed should reduce sail force."""
        state_still = make_state()
        state_moving = make_state(u=4.0)

        fx0, fy0 = sail.compute(state_still)
        fx1, fy1 = sail.compute(state_moving)

        assert abs(fx1) < abs(fx0)
        assert abs(fy1) < abs(fy0)

    def test_sail_angle_changes_force_direction(self, sail: BasicSail) -> None:
        """Changing sail angle should rotate force direction."""
        state = make_state()

        fx0, fy0 = sail.compute(state, sail_angle=0.0)
        fx1, fy1 = sail.compute(state, sail_angle=30.0)

        # Forces should not be identical
        assert not np.isclose(fx0, fx1)
        assert not np.isclose(fy0, fy1)
