import numpy as np
import pytest

from sailbench.foils.basic_rudder import BasicRudder
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


def make_state(u: float = 0.0, v: float = 0.0, r: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(
        np.array(
            [
                0.0,  # x
                0.0,  # y
                1.0,  # cos(psi)
                0.0,  # sin(psi)
                u,
                v,
                r,
            ]
        )
    )


class TestRudder:
    """Rudder model physics tests."""

    def test_zero_flow_zero_force(self, rudder: BasicRudder, tf_tree: TFTree2D) -> None:
        """Rudder should produce zero force with zero velocity."""
        state = make_state()

        fx, fy = rudder.compute(state,tf_tree)

        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_forward_flow(self, rudder: BasicRudder, tf_tree: TFTree2D) -> None:
        """Rudder should produce drag in forward flow."""
        u, v = 5.0, 0.0
        state = make_state(u=u, v=v)

        fx, fy = rudder.compute(state, tf_tree)

        # Symmetric foil at zero incidence makes no lift
        assert abs(fy) < 0.5

        # The rudder is a passive foil: it can only remove energy from the boat
        assert fx * u + fy * v < 0

    def test_angle90_print_test(self, rudder:BasicRudder, tf_tree: TFTree2D) -> None:
        """Rudder set at 90 degrees."""
        state = make_state(u = 5.0)
        fx, fy = rudder.compute(state, tf_tree)
        #print ("fx: " + str(fx) + ", fy: " + str(fy))

        fx90, fy90 = rudder.compute(state, tf_tree)
        #print ("fx90: " + str(fx90) + ", fy90: " + str(fy90))

    @pytest.mark.parametrize(
        ("u", "v"),
        [
            (1, 0.2),
            (1, -0.2),
            # (-5.0, 5.0),
            # (-5.0, -5.0),
        ],
    )
    def test_quadrant_flow(self, rudder: BasicRudder, u: float, v: float, tf_tree: TFTree2D) -> None:
        """Rudder lateral force should oppose lateral flow direction."""
        state = make_state(u=u, v=v)

        fx, fy = rudder.compute(state, tf_tree)

        # Lift dominates drag for mostly forward flow at a small leeway angle
        assert abs(fx) < abs(fy)

        # Lift should oppose lateral velocity (restoring force)
        if abs(v) > 1e-6:
            assert np.sign(fy) == -np.sign(v)

        # The rudder is a passive foil: it can only remove energy from the boat.
        # Its lift is perpendicular to the flow, so at leeway it has a forward
        # body-x component; what must stay negative is the net mechanical power.
        assert fx * u + fy * v < 0
