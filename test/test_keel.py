import numpy as np
import pytest

from sailbench.foils.basic_keel import BasicKeel
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


class TestKeel:
    """Keel model physics tests."""

    def test_zero_flow_zero_force(self, keel: BasicKeel, tf_tree: TFTree2D) -> None:
        """Keel should produce zero force with zero velocity."""
        state = make_state()

        fx, fy = keel.compute(state, tf_tree)

        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_forward_flow(self, keel: BasicKeel, tf_tree: TFTree2D) -> None:
        """Keel should produce drag in forward flow."""
        u, v = 5.0, 0.0
        state = make_state(u=u, v=v)

        fx, fy = keel.compute(state, tf_tree)

        # Symmetric foil at zero incidence makes no lift
        assert abs(fy) < 0.5

        # The keel is a passive foil: it can only remove energy from the boat
        assert fx * u + fy * v < 0

    @pytest.mark.parametrize(
        ("u", "v"),
        [
            (1, 0.2),
            (1, -0.2),
        ],
    )
    def test_quadrant_flow(self, keel: BasicKeel, u: float, v: float, tf_tree: TFTree2D) -> None:
        """Keel lateral force should oppose lateral flow direction."""
        state = make_state(u=u, v=v)

        fx, fy = keel.compute(state, tf_tree)

        # Lift dominates drag for mostly forward flow at a small leeway angle
        assert abs(fx) < abs(fy)

        # Lift should oppose lateral velocity (restoring force)
        if abs(v) > 1e-6:
            assert np.sign(fy) == -np.sign(v)

        # The keel is a passive foil: it can only remove energy from the boat.
        # Its lift is perpendicular to the flow, so at leeway it has a forward
        # body-x component; what must stay negative is the net mechanical power.
        assert fx * u + fy * v < 0
