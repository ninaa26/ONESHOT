import numpy as np
import pytest

from sailbench.foils.basic_rudder import BasicRudder
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D


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
    """Keel model physics tests."""

    def test_zero_flow_zero_force(self, rudder: BasicRudder, tf_tree: TFTree2D) -> None:
        """Rudder should produce zero force with zero velocity."""
        state = make_state()

        fx, fy = rudder.compute(state,tf_tree)

        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_forward_flow(self, rudder: BasicRudder, tf_tree: TFTree2D) -> None:
        """Rudder should produce drag (never thrust) in forward flow."""
        state = make_state(u=5.0)

        fx, fy = rudder.compute(state, tf_tree)
        assert fx < -1.0  # Drag should be negative in forward flow
        assert abs(fy) < 0.5  # Minimal lift expected in straight flow

    @pytest.mark.parametrize("delta_deg", [5.0, 10.0, -10.0])
    def test_deflection_gives_lift_with_profile_drag_only(
        self, rudder: BasicRudder, tf_tree: TFTree2D, delta_deg: float
    ) -> None:
        """Deflecting the rudder in straight flow should steer, not brake.

        With the boat moving straight ahead the flow is exactly along boat -x, so
        lift is purely lateral and the only longitudinal force is profile drag,
        which is a few percent of lift for pre-stall deflections.
        """
        d = np.radians(delta_deg)
        tf_tree.add_frame(name="rudder", parent="boat", transform=Transform2D(0.0, 0.0, np.cos(d), np.sin(d)))
        state = make_state(u=1.0)

        fx, fy = rudder.compute(state, tf_tree)

        assert np.sign(fy) == np.sign(delta_deg)
        assert fx < 0.0
        assert abs(fx) < 0.05 * abs(fy)  # Fx is profile drag only, not a lift component

    @pytest.mark.parametrize(
        ("u", "v", "r", "delta_deg"),
        [
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 10.0),
            (1.0, 0.2, 0.0, 0.0),
            (1.0, -0.2, 0.0, 15.0),
            (1.0, 0.0, 0.5, 0.0),
            (1.0, 0.1, 0.3, 20.0),
            (0.5, 0.0, -0.4, -30.0),
        ],
    )
    def test_rudder_never_adds_energy(
        self, rudder: BasicRudder, tf_tree: TFTree2D, u: float, v: float, r: float, delta_deg: float
    ) -> None:
        """In still water a passive foil can only remove kinetic energy: F . v_local <= 0."""
        d = np.radians(delta_deg)
        x_pos, y_pos = float(rudder.p.get("x_pos", 0.0)), float(rudder.p.get("y_pos", 0.0))
        tf_tree.add_frame(name="rudder", parent="boat", transform=Transform2D(x_pos, y_pos, np.cos(d), np.sin(d)))
        state = make_state(u=u, v=v, r=r)

        fx, fy = rudder.compute(state, tf_tree)

        u_local, v_local = u - r * y_pos, v + r * x_pos
        assert fx * u_local + fy * v_local <= 1e-9

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
        """Keel lateral force should oppose lateral flow direction."""
        state = make_state(u=u, v=v)

        fx, fy = rudder.compute(state,tf_tree)

        # Lift is perpendicular to the flow, so it has a forward component when
        # the boat slides sideways; the invariant is that the foil does no work.
        assert fx * u + fy * v <= 1e-9

        # Longitudinal force should be smaller than lift for mostly side-flow
        assert abs(fx) < abs(fy)

        # Lift should oppose lateral velocity (restoring force)
        if abs(v) > 1e-6:
            assert np.sign(fy) == -np.sign(v)
