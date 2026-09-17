import numpy as np
import pytest

from sailbench.foils.basic_rudder import BasicRudder, FiniteSpanRudder
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

    @pytest.mark.parametrize("delta_deg", [5.0, 10.0, -10.0])
    def test_deflection_steers_without_braking(
        self, rudder: BasicRudder, tf_tree: TFTree2D, delta_deg: float
    ) -> None:
        """Deflecting the rudder in straight flow should steer, not brake.

        With the boat moving straight ahead the flow is exactly along boat -x, so
        lift is purely lateral and the only longitudinal force is profile drag,
        a few percent of lift for pre-stall deflections. Resolving the forces in
        a frame built from the wrong angle put a lift component along -x instead,
        and small corrections cost 17x their real drag.
        """
        d = np.radians(delta_deg)
        tf_tree.add_frame(name="rudder", parent="boat", transform=Transform2D(0.0, 0.0, np.cos(d), np.sin(d)))
        state = make_state(u=1.0)

        fx, fy = rudder.compute(state, tf_tree)

        assert np.sign(fy) == np.sign(delta_deg)
        assert fx < 0.0
        assert abs(fx) < 0.05 * abs(fy)

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
    def test_never_adds_energy(
        self, rudder: BasicRudder, tf_tree: TFTree2D, u: float, v: float, r: float, delta_deg: float
    ) -> None:
        """In still water a passive foil can only remove kinetic energy: F . v_local <= 0.

        Covers deflection and yaw rate together, which the straight-flow tests do not.
        """
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


class TestRudderSelection:
    """`basic` is the clamped 2-D section and `finite_span` the lifting-line blend; neither guesses."""

    BASE = {"airfoil_name": "NACA0012", "alpha_min": -179, "alpha_max": 179, "res": [1.2e5], "area": 0.0462}

    def test_basic_refuses_finite_span_keys(self) -> None:
        """A span handed to the 2-D model is an error, not a half-applied correction."""
        with pytest.raises(ValueError, match="finite_span"):
            BasicRudder({**self.BASE, "span": 0.478})
        with pytest.raises(ValueError, match="finite_span"):
            BasicRudder({**self.BASE, "alpha_sep_deg": 25.0})

    def test_finite_span_requires_its_keys(self) -> None:
        """No aspect ratio, or no separation angle, is a config error pointing at `basic`."""
        with pytest.raises(ValueError, match="span"):
            FiniteSpanRudder({**self.BASE, "alpha_sep_deg": 25.0})
        with pytest.raises(ValueError, match="alpha_sep_deg"):
            FiniteSpanRudder({**self.BASE, "span": 0.478})

    def test_finite_span_refuses_clamps(self) -> None:
        """Clamping a blended coefficient puts back the kink the blend removes."""
        for key in ("aoa_limit_deg", "cl_max", "cd_max"):
            with pytest.raises(ValueError, match=key):
                FiniteSpanRudder({**self.BASE, "span": 0.478, "alpha_sep_deg": 25.0, key: 1.0})

    def test_hub_wires_both_by_model_type(self) -> None:
        """The hub's registry exposes both rudders, and each shipped config gets the one it names."""
        from sailbench.sim.sailboat_hub import RUDDER_MODELS, SailboatHub

        assert RUDDER_MODELS["basic"] is BasicRudder
        assert RUDDER_MODELS["finite_span"] is FiniteSpanRudder
        assert type(SailboatHub("basic_sailbot.yaml").rudder) is BasicRudder
        assert type(SailboatHub("flingo_floty.yaml").rudder) is FiniteSpanRudder

    def test_finite_span_is_a_rudder(self) -> None:
        """Only the coefficient lookup differs; the force resolution is shared."""
        assert isinstance(FiniteSpanRudder({**self.BASE, "span": 0.478, "alpha_sep_deg": 25.0}), BasicRudder)
