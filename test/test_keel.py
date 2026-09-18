import numpy as np
import pytest

from sailbench.foils.basic_keel import BasicKeel, FiniteSpanKeel
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


class TestKeelSelection:
    """`basic` is the raw 2-D section and `finite_span` the lifting-line blend; neither guesses."""

    BASE = {"airfoil_name": "NACA0010", "alpha_min": -179, "alpha_max": 179, "res": [2.6e5], "area": 0.1225}

    def test_basic_refuses_finite_span_keys(self) -> None:
        """A span handed to the 2-D model is an error, not a silently applied correction."""
        with pytest.raises(ValueError, match="finite_span"):
            BasicKeel({**self.BASE, "span": 0.7})
        with pytest.raises(ValueError, match="finite_span"):
            BasicKeel({**self.BASE, "alpha_sep_deg": 25.0})

    def test_finite_span_requires_its_keys(self) -> None:
        """No aspect ratio, or no separation angle, is a config error pointing at `basic`."""
        with pytest.raises(ValueError, match="span"):
            FiniteSpanKeel({**self.BASE, "alpha_sep_deg": 25.0})
        with pytest.raises(ValueError, match="alpha_sep_deg"):
            FiniteSpanKeel({**self.BASE, "span": 0.7})

    def test_hub_wires_both_by_model_type(self) -> None:
        """Both keels register themselves, and each shipped config gets the one it names."""
        from sailbench.sim.registry import registered
        from sailbench.sim.sailboat_hub import SailboatHub

        keels = registered("keel")
        assert keels["basic"] is BasicKeel
        assert keels["finite_span"] is FiniteSpanKeel
        assert type(SailboatHub("basic_sailbot.yaml").keel) is BasicKeel
        assert type(SailboatHub("flingo_floty.yaml").keel) is FiniteSpanKeel

    def test_finite_span_pays_for_its_lift(self) -> None:
        """Same section, same leeway: the finite-span keel makes less lift and more drag."""
        state = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1.5, 0.15, 0.0]))
        tf = TFTree2D()
        basic = BasicKeel(self.BASE).compute(state, tf)
        finite = FiniteSpanKeel({**self.BASE, "span": 0.7, "end_plate_factor": 2.0, "alpha_sep_deg": 25.0}).compute(
            state, tf
        )
        assert abs(finite[1]) < abs(basic[1])  # less side force
        assert finite[0] < basic[0]  # more drag (both negative, along -x)
