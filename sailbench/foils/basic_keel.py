"""Keel foil models.

Two keels, picked by the keel section's ``model_type``:

``basic`` -- :class:`BasicKeel`
    The 2-D section polar as NeuralFoil returns it. Lift is free of induced
    drag, so leeway comes out near zero and the boat points unrealistically
    high.

``finite_span`` -- :class:`FiniteSpanKeel`
    Lifting-line correction from ``span`` (or ``effective_aspect_ratio``), so
    the fin pays induced drag for its side force, and a blend towards a flat
    plate past ``alpha_sep_deg`` so the polar means something at large leeway.

Each refuses the other's keys rather than half-applying them: a config says
which keel it means and gets exactly that one.
"""

from typing import Any

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


# Keys that only make sense for the finite-span model.
FINITE_SPAN_KEYS = ("span", "effective_aspect_ratio", "alpha_sep_deg")


class BasicKeel(Foil):
    """Keel on the raw 2-D section polar (``model_type: basic``)."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the keel model.

        Args:
            params (dict): Dictionary of parameters for the keel model.

        """
        super().__init__(params)
        self._check_keys()

    def _check_keys(self) -> None:
        """Reject keys that belong to the finite-span model."""
        stray = [k for k in FINITE_SPAN_KEYS if k in self.p]
        if stray:
            msg = (
                f"keel model_type: basic is a 2-D section and got {', '.join(stray)}; "
                "use model_type: finite_span for induced drag and stall blending"
            )
            raise ValueError(msg)

    def coefficients(self, aoa: float) -> tuple[float, float]:
        """Return the section's ``(cl, cd)`` at an angle of attack [rad]."""
        return self.cl_cd(aoa, re=self.get_reynolds())

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the lift and drag coefficients for the keel.

        Assume the keel is parallel to the axis of the boat.

        Args:
            state (State): Current boat state.
            tf_tree (TFTree2D): Current transform tree of the boat, used to get component positions.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within keel frame)

        """
        # Local inflow at the keel, including the yaw-rate contribution. Body
        # velocity at a point (x, y) is [u - r*y, v + r*x]; the rudder has always
        # done this, the keel did not. Sitting forward of the centre of rotation
        # it sees real sideslip whenever the boat turns, and ignoring that meant
        # the keel provided no yaw damping at all -- which the hull's damping
        # coefficient had been silently making up for.
        x_pos = float(self.p.get("x_pos", 0.0))
        y_pos = float(self.p.get("y_pos", 0.0))
        u_local = float(state.u - state.r * y_pos)
        v_local = float(state.v + state.r * x_pos)

        local_track = np.degrees(np.arctan2(v_local, u_local))
        # Coefficients are looked up at the track angle rather than the angle of
        # attack; the fluid frame's 180-degree flip below supplies the negation.
        aoa = local_track

        # get lift and drag coefficients
        cl, cd = self.coefficients(np.radians(aoa))
        # compute dynamic pressure
        rho = float(self.p.get("rho_water", 1000.0))  # kg/m^3
        v = float(np.hypot(u_local, v_local))
        q = 0.5 * rho * v**2
        # compute forces
        s = self.p.get("area", 1.0)  # m^2
        lift = cl * q * s
        drag = cd * q * s
        # Fluid-frame force (fluid x = flow direction; drag opposes motion => +drag along flow).
        # The frame is derived from the state here rather than read back out of the
        # transform tree: a fluid frame that someone else forgot to update silently
        # turns this drag into thrust.
        f_fluid = np.array([drag, lift])
        r_fluid_to_boat = utils.fluid_transform_from_velocity(u_local, v_local).rotation_matrix()
        return np.asarray(r_fluid_to_boat @ f_fluid, dtype=float)


class FiniteSpanKeel(BasicKeel):
    """Keel with lifting-line induced drag and post-stall blending (``model_type: finite_span``).

    Requires ``span`` (with ``area``) or ``effective_aspect_ratio``, and
    ``alpha_sep_deg``.
    """

    def _check_keys(self) -> None:
        """Require the finite-span keys."""
        if self.effective_aspect_ratio() <= 0.0:
            msg = (
                "keel model_type: finite_span needs span (with area) or effective_aspect_ratio; "
                "for the plain 2-D section use model_type: basic"
            )
            raise ValueError(msg)
        if not self.stall_blending:
            msg = "keel model_type: finite_span needs alpha_sep_deg > 0"
            raise ValueError(msg)

    def coefficients(self, aoa: float) -> tuple[float, float]:
        """Return finite-span ``(cl, cd)`` at an angle of attack [rad], blended into stall."""
        cl, cd = self.cl_cd(aoa, re=self.get_reynolds())
        cl, cd = self.apply_finite_span(cl, cd)
        return self.blend_stall(aoa, cl, cd)
