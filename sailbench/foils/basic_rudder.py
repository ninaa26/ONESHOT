"""Rudder foil models.

Two rudders, picked by the rudder section's ``model_type``:

``basic`` -- :class:`BasicRudder`
    The 2-D section polar as NeuralFoil returns it, with the angle of attack
    pinned at ``aoa_limit_deg`` and the coefficients at ``cl_max`` / ``cd_max``
    past stall. Lift is free of induced drag and the rudder stops being a
    control surface once it hits the clamps.

``finite_span`` -- :class:`FiniteSpanRudder`
    Lifting-line correction from ``span`` (or ``effective_aspect_ratio``), so
    the blade pays induced drag for its lift, and a blend towards a flat plate
    past ``alpha_sep_deg`` instead of a clamp, so deflection keeps changing the
    force all the way to broadside.

Each refuses the other's keys rather than half-applying them: a config says
which rudder it means and gets exactly that one.
"""

from typing import Any

import numpy as np

from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


# Keys that only make sense for one of the two models.
FINITE_SPAN_KEYS = ("span", "effective_aspect_ratio", "alpha_sep_deg")
CLAMP_KEYS = ("aoa_limit_deg", "cl_max", "cd_max")


class BasicRudder(Foil):
    """Rudder on the raw 2-D section polar, clamped past stall (``model_type: basic``)."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the rudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)
        self._check_keys()

    def _check_keys(self) -> None:
        """Reject keys that belong to the finite-span model."""
        stray = [k for k in FINITE_SPAN_KEYS if k in self.p]
        if stray:
            msg = (
                f"rudder model_type: basic is a 2-D section and got {', '.join(stray)}; "
                "use model_type: finite_span for induced drag and stall blending"
            )
            raise ValueError(msg)

    def coefficients(self, aoa: float) -> tuple[float, float]:
        """Return ``(cl, cd)`` at an angle of attack [rad].

        The angle is pinned at ``aoa_limit_deg`` and the coefficients at
        ``cl_max`` / ``cd_max``, so past stall every deflection returns the
        same force.
        """
        aoa_limit_deg = float(self.p.get("aoa_limit_deg", 25.0))
        aoa = float(np.clip(aoa, -np.radians(aoa_limit_deg), np.radians(aoa_limit_deg)))
        cl, cd = self.cl_cd(aoa, re=self.get_reynolds())
        cl_max = float(self.p.get("cl_max", 1.0))
        cl = float(np.clip(cl, -cl_max, cl_max))
        cd = float(np.clip(cd, 0.0, float(self.p.get("cd_max", 1.2))))
        return cl, cd

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the lift and drag coefficients for the rudder.

        Args:
            state (np.ndarray): Current boat state.
            tf_tree (TFTree2D): Current transform tree of the boat, used to get component positions.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within rudder frame)


        """
        # Use local inflow at rudder position, including yaw-rate contribution.
        # Body velocity at point (x, y): [u - r*y, v + r*x].
        x_pos = float(self.p.get("x_pos", 0.0))
        y_pos = float(self.p.get("y_pos", 0.0))
        u_local = float(state.u - state.r * y_pos)
        v_local = float(state.v + state.r * x_pos)

        v_local_boat = np.array([u_local, v_local], dtype=float)
        speed = float(np.hypot(u_local, v_local))
        if speed < 1e-6:
            return np.array([0.0, 0.0], dtype=float)

        # Resolve local velocity in rudder frame; AoA is opposite of local track angle.
        v_local_rudder = tf_tree.vector_to_frame(v_local_boat, "boat", "rudder")
        track = float(np.arctan2(v_local_rudder[1], v_local_rudder[0]))
        aoa = -track

        # Past stall the section polar stops meaning anything; which model is
        # configured decides whether that is handled by clamping or blending.
        cl, cd = self.coefficients(aoa)

        # Dynamic pressure and net foil forces.
        rho = float(self.p.get("rho_water", 1000.0))  # kg/m^3
        q = 0.5 * rho * speed**2
        area = float(self.p.get("area", 1.0))  # m^2
        effectiveness = float(self.p.get("effectiveness", 0.25))

        drag = cd * q * area * effectiveness
        lift = cl * q * area * effectiveness

        # Resolve in the flow frame, whose +x axis is the rudder's direction of travel:
        # drag opposes that motion, lift acts perpendicular to it.
        f_flow = np.array([-drag, lift], dtype=float)
        c, s = np.cos(track), np.sin(track)
        r_flow_to_rudder = np.array([[c, -s], [s, c]], dtype=float)
        f_rudder = r_flow_to_rudder @ f_flow
        return tf_tree.vector_to_frame(f_rudder, "rudder", "boat")


class FiniteSpanRudder(BasicRudder):
    """Rudder with lifting-line induced drag and post-stall blending (``model_type: finite_span``).

    Requires ``span`` (with ``area``) or ``effective_aspect_ratio``, and
    ``alpha_sep_deg``. The clamps of the basic model are refused: clamping a
    blended coefficient would put back the kink the blend exists to remove.
    """

    def _check_keys(self) -> None:
        """Require the finite-span keys and reject the clamps."""
        if self.effective_aspect_ratio() <= 0.0:
            msg = (
                "rudder model_type: finite_span needs span (with area) or effective_aspect_ratio; "
                "for the plain 2-D section use model_type: basic"
            )
            raise ValueError(msg)
        if not self.stall_blending:
            msg = "rudder model_type: finite_span needs alpha_sep_deg > 0"
            raise ValueError(msg)
        stray = [k for k in CLAMP_KEYS if k in self.p]
        if stray:
            msg = f"rudder model_type: finite_span blends past stall and got clamp keys {', '.join(stray)}"
            raise ValueError(msg)

    def coefficients(self, aoa: float) -> tuple[float, float]:
        """Return finite-span ``(cl, cd)`` at an angle of attack [rad], blended into stall."""
        cl, cd = self.cl_cd(aoa, re=self.get_reynolds())
        cl, cd = self.apply_finite_span(cl, cd)
        return self.blend_stall(aoa, cl, cd)
