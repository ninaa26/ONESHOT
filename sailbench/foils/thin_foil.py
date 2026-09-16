"""Keel and rudder as finite-span foils, sharing one force kernel.

Both appendages are the same physics with a different frame, so they are one
class. Everything about the lift/drag sign convention lives in
:mod:`sailbench.models.foil_theory`, so it cannot drift apart between the two
the way the old ``BasicKeel`` / ``BasicRudder`` pair did (the keel rotated its
force through the tf tree's fluid frame, the rudder rolled its own rotation
matrix, and only one of them was right).

Aspect ratio follows normal appendage practice: the hull and the free surface
act as an end plate, so the *effective* aspect ratio is roughly twice the
geometric one. That effective AR sets both the lift-curve slope and the induced
drag, which is the dominant drag term on a sailboat going upwind and was
missing entirely before.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sailbench.models.foil_theory import foil_force
from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D


class ThinFoil(Model):
    """A finite-span symmetric appendage (keel or rudder).

    Config keys:
        area: planform area [m^2].
        span: submerged span / depth [m]. Required for a sane aspect ratio.
        end_plate_factor: effective-AR multiplier, default 2.0.
        oswald: span efficiency, default 0.9.
        cd0: parasitic drag coefficient, default 0.008.
        x_pos, y_pos: position relative to the boat's centre of rotation [m].
        water_density: [kg/m^3], default 1000.
    """

    def __init__(self, params: dict[str, Any], frame: str) -> None:
        """Initialize a foil bound to a tf-tree frame ("keel" or "rudder")."""
        super().__init__(params)
        self.frame = frame
        self.area = float(self.p.get("area", 1.0))
        span = float(self.p.get("span", np.sqrt(self.area)))
        geometric_ar = span * span / max(self.area, 1e-9)
        self.aspect_ratio = geometric_ar * float(self.p.get("end_plate_factor", 2.0))
        self.oswald = float(self.p.get("oswald", 0.9))
        self.cd0 = float(self.p.get("cd0", 0.008))
        self.rho = float(self.p.get("water_density", 1000.0))
        self.x_pos = float(self.p.get("x_pos", 0.0))
        self.y_pos = float(self.p.get("y_pos", 0.0))

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Return ``[Fx, Fy]`` in the boat frame [N]."""
        # Velocity of the foil through the water, including yaw rate.
        u_local = float(state.u - state.r * self.y_pos)
        v_local = float(state.v + state.r * self.x_pos)
        if np.hypot(u_local, v_local) < 1e-9:
            return np.zeros(2, dtype=float)

        # Flow of water relative to the foil, in the foil's own frame.
        flow_boat = np.array([-u_local, -v_local], dtype=float)
        flow_foil = tf_tree.vector_to_frame(flow_boat, "boat", self.frame)

        f_foil = foil_force(
            flow_foil,
            area=self.area,
            rho=self.rho,
            aspect_ratio=self.aspect_ratio,
            cd0=self.cd0,
            oswald=self.oswald,
        )
        return tf_tree.vector_to_frame(f_foil, self.frame, "boat")
