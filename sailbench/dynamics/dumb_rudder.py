"""Dumb rudder model."""

import numpy as np
from numpy.typing import NDArray

from sailbench.tf.tf_tree import TFTree2D
from sailbench.models.model import Model, State


class DumbRudderModel(Model):
    """Dumb rudder model."""

    def compute(self, state: State, tf_tree: TFTree2D) -> NDArray[np.float64]:
        """Apply a simple yaw moment based on speed and rudder angle.

        Args:
            state (State): State object

        Returns:
            np.ndarray: [Fx, Fy, Mz] in boat frame. Fx and Fy are zero; Mz is a
            heuristic moment proportional to speed and rudder angle.

        """
        # Boat speed magnitude in body frame
        u, v = state.u, state.v
        speed = float(np.hypot(u, v))
        if speed < 1e-6:
            return np.array([0.0, 0.0, 0.0], dtype=float)

        # Rudder angle relative to boat: Transform2D from boat -> rudder is stored directly.
        rudder_tf = tf_tree.transforms.get("rudder")
        if rudder_tf is None:
            # No rudder frame; no effect.
            return np.array([0.0, 0.0, 0.0], dtype=float)

        rudder_angle = float(np.arctan2(rudder_tf.s, rudder_tf.c))  # radians

        # Simple proportional yaw moment model:
        #   Mz = k_m * speed^2 * sin(rudder_angle)
        # k_m can be tuned per-boat; default is 1.0.
        k_m = float(self.p.get("moment_coeff", 1.0))
        mz = k_m  * np.sin(rudder_angle)

        return np.array([0.0, 0.0, -mz], dtype=float)