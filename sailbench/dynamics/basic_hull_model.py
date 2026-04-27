"""Basic hull drag model."""

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D


class BasicHullModel(Model):
    """Quadratic hull drag from simple geometry-based coefficients."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""
        del tf_tree

        u, v, r = state.u, state.v, state.r
        roll_rad = float(np.radians(state.roll_deg))
        c_roll = float(np.cos(roll_rad))
        s_roll = float(np.sin(roll_rad))

        l = float(self.p["L"])
        b = float(self.p["B"])
        t = float(self.p["T"])
        rho = float(self.p.get("rho", 1000.0))

        # Simple geometry proxy:
        # - as the hull heels, projected bottom/planform contribution drops with cos(roll)
        # - lateral side area uses port/starboard immersion based on signed heel
        s = 1.7 * l * (t + b * abs(c_roll))
        extra_draft = 0.5 * b * s_roll
        starboard_draft = max(1e-6, t + extra_draft)
        port_draft = max(1e-6, t - extra_draft)
        # v>0 (moving to starboard) -> drag from starboard side; v<0 -> port side.
        aside = l * (starboard_draft if v >= 0.0 else port_draft)
        t_eff = 0.5 * (starboard_draft + port_draft)

        k_u = 0.5 * rho * s * 0.004
        k_v = 0.5 * rho * aside
        k_r = (1.0 / 8.0) * rho * t_eff * (l**4)

        fx = -k_u * u * abs(u)
        fy = -k_v * v * abs(v)
        mz = -k_r * r * abs(r)

        return np.array([fx, fy, mz], dtype=float)
