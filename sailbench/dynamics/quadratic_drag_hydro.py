"""Linear hydrodynamic hull model."""

import numpy as np

from sailbench.tf.tf_tree import TFTree2D
from sailbench.models.model import Model, State
from sailbench.utils.coordinate_helper import get_local_track


class QuadraticHydroModel(Model):
    """Quadratic viscous drag hydrodynamic hull model."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""

        u, v, r = state.u, state.v, state.r

        # Transform into hull frame
        u_local = state.psi[0] * u + state.psi[1] * v
        v_local = -state.psi[1] * u + state.psi[0] * v

        # Quadratic damping coefficients
        xu2 = float(self.p.get("Xu2", 0.0))  # [N·s²/m²]
        yv2 = float(self.p.get("Yv2", 0.0))  # [N·s²/m²]
        nr2 = float(self.p.get("Nr2", 0.0))  # [N·m·s²/rad²]

        # Quadratic drag
        x = -xu2 * u_local * abs(u_local)
        y = -yv2 * v_local * abs(v_local)
        n = -nr2 * r * abs(r)

        return np.array([x, y, n], dtype=float)
    