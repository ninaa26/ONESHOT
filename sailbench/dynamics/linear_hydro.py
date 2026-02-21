"""Linear hydrodynamic hull model."""

import numpy as np

from sailbench.models.model import Model, State
from sailbench.utils.coordinate_helper import get_local_track


class LinearHydroModel(Model):
    """Linear viscous drag hydrodynamic hull model."""

    def compute(self, state: State) -> np.ndarray:
        """Compute forces on hull model.

        Args:
            state (State): State object

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        u, v, r = state.u, state.v, state.r

        u_local = state.psi[0] * u + state.psi[1] * v
        v_local = -state.psi[1] * u + state.psi[0] * v

        xu1 = float(self.p.get("Xu1", 0.0))  # [N·s/m]
        yv1 = float(self.p.get("Yv1", 0.0))  # [N·s/m]
        nr1 = float(self.p.get("Nr1", 0.0))  # [N·m·s/rad]

        x = -xu1 * u_local
        y = -yv1 * v_local
        n = -nr1 * r
        return np.array([x, y, n], dtype=float)
