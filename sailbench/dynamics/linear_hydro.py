"""Linear hydrodynamic hull model."""

import numpy as np

from sailbench.models.model import Model


class LinearHydroModel(Model):
    """Linear viscous drag hydrodynamic hull model."""

    def compute(self, state: np.ndarray) -> np.ndarray:
        """Compute forces on hull model.

        Args:
            state (np.ndarray): State vector containing [u, v, r]

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        u, v, r = state[:3]

        xu1 = float(self.p.get("Xu1", 0.0))  # [N·s/m]
        yv1 = float(self.p.get("Yv1", 0.0))  # [N·s/m]
        nr1 = float(self.p.get("Nr1", 0.0))  # [N·m·s/rad]

        x = -xu1 * u
        y = -yv1 * v
        n = -nr1 * r
        return np.array([x, y, n], dtype=float)
