"""Linear viscous drag hull model for Boat3DOF.

State format: [u, v, r, x, y, psi]
Returns: [X, Y, N] in body axes.
"""

import numpy as np


class LinearDragModel:
    """Linear viscous drag hydrodynamic hull model for 3DOF boat."""

    def __init__(self, p: dict) -> None:
        """Initialize with boat/hull params (Xu1, Yv1, Nr1)."""
        self.p = p
        self.Xu1 = float(p.get("Xu1", 0.0))  # [N·s/m] surge
        self.Yv1 = float(p.get("Yv1", 0.0))  # [N·s/m] sway
        self.Nr1 = float(p.get("Nr1", 0.0))  # [N·m·s/rad] yaw

    def compute(self, state: np.ndarray) -> np.ndarray:
        """Compute hull drag forces and moment.

        Args:
            state: [u, v, r, x, y, psi]

        Returns:
            [X, Y, N] in body axes (N)
        """
        u, v, r = float(state[0]), float(state[1]), float(state[2])
        X = -self.Xu1 * u
        Y = -self.Yv1 * v
        N = -self.Nr1 * r
        return np.array([X, Y, N], dtype=float)
