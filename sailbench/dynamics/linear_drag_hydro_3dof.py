"""
LinearDragHydro3DOF
-------------------
Pure linear damping in surge, sway, and yaw:
    X = -Xu1 * u
    Y = -Yv1 * v
    N = -Nr1 * r
Use this as a simple baseline hull model.
"""

import numpy as np
from .base_hydro import HydroModel3DOF


class LinearDragModel(HydroModel3DOF):
    def compute(
        self, state: np.ndarray, statedot: np.ndarray | None = None
    ) -> np.ndarray:
        u, v, r = state[:3]

        Xu1 = float(self.p.get("Xu1", 0.0))  # [N·s/m]
        Yv1 = float(self.p.get("Yv1", 0.0))  # [N·s/m]
        Nr1 = float(self.p.get("Nr1", 0.0))  # [N·m·s/rad]

        X = -Xu1 * u
        Y = -Yv1 * v
        N = -Nr1 * r
        return np.array([X, Y, N], dtype=float)
