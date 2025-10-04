import numpy as np
from .foil_base import Foil3DOF


class KeelModel3DOF(Foil3DOF):
    def __init__(self, p: dict):
        super().__init__(p, cache_tag="keel", Re=1e6)

    # ------------------------------------------------------------------
    def compute(self, state: np.ndarray) -> np.ndarray:
        u, v = state[0], state[1]
        V = np.hypot(u, v) + 1e-9

        beta = np.arctan2(v, u + 1e-9)  # leeway
        CL, CD = self.cl_cd(beta)

        q = 0.5 * self.p["rho_water"] * V**2
        L = q * self.p["S_k"] * CL
        D = q * self.p["S_k"] * CD

        Y_k = -L
        X_k = -D
        N_k = Y_k * self.p["x_k"]
        return np.array([X_k, Y_k, N_k])
