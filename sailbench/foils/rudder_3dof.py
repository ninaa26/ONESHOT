import numpy as np
from .foil_base import Foil3DOF


class RudderModel3DOF(Foil3DOF):
    def __init__(self, p: dict):
        super().__init__(p, cache_tag="rudder", Re=5e5)

    # ------------------------------------------------------------------
    def compute(self, state: np.ndarray, inputs: dict) -> np.ndarray:
        u, v = state[0], state[1]
        V = np.hypot(u, v) + 1e-9

        beta = np.arctan2(v, u + 1e-9)
        delta = np.clip(
            inputs.get("delta_rudder", 0.0),
            -np.deg2rad(self.p["delta_max_deg"]),
            +np.deg2rad(self.p["delta_max_deg"]),
        )
        alpha = beta - delta

        CL, CD = self.cl_cd(alpha)

        q = 0.5 * self.p["rho_water"] * V**2
        L = q * self.p["S_r"] * CL
        D = q * self.p["S_r"] * CD

        # sign conventions
        Y_r = -L
        X_r = -D
        N_r = Y_r * self.p["x_r"]
        return np.array([X_r, Y_r, N_r])
