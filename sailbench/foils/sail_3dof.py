"""
SailModel3DOF
-------------
• Inherits Foil3DOF
• Inputs:
    state = [u, v, r, x, y, psi]  (body velocities, pose)
    inputs = {"delta_sail": <rad>}  (0 = centreline; + = eased to starboard)
    env    = {"wind_speed": <m/s>, "wind_dir": <rad, FROM>}
• Returns: np.array([Fx, Fy, N]) in boat body axes.
"""

import numpy as np
import math
from .foil_base import Foil3DOF


class SailModel3DOF(Foil3DOF):
    def __init__(self, p: dict):
        # Default Re for a small sail at low speed; override via p["Re_air"] if desired.
        Re_air = float(p.get("Re_air", 3e5))
        super().__init__(
            p, cache_tag="sail", alphas=np.arange(-30, 31, 1), Re=Re_air  # deg
        )
        # Convenience shorthands
        self.rho_air = float(p.get("rho_air", 1.225))
        self.S_s = float(p["S_s"])
        self.x_s = float(p["x_s"])
        self.delta_max = math.radians(
            float(p.get("delta_max_deg", 85.0))
        )  # clamp if provided

    # ------------------------------------------------------------------
    def compute(self, state: np.ndarray, inputs: dict, env: dict) -> np.ndarray:
        """
        env['wind_dir'] is the direction the wind is COMING FROM, in radians (inertial frame).
        """
        u, v, r, x, y, psi = state.astype(float)

        # True wind vector in inertial (FROM angle → velocity points opposite)
        V_tw = float(env["wind_speed"])
        psi_w = float(env["wind_dir"])
        Vw_i = -V_tw * np.array([math.cos(psi_w), math.sin(psi_w)])  # toward direction

        # Boat velocity in inertial
        Vb_i = np.array(
            [
                u * math.cos(psi) - v * math.sin(psi),
                u * math.sin(psi) + v * math.cos(psi),
            ]
        )

        # Apparent wind in inertial, then to body frame
        Vaw_i = Vw_i - Vb_i
        c, s = math.cos(-psi), math.sin(-psi)
        R_bi = np.array([[c, -s], [s, c]])  # inertial -> body
        Vaw_b = R_bi @ Vaw_i
        awx, awy = Vaw_b
        V_aw = float(np.hypot(awx, awy) + 1e-9)

        # Apparent wind angle in body frame, sail trim, AoA
        beta_aw = math.atan2(awy, awx)  # rad
        delta_s = float(inputs.get("delta_sail", 0.0))
        # Clamp trim (optional)
        delta_s = max(0.0, min(self.delta_max, delta_s))
        alpha = beta_aw - delta_s  # rad

        # CL/CD from XFoil polar (Foil3DOF handles clipping to polar range)
        CL, CD = self.cl_cd(alpha)

        # Dynamic pressure and 2D forces in wind-axes (D along -Vaw, L ⟂ to Vaw)
        q = 0.5 * self.rho_air * V_aw**2
        L = q * self.S_s * CL
        D = q * self.S_s * CD

        # Resolve to body axes
        # Unit vectors along/normal to apparent wind in body frame
        ca, sa = math.cos(beta_aw), math.sin(beta_aw)
        # Drag acts opposite apparent wind direction; Lift is 90° CW from wind (to mimic sail suction)
        Fx = -D * ca + L * sa
        Fy = -D * sa - L * ca

        # Yaw moment about CG
        N = Fy * self.x_s

        return np.array([Fx, Fy, N], dtype=float)
