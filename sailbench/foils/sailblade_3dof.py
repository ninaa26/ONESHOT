import math

import numpy as np


class SailBEM3DOF:
    """Blade-element sail model.
    - Discretizes the sail span, applies twist, sums forces.
    - Uses linear CL slope with soft stall + profile drag.
    - Adds global induced drag using AR and Oswald e.
    Inputs:
      state = [u,v,r,x,y,psi]
      inputs = {"delta_sail": rad}    # 0 = on centerline, + outboard
      env    = {"wind_speed": m/s, "wind_dir": rad FROM}
    Returns: np.array([Fx, Fy, N]) in body axes.
    """

    def __init__(self, p: dict):
        self.rho = float(p.get("rho_air", 1.225))
        self.S = float(p["S_s"])  # total sail area [m^2]
        self.x_s = float(p["x_s"])  # yaw lever [m]
        self.b = float(p.get("span", math.sqrt(p["S_s"] * float(p.get("AR", 4.0)))))  # span [m]
        self.Nel = int(p.get("nel", 12))  # blade elements along span

        # Aero coefficients
        self.CL_alpha = float(p.get("CL_alpha", 2 * np.pi))  # per rad
        self.stall_deg = float(p.get("stall_deg", 25.0))
        self.CD0 = float(p.get("CD0", 0.02))
        self.k_prof = float(p.get("k_prof", 0.01))  # viscous quadratic
        self.e_oswald = float(p.get("e_oswald", 0.8))  # induced-drag efficiency
        self.AR = float(p.get("AR", self.b**2 / self.S))

        # Twist (root near centerline; tip eased)
        self.twist_root = math.radians(float(p.get("twist_root_deg", 0.0)))
        self.twist_tip = math.radians(float(p.get("twist_tip_deg", 12.0)))
        self.delta_max = math.radians(float(p.get("delta_max_deg", 85.0)))

    # ---------------------------- helpers ----------------------------
    @staticmethod
    def _sat(x, limit):
        # smooth-ish clip to +/- limit (radians)
        return float(np.clip(x, -limit, +limit))

    # ----------------------------- API -------------------------------
    def compute(self, state: np.ndarray, inputs: dict, env: dict) -> np.ndarray:
        u, v, r, x, y, psi = [float(s) for s in state]

        # True wind in inertial (FROM angle -> velocity points opposite)
        V_tw = float(env["wind_speed"])
        psi_w = float(env["wind_dir"])
        Vw_i = np.array([-V_tw * math.cos(psi_w), -V_tw * math.sin(psi_w)])

        # Boat velocity in inertial
        Vb_i = np.array(
            [
                u * math.cos(psi) - v * math.sin(psi),
                u * math.sin(psi) + v * math.cos(psi),
            ],
        )
        # Apparent wind in body
        Vaw_i = Vw_i - Vb_i
        c, s = math.cos(-psi), math.sin(-psi)
        Vaw_b = np.array([c * Vaw_i[0] - s * Vaw_i[1], s * Vaw_i[0] + c * Vaw_i[1]])
        awx, awy = Vaw_b
        Vaw = math.hypot(awx, awy) + 1e-9
        beta_aw = math.atan2(awy, awx)  # apparent-wind angle in body

        # Control
        delta_s = float(inputs.get("delta_sail", 0.0))
        delta_s = max(0.0, min(self.delta_max, delta_s))

        # Spanwise discretization (rectangular chord for simplicity)
        # If you know your chord(z), replace S/Nel with c_i * dz.
        S_i = self.S / self.Nel
        zeta = np.linspace(0.0, 1.0, self.Nel, endpoint=False) + 0.5 / self.Nel  # midpoints 0..1

        q = 0.5 * self.rho * Vaw**2

        L_sum = 0.0
        D_prof_sum = 0.0

        stall = math.radians(self.stall_deg)

        for ζ in zeta:
            twist = self.twist_root + (self.twist_tip - self.twist_root) * ζ
            alpha = beta_aw - (delta_s + twist)
            alpha_eff = self._sat(alpha, stall)
            CL = self.CL_alpha * alpha_eff
            CD_prof = self.CD0 + self.k_prof * CL**2

            L_i = q * S_i * CL
            D_i = q * S_i * CD_prof

            L_sum += L_i
            D_prof_sum += D_i

        # Add global induced drag from total lift
        CDi_total = (L_sum / (q * self.S)) ** 2 / (math.pi * self.AR * self.e_oswald)
        D_induced = q * self.S * CDi_total

        D_total = D_prof_sum + D_induced

        # Resolve Lift/Drag along body axes (relative to apparent wind)
        ca, sa = math.cos(beta_aw), math.sin(beta_aw)
        Fx = -D_total * ca + L_sum * sa
        Fy = -D_total * sa - L_sum * ca
        N = Fy * self.x_s

        return np.array([Fx, Fy, N], dtype=float)
