"""Displacement-hull resistance with a real hull-speed limit.

The previous ``BasicHullModel`` had only a skin-friction-like surge term, so
nothing in the simulator stopped the boat from sailing past hull speed. A
displacement hull's resistance is dominated by wave making above about
Fn = 0.3, and rises steeply through Fn = 0.4:

    Fn = u / sqrt(g * Lwl)

This model splits resistance the way every VPP does (ORC VPP Documentation
2023, sec. 6.1-6.3; Delft Systematic Yacht Hull Series, Keuning & Katgert):

* **viscous** -- ITTC-57 friction line with a form factor,
* **residuary (wave-making)** -- a steep power law in Fn / Fn_hull, which is
  the cheap stand-in for the Delft regression that a 1 m model hull does not
  have tank data for,
* **cross-flow** -- sway and yaw damping from strip theory.

It also reports the **added mass** of the hull, which the hub folds into the
mass matrix. Added mass in sway and yaw is the same order as the rigid-body
values for a slender hull, so leaving it out (as the old model did) makes the
boat turn and slide several times faster than the real thing.
"""

from __future__ import annotations

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

G = 9.81
# Kinematic viscosity of fresh water at ~15 C [m^2/s].
NU_WATER = 1.14e-6


class DisplacementHull(Model):
    """Resistance and added mass for a displacement canoe body.

    Config keys:
        L, B, T: waterline length, beam, canoe-body draft [m].
        rho_water: [kg/m^3], default 1000.
        mass: boat mass [kg], used for the residuary term.
        form_factor: (1 + k) on the friction line, default 1.15.
        cr_coeff: residuary resistance at hull speed, as a fraction of
            displacement weight. Default 0.04.
        cr_exp: steepness of the residuary rise, default 6.
        fn_hull: Froude number defining hull speed, default 0.4.
        cd_cross: cross-flow drag coefficient of the lateral plane, default 1.0.
        yv1, nr1: optional *linear* sway / yaw damping [N.s/m], [N.m.s/rad].
            These config keys previously existed but were read by nothing.
        m_add_sway_factor: sway added mass as a multiple of boat mass,
            default 1.0.
        m_add_yaw_factor: yaw added inertia as a multiple of m * (L/6)^2,
            default 1.0.
        m_add_surge_factor: surge added mass as a multiple of boat mass,
            default 0.05.
    """

    def __init__(self, params: dict) -> None:
        """Initialize hull geometry and precompute wetted area."""
        super().__init__(params)
        self.L = float(self.p["L"])
        self.B = float(self.p["B"])
        self.T = float(self.p["T"])
        self.rho = float(self.p.get("rho_water", self.p.get("rho", 1000.0)))
        self.mass = float(self.p.get("mass", 27.0))

        # Displaced volume and canoe-body wetted area (Mumford-style estimate).
        self.volume = self.mass / self.rho
        self.wetted_area = 1.7 * self.L * self.T + self.volume / max(self.T, 1e-6)
        self.lateral_area = self.L * self.T

        self.form_factor = float(self.p.get("form_factor", 1.15))
        self.cr_coeff = float(self.p.get("cr_coeff", 0.04))
        self.cr_exp = float(self.p.get("cr_exp", 6.0))
        self.fn_hull = float(self.p.get("fn_hull", 0.4))
        self.cd_cross = float(self.p.get("cd_cross", 1.0))
        self.yv1 = float(self.p.get("yv1", 0.0))
        self.nr1 = float(self.p.get("nr1", 0.0))

    # --- added mass -----------------------------------------------------
    def added_mass(self) -> tuple[float, float, float]:
        """Return ``(X_udot, Y_vdot, N_rdot)`` added mass / inertia, positive."""
        m_surge = float(self.p.get("m_add_surge_factor", 0.05)) * self.mass
        m_sway = float(self.p.get("m_add_sway_factor", 1.0)) * self.mass
        i_yaw = float(self.p.get("m_add_yaw_factor", 1.0)) * self.mass * (self.L / 6.0) ** 2
        return m_surge, m_sway, i_yaw

    def friction_coefficient(self, speed: float) -> float:
        """ITTC-57 frictional resistance coefficient."""
        re = max(abs(speed) * self.L / NU_WATER, 1.0e3)
        return 0.075 / (np.log10(re) - 2.0) ** 2

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Return ``[Fx, Fy, Mz]`` in the boat frame."""
        del tf_tree
        u, v, r = float(state.u), float(state.v), float(state.r)

        # --- surge: viscous + residuary ---
        cf = self.friction_coefficient(u)
        r_viscous = 0.5 * self.rho * self.wetted_area * self.form_factor * cf * u * abs(u)

        fn = abs(u) / np.sqrt(G * self.L)
        weight = self.mass * G
        r_residuary = np.sign(u) * weight * self.cr_coeff * (fn / self.fn_hull) ** self.cr_exp

        fx = -(r_viscous + r_residuary)

        # --- sway: cross-flow drag on the lateral plane + linear term ---
        fy = -0.5 * self.rho * self.cd_cross * self.lateral_area * v * abs(v) - self.yv1 * v

        # --- yaw: strip-theory cross-flow integral, int|x|^3 dx = L^4/32 ---
        mz = -0.5 * self.rho * self.cd_cross * self.T * (self.L**4 / 32.0) * r * abs(r) - self.nr1 * r

        return np.array([fx, fy, mz], dtype=float)
