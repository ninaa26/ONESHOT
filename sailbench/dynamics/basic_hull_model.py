"""Basic hull drag model."""

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

GRAVITY = 9.81  # [m/s^2]


class BasicHullModel(Model):
    """Quadratic hull drag from simple geometry-based coefficients."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""
        del tf_tree

        u, v, r = state.u, state.v, state.r

        l = float(self.p["L"])
        b = float(self.p["B"])
        t = float(self.p["T"])
        rho = float(self.p.get("rho_water", 1000.0))

        s = 1.7 * l * (b + t)
        aside = l * t

        k_u = 0.5 * rho * s * 0.004
        k_v = 0.5 * rho * aside
        k_r = (1.0 / 8.0) * rho * t * (l**4)

        fx = -k_u * u * abs(u)
        fy = -k_v * v * abs(v)
        mz = -k_r * r * abs(r)

        fx += self._residuary_resistance(u, rho, l, t)

        return np.array([fx, fy, mz], dtype=float)

    def _residuary_resistance(self, u: float, rho: float, l: float, t: float) -> float:
        """Wave-making resistance, the term that makes a displacement hull have a top speed.

        The model above is skin friction only, which grows as u^2 and so never
        stops the boat: nothing here resisted it past hull speed. For a hull of
        this size wave-making dominates above roughly Froude 0.3, and its absence
        is why the boat reached Froude 0.77 against a hull-speed scale of 0.4.

        Buehler et al. (Robotic Sailing, 2018) use a quartic in the ratio of speed
        to hull speed, which is a hard enough wall to cap the boat near
        v_hull = 0.4*sqrt(g*L) without a discontinuity. `c_wave` sets how hard.

        Absent from the config, `c_wave` is zero and this term does nothing, so
        configs that have not opted in keep their previous behaviour exactly.
        """
        c_wave = float(self.p.get("c_wave", 0.0))
        if c_wave <= 0.0 or abs(u) < 1e-9:
            return 0.0

        v_hull = 0.4 * np.sqrt(GRAVITY * l)
        q = 0.5 * rho * u * u
        return float(-np.sign(u) * c_wave * q * (l * t) * (abs(u) / v_hull) ** 4)
