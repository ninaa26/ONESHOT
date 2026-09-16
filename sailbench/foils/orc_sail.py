"""Sail model built on the ORC VPP aerodynamic formulation.

Reference: *ORC VPP Documentation 2023*, Offshore Racing Congress, sections
5.1-5.2. The mainsail CLmax/CD0 envelope in :data:`MAIN_AWA_DEG` /
:data:`MAIN_CL` / :data:`MAIN_CD0` is Table 5.1 ("low" set, i.e. a rig with no
adjustable check stays -- the right choice for a model boat). Those single-sail
coefficients were last revised in 2016 and are unchanged by the 2026 VPP
update, which only touched the hull residuary-resistance model.

Why this shape of model
-----------------------
The coefficients are functions of **apparent wind angle**, not of a geometric
angle of attack in a rotating sail frame, and the drive/heel split is a single
explicit resolution::

    CR = CL sin(beta) - CD cos(beta)      (drive, boat +x)
    CH = CL cos(beta) + CD sin(beta)      (heel/side)

That removes the entire class of chained-rotation sign errors, and because
Flingo Floaty is a single-sail rig the ORC mainsail table applies directly with
no main/jib blending.

Sheet trim enters through ORC's ``flat`` parameter: the table gives the *maximum
achievable* lift at each apparent wind angle, and ``flat`` scales it down for
trim that is not optimal. An over-eased sail drives ``flat`` to zero, which is
luffing -- so luffing falls out of the model instead of needing a hand-rolled
ramp.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

# --- ORC VPP 2023 Table 5.1, mainsail, "low" coefficient set ----------------
MAIN_AWA_DEG = np.array([0.0, 7.0, 9.0, 12.0, 28.0, 60.0, 90.0, 120.0, 150.0, 180.0])
MAIN_CL = np.array([0.00000, 0.86207, 1.05172, 1.16379, 1.34698, 1.35345, 1.26724, 0.93103, 0.38793, -0.11207])
MAIN_CD0 = np.array([0.04310, 0.02586, 0.02328, 0.02328, 0.03259, 0.11302, 0.38250, 0.96888, 1.31578, 1.34483])
# Two-dimensional quadratic viscous drag coefficient (ORC "kpm").
KPM = 0.01379


class ORCSail(Model):
    """Single-sail aerodynamic model using the ORC VPP coefficient envelope.

    Config keys (all optional except ``area``):
        area: sail area [m^2].
        heff: effective rig height [m] for induced drag. Defaults to
            ``1.8 * sqrt(area)``, a reasonable high-aspect model-boat rig.
        wind_speed, wind_dir_deg: true wind (direction it blows *to*).
        air_density: [kg/m^3], default 1.225.
        alpha_opt_deg: sail angle of attack giving peak lift, default 22.
        flat_stall_floor: residual lift fraction when badly over-sheeted.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the ORC sail model."""
        super().__init__(params)
        self.area = float(self.p.get("area", 1.0))
        self.heff = float(self.p.get("heff", 1.8 * np.sqrt(max(self.area, 1e-6))))
        self.alpha_opt = np.radians(float(self.p.get("alpha_opt_deg", 22.0)))
        self.flat_floor = float(self.p.get("flat_stall_floor", 0.55))
        # Diagnostics for the HUD / debugging.
        self.last_awa_deg = 0.0
        self.last_flat = 0.0
        self.last_cl = 0.0
        self.last_cd = 0.0

    # --- coefficient envelope ------------------------------------------
    def envelope(self, awa_deg: float) -> tuple[float, float]:
        """Max achievable CL and parasitic CD0 at an apparent wind angle."""
        b = float(np.clip(abs(awa_deg), 0.0, 180.0))
        return (
            float(np.interp(b, MAIN_AWA_DEG, MAIN_CL)),
            float(np.interp(b, MAIN_AWA_DEG, MAIN_CD0)),
        )

    def flat_from_trim(self, alpha_rad: float) -> float:
        """ORC ``flat`` depowering factor from the sail's angle of attack.

        Peaks at ``alpha_opt``. Falls to 0 when the sail is eased until it
        luffs, and decays to ``flat_stall_floor`` when over-sheeted past stall.
        """
        a = abs(float(alpha_rad))
        x = a / self.alpha_opt
        if x <= 1.0:
            return float(np.sin(0.5 * np.pi * np.clip(x, 0.0, 1.0)))
        decay = float(np.clip((x - 1.0) / 1.5, 0.0, 1.0))
        return float(1.0 - (1.0 - self.flat_floor) * decay)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Return ``[Fx, Fy]`` in the boat frame [N]."""
        wind_speed = float(self.p.get("wind_speed", 0.0))
        wind_rad = np.radians(float(self.p.get("wind_dir_deg", 0.0)))
        rho = float(self.p.get("air_density", 1.225))

        # Apparent wind in the boat frame.
        wind_world = wind_speed * np.array([np.cos(wind_rad), np.sin(wind_rad)])
        v_boat_world = tf_tree.vector_to_frame(np.array([state.u, state.v], dtype=float), "boat", "world")
        aw_boat = tf_tree.vector_to_frame(wind_world - v_boat_world, "world", "boat")

        aw_speed = float(np.hypot(aw_boat[0], aw_boat[1]))
        if aw_speed < 1e-6:
            self.last_awa_deg = self.last_flat = self.last_cl = self.last_cd = 0.0
            return np.zeros(2, dtype=float)

        # Apparent wind angle: direction the wind blows FROM, off the bow.
        # Positive = wind from the +y side.
        awa = float(np.arctan2(-aw_boat[1], -aw_boat[0]))
        beta = abs(awa)
        wind_side = 1.0 if awa >= 0.0 else -1.0

        # Sail angle of attack = apparent wind angle minus boom angle.
        boom = abs(float(np.arctan2(tf_tree.transforms["sail"].s, tf_tree.transforms["sail"].c)))
        alpha = beta - boom

        cl_max, cd0 = self.envelope(np.degrees(beta))
        flat = self.flat_from_trim(alpha) if alpha > 0.0 else 0.0

        cl = cl_max * flat
        # ORC eq.: CDi = [KPM + Aref / (pi * heff^2)] * (CLmax * flat)^2
        cd_induced = (KPM + self.area / (np.pi * self.heff**2)) * cl * cl
        cd = cd0 + cd_induced

        # ORC drive / heel resolution.
        cr = cl * np.sin(beta) - cd * np.cos(beta)
        ch = cl * np.cos(beta) + cd * np.sin(beta)

        q = 0.5 * rho * aw_speed * aw_speed * self.area

        self.last_awa_deg = float(np.degrees(awa))
        self.last_flat = float(flat)
        self.last_cl = float(cl)
        self.last_cd = float(cd)

        # Wind from +y pushes the boat toward -y.
        return np.array([q * cr, -wind_side * q * ch], dtype=float)
