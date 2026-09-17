"""Sail model built on the ORC VPP aerodynamic formulation.

Reference: *ORC VPP Documentation 2023*, Offshore Racing Congress, sections
5.1-5.2. The CLmax/CD0 envelope in :data:`MAIN_AWA_DEG` / :data:`MAIN_CL` /
:data:`MAIN_CD0` is Table 5.1 (mainsail, "low" set -- a rig with no adjustable
check stays, the right choice for a model boat). Those single-sail coefficients
were last revised in 2016 and are unchanged by the 2026 VPP update, which only
touched hull residuary resistance.

Why this shape of model
-----------------------
The coefficients are functions of **apparent wind angle**, not of a geometric
angle of attack in a rotating sail frame, and the drive/heel split is a single
explicit resolution::

    CR = CL sin(beta) - CD cos(beta)      (drive, boat +x)
    CH = CL cos(beta) + CD sin(beta)      (heel/side)

That makes the class of chained-rotation sign errors unrepresentable, and it
replaces a symmetric NACA section -- which is the wrong shape for a cambered
membrane and carries no camber, so it makes no lift at zero incidence.

Sheet trim enters through ORC's ``flat`` parameter: the table gives the *maximum
achievable* lift at each apparent wind angle and ``flat`` scales it down for trim
that is not optimal. An over-eased sail drives ``flat`` to zero, so luffing falls
out of the model rather than needing a hand-rolled ramp.

Depowering
----------
ORC does not read ``flat`` off a trim curve and stop there: it *chooses* ``flat``
(and ``reef``) so the rig's heeling moment stays inside the boat's righting
moment. Without that constraint the model sails at permanent full power, which in
a 3-DOF simulator with no heel degree of freedom means the boat never pays for
the side force it generates. :meth:`flat_for_righting_moment` supplies it. It is
inactive unless the boat's righting moment is configured.

On sloops
---------
Flingo carries a main and a jib, so the single envelope here stands in for both.
ORC tabulates headsails separately; this model does not, and the jib is 39% of
Flingo's sail area, so its contribution is being represented by mainsail
coefficients. A headsail makes somewhat more lift at low apparent wind angles and
less at high ones, so expect this to understate upwind drive slightly and
overstate it deep downwind. Blending a real headsail table is the refinement, not
a different structure.
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
        area: sail area [m^2]. For a sloop, main plus jib.
        heff: effective rig height [m], used for induced drag. Prefer the
            measured masthead height above the waterline. Defaults to
            ``1.8 * sqrt(area)`` when absent.
        wind_speed, wind_dir_deg: true wind (direction it blows *to*).
        rho_air: [kg/m^3], supplied by the hub's ``environment`` block.
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
        # Righting-moment limit. Both keys are needed or neither: a limit without
        # an arm cannot be turned into a force, and an arm without a limit does
        # nothing. Absent, the sail carries whatever the trim curve asks for.
        limit = self.p.get("max_heeling_moment_nm")
        arm = self.p.get("heel_arm_m")
        if (limit is None) != (arm is None):
            msg = (
                "ORCSail needs max_heeling_moment_nm and heel_arm_m together: "
                f"got max_heeling_moment_nm={limit!r}, heel_arm_m={arm!r}"
            )
            raise ValueError(msg)
        self.max_heeling_moment = None if limit is None else float(limit)
        self.heel_arm = 0.0 if arm is None else float(arm)
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

        Peaks at ``alpha_opt``. Falls to 0 when the sail is eased until it luffs,
        and decays to ``flat_stall_floor`` when over-sheeted past stall.
        """
        a = abs(float(alpha_rad))
        x = a / self.alpha_opt
        if x <= 1.0:
            return float(np.sin(0.5 * np.pi * np.clip(x, 0.0, 1.0)))
        decay = float(np.clip((x - 1.0) / 1.5, 0.0, 1.0))
        return float(1.0 - (1.0 - self.flat_floor) * decay)

    def flat_for_righting_moment(
        self, flat: float, beta: float, cl_max: float, cd0: float, q_area: float
    ) -> float:
        """Largest `flat` up to `flat` whose heeling moment fits the righting moment.

        The heel coefficient is a quadratic in ``flat``::

            CH(f) = k cl_max^2 sin(beta) f^2 + cl_max cos(beta) f + cd0 sin(beta)

        opening upward, so the values satisfying ``CH(f) <= CH_max`` form an
        interval and the answer is closed-form -- no iteration, and no assumption
        that CH rises with f.

        That assumption fails downwind, which is not a detail. Past 90 degrees
        cos(beta) is negative, so *more* lift reduces heel, and the heel force is
        dominated by ``cd0 sin(beta)`` that no amount of easing touches. The
        constraint can then be infeasible with ``flat`` alone: ORC reduces sail
        *area* with ``reef`` for exactly this case, which is not modelled here.
        When that happens this returns the least-heeling trim available rather
        than easing further and making it worse.

        Returns `flat` unchanged when no righting moment is configured.
        """
        if self.max_heeling_moment is None or q_area <= 0.0 or self.heel_arm <= 0.0:
            return flat

        ch_max = self.max_heeling_moment / (q_area * self.heel_arm)
        k = KPM + self.area / (np.pi * self.heff**2)
        a = k * cl_max * cl_max * np.sin(beta)
        b = cl_max * np.cos(beta)
        c = cd0 * np.sin(beta)

        def ch(f: float) -> float:
            return a * f * f + b * f + c

        if ch(flat) <= ch_max:
            return flat  # the trim the helm asked for already fits

        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return flat  # no lift and no induced drag: nothing to depower
            eased = (ch_max - c) / b
            return float(np.clip(eased, 0.0, flat)) if b > 0.0 else flat

        disc = b * b - 4.0 * a * (c - ch_max)
        if disc > 0.0:
            root = np.sqrt(disc)
            lo = (-b - root) / (2.0 * a)
            hi = (-b + root) / (2.0 * a)
            # Feasible trims are [lo, hi]; intersect with what the helm can ease to.
            if max(lo, 0.0) <= min(hi, flat):
                return float(min(hi, flat))

        # Infeasible with flat alone: give the least heel available on [0, flat].
        vertex = -b / (2.0 * a)
        candidates = [0.0, flat]
        if 0.0 < vertex < flat:
            candidates.append(float(vertex))
        return float(min(candidates, key=ch))

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Return ``[Fx, Fy]`` in the boat frame [N]."""
        wind_speed = float(self.p.get("wind_speed", 0.0))
        wind_rad = np.radians(float(self.p.get("wind_dir_deg", 0.0)))
        rho = float(self.p.get("rho_air", 1.225))

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

        q = 0.5 * rho * aw_speed * aw_speed * self.area
        flat = self.flat_for_righting_moment(flat, beta, cl_max, cd0, q)

        cl = cl_max * flat
        # ORC eq.: CDi = [KPM + Aref / (pi * heff^2)] * (CLmax * flat)^2
        cd_induced = (KPM + self.area / (np.pi * self.heff**2)) * cl * cl
        cd = cd0 + cd_induced

        # ORC drive / heel resolution.
        cr = cl * np.sin(beta) - cd * np.cos(beta)
        ch = cl * np.cos(beta) + cd * np.sin(beta)

        self.last_awa_deg = float(np.degrees(awa))
        self.last_flat = float(flat)
        self.last_cl = float(cl)
        self.last_cd = float(cd)

        # Wind from +y pushes the boat toward -y.
        return np.array([q * cr, -wind_side * q * ch], dtype=float)
