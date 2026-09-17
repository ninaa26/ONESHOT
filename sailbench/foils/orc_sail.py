"""Sail model built on the ORC VPP aerodynamic formulation.

Reference: *ORC VPP Documentation 2023*, Offshore Racing Congress, sections
5.1-5.5. The CLmax/CD0 envelopes are Table 5.1 (mainsail) and Table 5.4 (jib),
both the "low" set -- a rig with no adjustable check stays or forestay, the
right choice for a model boat. Those single-sail coefficients were last revised
in 2016 and are unchanged by the 2026 VPP update, which only touched hull
residuary resistance.

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
ORC tabulates the main and the jib separately and combines them into one
"collective" rig, section 5.4.1: each coefficient is the area-weighted sum of the
individual sails' coefficients, normalised by the reference area,

    CLmax = sum_i CLmax_i * bk_i * A_i / Aref            (5.35)
    CD0   = sum_i CD0_i   * bk_i * A_i / Aref            (5.36)
    KPP   = sum_i kp_i * CLmax_i^2 * bk_i * A_i / (Aref * CLmax^2)   (5.41)

with ``bk_i`` a blanketing factor. The two rigs are separate models so a config
says which one it means: :class:`ORCMainSail` is a single mainsail and refuses a
``jib_area``; :class:`ORCWithJibSail` is main plus jib and requires one. They share
everything but the list of sails :meth:`ORCMainSail.envelope` sums over. The jib
makes more lift than the main at low apparent wind angles and none past about
150 degrees, so a sloop points better and runs slower than a main-only rig of
the same area.

Blanketing is 1 for both sails here. ORC's mainsail blanketing only differs
from 1 with a mizzen staysail, and the jib's only for an overlapping genoa
(``fj`` in section 5.6.2 is zero when the jib fits inside the foretriangle),
neither of which a model sloop carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

# --- ORC VPP 2023 Table 5.1, mainsail, "low" coefficient set ----------------
MAIN_AWA_DEG = np.array([0.0, 7.0, 9.0, 12.0, 28.0, 60.0, 90.0, 120.0, 150.0, 180.0])
MAIN_CL = np.array([0.00000, 0.86207, 1.05172, 1.16379, 1.34698, 1.35345, 1.26724, 0.93103, 0.38793, -0.11207])
MAIN_CD0 = np.array([0.04310, 0.02586, 0.02328, 0.02328, 0.03259, 0.11302, 0.38250, 0.96888, 1.31578, 1.34483])
# Two-dimensional quadratic viscous drag coefficient (ORC "kpm").
KPM = 0.01379

# --- ORC VPP 2023 Table 5.4, jib, "low" coefficient set --------------------
# The table starts at 7 degrees; np.interp holds the first value below that,
# which is the CL = 0 luff the mainsail table states explicitly.
JIB_AWA_DEG = np.array([7.0, 15.0, 20.0, 27.0, 50.0, 60.0, 100.0, 150.0, 180.0])
JIB_CL = np.array([0.00000, 1.00000, 1.37500, 1.45000, 1.45000, 1.25000, 0.40000, 0.00000, -0.10000])
JIB_CD0 = np.array([0.05000, 0.03200, 0.03100, 0.03700, 0.25000, 0.35000, 0.73000, 0.95000, 0.90000])
# ORC "kpj".
KPJ = 0.016


@dataclass(frozen=True)
class SailTable:
    """One sail's ORC coefficient table: CLmax and CD0 against apparent wind angle, plus its ``kp``."""

    awa_deg: np.ndarray
    cl: np.ndarray
    cd0: np.ndarray
    kp: float

    def at(self, awa_deg: float) -> tuple[float, float]:
        """Return ``(CLmax, CD0)`` interpolated at an apparent wind angle."""
        return float(np.interp(awa_deg, self.awa_deg, self.cl)), float(np.interp(awa_deg, self.awa_deg, self.cd0))


MAIN_TABLE = SailTable(MAIN_AWA_DEG, MAIN_CL, MAIN_CD0, KPM)
JIB_TABLE = SailTable(JIB_AWA_DEG, JIB_CL, JIB_CD0, KPJ)

# --- ORC Figure 5.14, kheff against apparent wind angle -------------------
# The effective rig height is not the masthead. Close-hauled the jib seals
# against the deck and the two sails act as one taller wing, so the rig sheds
# less tip vortex than its height suggests; eased onto a reach that seal is
# lost and the interaction turns unfavourable. ORC gives the curve only as a
# figure; these are traced from the published plots at 5-degree intervals and
# held at 0.80 past 80 degrees, where both figures end flat.
#
# Two editions are carried because the peak is a rating parameter, not a
# measurement. ORC raised it from 1.22 to 1.4513 in 2023 as one half of a
# "package" whose other half was deeper depowering (minimum flat 0.62 -> 0.42)
# and a stronger twist function. Inside the VPP those offset. In a simulator
# with no righting-moment limit only the power-adding half is felt, so the
# 2022 curve is the more defensible choice until that limit is configured.
# Below 1.0 -- the reaching side -- the two curves are nearly identical.
KHEFF_AWA_DEG = np.arange(0.0, 81.0, 5.0)
KHEFF_2022 = np.array([
    1.000, 1.093, 1.169, 1.210, 1.2200, 1.178, 1.118, 1.059, 0.999,
    0.939, 0.899, 0.871, 0.845, 0.824, 0.809, 0.801, 0.800,
])
KHEFF_2023 = np.array([
    1.000, 1.195, 1.350, 1.433, 1.4513, 1.365, 1.248, 1.133, 1.028,
    0.948, 0.899, 0.868, 0.844, 0.825, 0.810, 0.802, 0.800,
])
KHEFF_CURVES = {"orc-2022": KHEFF_2022, "orc-2023": KHEFF_2023}


class ORCMainSail(Model):
    """Mainsail-only aerodynamic model using the ORC VPP coefficient envelope.

    For a main-and-jib rig use :class:`ORCWithJibSail`. This model rejects a
    ``jib_area`` rather than ignore it, so a sloop config cannot quietly run as
    a single sail.

    Config keys (all optional except ``area``):
        area: reference sail area [m^2].
        heff: rig height [m], the highest point of the sail plan above the
            waterline (ORC's ``b + HBI``). Defaults to ``1.8 * sqrt(area)``.
        heff_model: ``orc-2022`` or ``orc-2023`` scales ``heff`` by that
            edition's ``kheff`` curve against apparent wind angle (Figure 5.14),
            so the rig is effectively taller close-hauled and shorter on a
            reach. The editions differ only in the close-hauled peak, 1.22
            against 1.45; see :data:`KHEFF_CURVES`. Absent, ``heff`` is constant.
        eff_span_corr: ORC's sail-plan correction to effective span, eq. 5.42,
            from roach, fractionality and overlap. Default 1.0 (no correction).
        wind_speed, wind_dir_deg: true wind (direction it blows *to*).
        rho_air: [kg/m^3], supplied by the hub's ``environment`` block.
        alpha_opt_deg: sail angle of attack giving peak lift, default 22.
        flat_stall_floor: residual lift fraction when badly over-sheeted.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the ORC sail model."""
        super().__init__(params)
        self.area = float(self.p.get("area", 1.0))
        self.sails = self._rig()
        self.heff = float(self.p.get("heff", 1.8 * np.sqrt(max(self.area, 1e-6))))
        heff_model = str(self.p.get("heff_model", "constant")).lower()
        if heff_model != "constant" and heff_model not in KHEFF_CURVES:
            known = ", ".join(["constant", *sorted(KHEFF_CURVES)])
            msg = f"ORCMainSail heff_model must be one of {known}: got {heff_model!r}"
            raise ValueError(msg)
        self.kheff = KHEFF_CURVES.get(heff_model)
        self.eff_span_corr = float(self.p.get("eff_span_corr", 1.0))
        self.alpha_opt = np.radians(float(self.p.get("alpha_opt_deg", 22.0)))
        self.flat_floor = float(self.p.get("flat_stall_floor", 0.55))
        # Righting-moment limit. Both keys are needed or neither: a limit without
        # an arm cannot be turned into a force, and an arm without a limit does
        # nothing. Absent, the sail carries whatever the trim curve asks for.
        limit = self.p.get("max_heeling_moment_nm")
        arm = self.p.get("heel_arm_m")
        if (limit is None) != (arm is None):
            msg = (
                "ORCMainSail needs max_heeling_moment_nm and heel_arm_m together: "
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
        self.last_heff = self.heff

    # --- rig ------------------------------------------------------------
    def _rig(self) -> list[tuple[SailTable, float]]:
        """Return the sails making up the rig as ``(table, area)`` pairs."""
        if "jib_area" in self.p:
            msg = (
                "ORCMainSail is a single mainsail and got jib_area="
                f"{self.p['jib_area']!r}; use model_type: orc_w_jib for a main-and-jib rig"
            )
            raise ValueError(msg)
        return [(MAIN_TABLE, self.area)]

    # --- coefficient envelope ------------------------------------------
    def envelope(self, awa_deg: float) -> tuple[float, float, float]:
        """Return the collective ``(CLmax, CD0, kpp)`` of the rig at an apparent wind angle.

        ORC eqs. 5.35, 5.36 and 5.41: each sail's table value weighted by its
        share of the reference area. ``kpp`` is weighted by lift squared as
        well, so the sail doing the lifting sets the quadratic viscous drag.
        With a single mainsail the result is the mainsail table unchanged.
        """
        b = float(np.clip(abs(awa_deg), 0.0, 180.0))
        cl = cd0 = 0.0
        lift_weight = kp_weight = kp_mean = 0.0
        for table, area in self.sails:
            w = area / self.area
            cl_i, cd_i = table.at(b)
            cl += w * cl_i
            cd0 += w * cd_i
            lift_weight += w * cl_i * cl_i
            kp_weight += table.kp * w * cl_i * cl_i
            kp_mean += w * table.kp
        # Eq. 5.41 divides by CLmax^2, which is zero head to wind. kpp then
        # multiplies CL^2 = 0 so its value is moot; the area-weighted mean keeps
        # it finite and continuous.
        kpp = kp_weight / lift_weight if lift_weight > 1e-12 else kp_mean
        return cl, cd0, kpp

    def effective_height(self, awa_deg: float) -> float:
        """Return the effective rig height for induced drag at an apparent wind angle.

        ORC eqs. 5.43 and 5.45: ``heff = eff_span_corr * kheff(beta) * (b + HBI)``.
        With ``heff_model`` unset ``kheff`` is 1 and this is the configured
        height scaled by ``eff_span_corr`` alone.
        """
        k = 1.0 if self.kheff is None else float(np.interp(abs(awa_deg), KHEFF_AWA_DEG, self.kheff))
        return self.eff_span_corr * k * self.heff

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
        self,
        flat: float,
        beta: float,
        cl_max: float,
        cd0: float,
        q_area: float,
        kpp: float = KPM,
        heff: float | None = None,
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
        h = self.heff if heff is None else heff
        k = kpp + self.area / (np.pi * h**2)
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
        rho = float(self.p.get("rho_air", 1.225))

        # Shared with the windage model so the two cannot disagree about the wind.
        aw_boat = utils.apparent_wind_boat(
            state,
            tf_tree,
            float(self.p.get("wind_speed", 0.0)),
            float(self.p.get("wind_dir_deg", 0.0)),
        )

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

        cl_max, cd0, kpp = self.envelope(np.degrees(beta))
        heff = self.effective_height(np.degrees(beta))
        flat = self.flat_from_trim(alpha) if alpha > 0.0 else 0.0

        q = 0.5 * rho * aw_speed * aw_speed * self.area
        flat = self.flat_for_righting_moment(flat, beta, cl_max, cd0, q, kpp, heff)

        cl = cl_max * flat
        # ORC eq. 5.46: CDi = [KPP + Aref / (pi * heff^2)] * (CLmax * flat)^2
        cd_induced = (kpp + self.area / (np.pi * heff**2)) * cl * cl
        cd = cd0 + cd_induced

        # ORC drive / heel resolution.
        cr = cl * np.sin(beta) - cd * np.cos(beta)
        ch = cl * np.cos(beta) + cd * np.sin(beta)

        self.last_awa_deg = float(np.degrees(awa))
        self.last_flat = float(flat)
        self.last_cl = float(cl)
        self.last_cd = float(cd)
        self.last_heff = float(heff)

        # Wind from +y pushes the boat toward -y.
        return np.array([q * cr, -wind_side * q * ch], dtype=float)


class ORCWithJibSail(ORCMainSail):
    """Main-and-jib aerodynamic model: ORC's collective rig, section 5.4.1.

    Same config as :class:`ORCMainSail` plus:
        jib_area: jib area [m^2], part of ``area``. Required. The envelope is
            the area-weighted blend of the main and jib tables; ``area`` stays
            the reference area the coefficients are normalised by.
    """

    def _rig(self) -> list[tuple[SailTable, float]]:
        """Return the main and jib, split by ``jib_area``."""
        jib_area = self.p.get("jib_area")
        if jib_area is None:
            msg = "ORCWithJibSail needs jib_area; for a single mainsail use model_type: orc_main"
            raise ValueError(msg)
        self.jib_area = float(jib_area)
        if not 0.0 < self.jib_area <= self.area:
            msg = f"ORCWithJibSail jib_area must lie within (0, area]: got jib_area={self.jib_area}, area={self.area}"
            raise ValueError(msg)
        self.main_area = self.area - self.jib_area
        return [(MAIN_TABLE, self.main_area), (JIB_TABLE, self.jib_area)]
