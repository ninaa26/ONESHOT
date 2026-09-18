"""Basic hull drag model."""

import numpy as np

from sailbench.dynamics import friction
from sailbench.models.model import Model, State
from sailbench.sim.registry import lookup, register
from sailbench.tf.tf_tree import TFTree2D

GRAVITY = 9.81  # [m/s^2]


@register(
    "hull",
    "basic",
    name="Geometry",
    blurb="Quadratic drag from hull geometry, with friction and wave-making",
)
class BasicHullModel(Model):
    """Quadratic hull drag from simple geometry-based coefficients."""

    REQUIRES: tuple[tuple[str, ...], ...] = (("L",), ("B",), ("T",))

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""
        del tf_tree

        u, v, r = state.u, state.v, state.r

        length = float(self.p["L"])
        b = float(self.p["B"])
        t = float(self.p["T"])
        rho = float(self.p.get("rho_water", 1000.0))

        # 1.7*L*(B+T) is a rough stand-in; prefer a measured hull area when the
        # config carries one. On flingo the approximation is about 2.5x high.
        s = float(self.p.get("wetted_surface_m2") or 1.7 * length * (b + t))
        aside = length * t

        # Cross-flow drag coefficient, shared by sway and yaw because they are the
        # same physical mechanism -- the hull being dragged sideways through the
        # water -- resolved over different lever arms. 1.0 is what the sway term
        # has always implied (0.5*rho*A matches 0.5*rho*Cd*A at Cd = 1).
        cd_cross = float(self.p.get("cross_flow_cd", 1.0))

        k_u = 0.5 * rho * s * self._friction_coefficient(u, length)
        k_v = 0.5 * rho * cd_cross * aside
        # Strip theory. A strip at x sees lateral speed r*x, so its drag is
        # 0.5*rho*Cd*T*dx*|r x|(r x) and its moment about the centre is x times
        # that. Integrating x^2|x| over [-L/2, L/2] gives L^4/32, hence
        #
        #     k_r = 0.5 * rho * Cd * T * L^4/32 = rho * Cd * T * L^4 / 64
        #
        # The previous coefficient was rho*T*L^4/8, which is 8/Cd times this --
        # a factor of 8 at the Cd the sway term already assumes. The boat was
        # resisting rotation eight times harder than its own sway model implies.
        #
        # `yaw_damping_cd` exists because cross-flow drag is not the only thing
        # resisting a turn: a yawing hull also sheds circulatory lift, which this
        # model has no term for. Strip theory says it should equal cross_flow_cd,
        # and it defaults to it, but a turning-circle measurement is what should
        # set it. At the derived value this boat turns inside one waterline
        # length, where a real hull needs two to four, so the omitted term is not
        # small. Calibrate this rather than cross_flow_cd, which sway also uses.
        cd_yaw = float(self.p.get("yaw_damping_cd", cd_cross))
        k_r = rho * cd_yaw * t * (length**4) / 64.0

        fx = -k_u * u * abs(u)
        fy = -k_v * v * abs(v)
        mz = -k_r * r * abs(r)

        fx += self._residuary_resistance(u, rho, length, t)

        return np.array([fx, fy, mz], dtype=float)

    def added_mass(self) -> tuple[float, float, float]:
        """Surge, sway and yaw added mass, by strip theory over the measured hull.

        Water has to be pushed aside for the hull to accelerate, and the boat
        carries that water with it. For a hull this beamy relative to its length
        the effect is not a correction: sway added mass comes out roughly equal to
        the boat's own mass, so omitting it makes the hull slide sideways and spin
        up about twice as readily as it should.

        Each station contributes a 2-D sway added mass of ``rho * pi * T^2`` per
        unit length -- the flat-plate result, i.e. a Lewis section coefficient of
        1 -- and the yaw term is the same integrand weighted by ``x^2``::

            A22 = integral rho pi T(x)^2 dx
            A66 = integral rho pi T(x)^2 x^2 dx

        Surge is not a strip-theory quantity: a slender hull accelerating along
        its own axis disturbs very little water, and the usual estimate is a small
        fraction of the displacement, which `added_mass_surge_fraction` sets.

        Returns zeros when no `sections` table is configured, so a hull that has
        not opted in behaves exactly as before.
        """
        sections = self.p.get("sections") or []
        if len(sections) < 2:
            return 0.0, 0.0, 0.0

        rho = float(self.p.get("rho_water", 1000.0))
        coeff = float(self.p.get("added_mass_section_coeff", 1.0))
        xs = [float(sec["x_m"]) for sec in sections]
        drafts = [float(sec["draft_m"]) for sec in sections]

        # Trapezoidal integration over however the stations happen to be spaced.
        strip = [rho * np.pi * coeff * t * t for t in drafts]
        order = np.argsort(xs)
        x_sorted = np.array(xs)[order]
        m_sorted = np.array(strip)[order]
        a22 = float(np.trapezoid(m_sorted, x_sorted))
        a66 = float(np.trapezoid(m_sorted * x_sorted**2, x_sorted))

        mass = float(self.p.get("mass", 0.0))
        a11 = float(self.p.get("added_mass_surge_fraction", 0.05)) * mass
        return a11, a22, a66

    def _friction_coefficient(self, u: float, length: float) -> float:
        """Skin-friction coefficient from the law this hull names.

        `friction_model` picks one of the laws in `sailbench.dynamics.friction`,
        and defaults to `flat`, the constant every config got before the choice
        existed. A name no law is registered under now raises instead of quietly
        meaning `flat`, which is what a typo used to buy.
        """
        law = lookup(friction.PART, str(self.p.get("friction_model", "flat")))
        return float(law(u, length, self.p))

    def _residuary_resistance(self, u: float, rho: float, length: float, t: float) -> float:
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

        v_hull = 0.4 * np.sqrt(GRAVITY * length)
        q = 0.5 * rho * u * u
        return float(-np.sign(u) * c_wave * q * (length * t) * (abs(u) / v_hull) ** 4)
