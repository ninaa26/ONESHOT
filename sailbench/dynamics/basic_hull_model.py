"""Hull resistance models.

Two hulls, picked by the hull section's ``model_type``:

``basic`` -- :class:`BasicHullModel`
    Quadratic resistance from L, B, T: a flat skin-friction coefficient,
    cross-flow drag in sway, strip-theory yaw damping, and optionally the
    wave-making term set by ``c_wave``. No added mass.

``measured`` -- :class:`MeasuredHullModel`
    The same resistance with the Hughes skin-friction line in place of the
    flat coefficient, over a measured ``wetted_surface_m2``, plus strip-theory
    added mass integrated over measured draft ``sections``. Both are required:
    a hull that calls itself measured and then runs on the 1.7*L*(B+T) guess
    for its area is the thing this split exists to prevent.

Each refuses the other's keys rather than half-applying them: a config says
which hull it means and gets exactly that one.
"""

from typing import Any

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

GRAVITY = 9.81  # [m/s^2]


# Keys that only make sense for the measured model.
MEASURED_KEYS = ("sections", "friction_model", "form_factor", "added_mass_section_coeff", "added_mass_surge_fraction")


class BasicHullModel(Model):
    """Quadratic hull drag from simple geometry-based coefficients (``model_type: basic``)."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the hull model."""
        super().__init__(params)
        self._check_keys()

    def _check_keys(self) -> None:
        """Reject keys that belong to the measured model."""
        stray = [k for k in MEASURED_KEYS if k in self.p]
        if stray:
            msg = (
                f"hull model_type: basic got {', '.join(stray)}; "
                "use model_type: measured for the Hughes friction line and added mass"
            )
            raise ValueError(msg)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""
        del tf_tree

        u, v, r = state.u, state.v, state.r

        l = float(self.p["L"])
        b = float(self.p["B"])
        t = float(self.p["T"])
        rho = float(self.p.get("rho_water", 1000.0))

        # 1.7*L*(B+T) is a rough stand-in; prefer a measured hull area when the
        # config carries one. On flingo the approximation is about 2.5x high.
        s = float(self.p.get("wetted_surface_m2") or 1.7 * l * (b + t))
        aside = l * t

        # Cross-flow drag coefficient, shared by sway and yaw because they are the
        # same physical mechanism -- the hull being dragged sideways through the
        # water -- resolved over different lever arms. 1.0 is what the sway term
        # has always implied (0.5*rho*A matches 0.5*rho*Cd*A at Cd = 1).
        cd_cross = float(self.p.get("cross_flow_cd", 1.0))

        k_u = 0.5 * rho * s * self._friction_coefficient(u, l)
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
        k_r = rho * cd_yaw * t * (l**4) / 64.0

        fx = -k_u * u * abs(u)
        fy = -k_v * v * abs(v)
        mz = -k_r * r * abs(r)

        fx += self._residuary_resistance(u, rho, l, t)

        return np.array([fx, fy, mz], dtype=float)

    def added_mass(self) -> tuple[float, float, float]:
        """Surge, sway and yaw added mass. The basic hull carries none."""
        return 0.0, 0.0, 0.0

    def _friction_coefficient(self, u: float, l: float) -> float:
        """Flat skin-friction coefficient, a plausible mid-range constant."""
        del u, l
        return 0.004

    def _residuary_resistance(self, u: float, rho: float, l: float, t: float) -> float:
        """Wave-making resistance, the term that makes a displacement hull have a top speed.

        The friction terms grow as u^2 and so never stop the boat: nothing else
        here resists it past hull speed. For a hull of this size wave-making
        dominates above roughly Froude 0.3, and its absence is why the boat
        reached Froude 0.77 against a hull-speed scale of 0.4.

        Buehler et al. (Robotic Sailing, 2018) use a quartic in the ratio of speed
        to hull speed, which is a hard enough wall to cap the boat near
        v_hull = 0.4*sqrt(g*L) without a discontinuity. `c_wave` sets how hard;
        zero, the default, switches the term off.
        """
        c_wave = float(self.p.get("c_wave", 0.0))
        if c_wave <= 0.0 or abs(u) < 1e-9:
            return 0.0

        v_hull = 0.4 * np.sqrt(GRAVITY * l)
        q = 0.5 * rho * u * u
        return float(-np.sign(u) * c_wave * q * (l * t) * (abs(u) / v_hull) ** 4)


class MeasuredHullModel(BasicHullModel):
    """Hull with the Hughes friction line and strip-theory added mass (``model_type: measured``).

    Requires ``wetted_surface_m2`` and at least two draft ``sections``.
    """

    def _check_keys(self) -> None:
        """Require the measured geometry."""
        if not self.p.get("wetted_surface_m2"):
            msg = "hull model_type: measured needs wetted_surface_m2; for the L*(B+T) estimate use model_type: basic"
            raise ValueError(msg)
        if len(self.p.get("sections") or []) < 2:
            msg = "hull model_type: measured needs at least two draft sections for added mass"
            raise ValueError(msg)
        if "friction_model" in self.p:
            msg = "hull model_type: measured always uses the Hughes friction line; drop friction_model"
            raise ValueError(msg)

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

        """
        sections = self.p["sections"]
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

    def _friction_coefficient(self, u: float, l: float) -> float:
        """Hughes skin-friction coefficient, times a form factor.

        The flat 0.004 of the basic hull is a plausible mid-range number but it
        does not vary with speed, and skin friction is the one term here that
        has a well-established empirical line:

            Re = 0.85 * |u| * L / nu     (0.85 accounts for the boundary layer
                                          not running the full waterline)
            Cf = 0.066 / (log10(Re) - 2.03)^2
            ff = 1.05                     (form factor: a hull is not a flat plate)
        """
        nu = float(self.p.get("nu_water", 1.19e-6))  # [m^2/s] fresh water, ~15 C
        re = 0.85 * abs(u) * l / nu
        if re < 1.0e4:  # below this the line is not valid and Cf is not the story
            return 0.004
        cf = 0.066 / (np.log10(re) - 2.03) ** 2
        return cf * float(self.p.get("form_factor", 1.05))
