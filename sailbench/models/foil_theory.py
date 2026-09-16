"""Shared foil theory: coefficients and the one force kernel every foil uses.

This module exists so that the lift/drag sign convention is written down in
exactly one place. Every foil in sailbench (keel, rudder, sail) resolves its
force through :func:`foil_force`, so a convention bug can only ever be fixed
once, and ``test/test_foil_theory.py`` pins the convention with physical
anchors (a keel must resist leeway; a rudder must turn the boat the way it is
deflected).

Coefficient model
-----------------
Finite-span thin-foil theory blended into flat-plate behaviour past stall:

* 3-D lift slope from Helmbold/Prandtl: ``a = a_2d / (1 + a_2d / (pi * AR * e))``
* attached lift saturating at a physical maximum,
  ``CL = CL_max * tanh(a * alpha / CL_max)`` -- slope ``a`` near zero
  incidence, bounded by ``CL_max`` however far the solver wanders
* separated (flat plate) ``CL = sin(2a)``, ``CD = 2 sin^2(a)``
* blended with ``s = 1 - exp(-(alpha / alpha_sep)^2)`` (Buehler et al., 2018),
  where ``alpha_sep`` is where the attached branch would reach ``CL_max``
* induced drag ``CD_i = CL^2 / (pi * AR * e)`` -- the term the previous
  NeuralFoil-based models omitted entirely.

The result is smooth, finite and physically sensible over the whole
[-180, 180] degree range, which a 2-D section solver is not.
"""

from __future__ import annotations

import numpy as np

# Thin-airfoil 2-D lift slope [1/rad].
CL_ALPHA_2D = 2.0 * np.pi
# Sentinel: derive the separation angle from the foil's own aspect ratio.
ALPHA_SEP_FROM_AR = -1.0
# Oswald span efficiency for a plain untwisted fin.
OSWALD_DEFAULT = 0.9
# Maximum attached-flow lift coefficient for a thin symmetric section.
CL_MAX_DEFAULT = 1.1


def fold_incidence(angle_rad: float) -> float:
    """Fold a flow angle into [-pi/2, pi/2] for a symmetric, bidirectional foil.

    A symmetric section does not know which end is the leading edge, so flow
    arriving from astern is aerodynamically the mirror of flow from ahead.
    Folding here is what keeps the models finite and continuous instead of
    asking a 2-D solver to extrapolate to +/-179 degrees.
    """
    a = float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))
    if a > 0.5 * np.pi:
        a -= np.pi
    elif a < -0.5 * np.pi:
        a += np.pi
    return a


def lift_slope_3d(aspect_ratio: float, oswald: float = OSWALD_DEFAULT) -> float:
    """Finite-span lift-curve slope [1/rad] (Helmbold/Prandtl correction)."""
    ar = max(float(aspect_ratio), 1e-3)
    return CL_ALPHA_2D / (1.0 + CL_ALPHA_2D / (np.pi * ar * float(oswald)))


def foil_coefficients(
    alpha_rad: float,
    aspect_ratio: float,
    cd0: float,
    oswald: float = OSWALD_DEFAULT,
    alpha_sep_rad: float = ALPHA_SEP_FROM_AR,
    cl_max: float = CL_MAX_DEFAULT,
) -> tuple[float, float]:
    """Return (CL, CD) for a symmetric foil at signed incidence ``alpha_rad``.

    ``alpha_rad`` is the angle of attack: the signed angle between the chord
    and the oncoming flow, already folded to [-pi/2, pi/2].
    """
    a = fold_incidence(alpha_rad)
    ar = max(float(aspect_ratio), 1e-3)
    e = float(oswald)

    # Attached-flow branch, saturating at cl_max so the solver can never be
    # handed an unbounded lift coefficient.
    clm = max(float(cl_max), 1e-6)
    slope = lift_slope_3d(ar, e)
    cl_attached = clm * float(np.tanh(slope * a / clm))

    # Separation onset: where the attached branch would reach CL_max. Low
    # aspect ratios have a shallower slope, so they separate later -- which is
    # what real low-AR appendages do.
    a_sep = float(alpha_sep_rad) if alpha_sep_rad > 0.0 else clm / slope

    # Separated / flat-plate branch.
    cl_separated = float(np.sin(2.0 * a))
    cd_separated = 2.0 * float(np.sin(a)) ** 2

    # Smooth blend: 0 = fully attached, 1 = fully separated.
    s = 1.0 - float(np.exp(-((a / a_sep) ** 2)))

    cl = (1.0 - s) * cl_attached + s * cl_separated
    cd_induced = cl * cl / (np.pi * ar * e)
    cd = float(cd0) + cd_induced + s * cd_separated

    return float(cl), float(cd)


def foil_force(
    flow_foil: np.ndarray,
    area: float,
    rho: float,
    aspect_ratio: float,
    cd0: float,
    oswald: float = OSWALD_DEFAULT,
    alpha_sep_rad: float = ALPHA_SEP_FROM_AR,
    cl_max: float = CL_MAX_DEFAULT,
) -> np.ndarray:
    """Force on a foil, in the foil frame, from the flow it sees.

    Args:
        flow_foil: velocity of the *fluid relative to the foil*, in the foil
            frame, with the chord along +x. This is ``-v_foil`` for a foil
            moving through still water.
        area: planform area [m^2].
        rho: fluid density [kg/m^3].
        aspect_ratio: geometric aspect ratio (span^2 / area).
        cd0: parasitic (zero-lift) drag coefficient.

    Returns:
        ``[Fx, Fy]`` in the foil frame [N].

    Convention (pinned by ``test/test_foil_theory.py``):
        drag acts *along* the flow, lift acts 90 degrees counter-clockwise
        from it, and the sign of CL carries which way the foil is pushed.
    """
    w = np.asarray(flow_foil, dtype=float)
    speed = float(np.hypot(w[0], w[1]))
    if speed < 1e-9:
        return np.zeros(2, dtype=float)

    drag_dir = w / speed
    lift_dir = np.array([-drag_dir[1], drag_dir[0]], dtype=float)

    # Angle of attack: chord is +x, flow arrives from direction -w.
    alpha = fold_incidence(np.arctan2(w[1], w[0]) - np.pi)

    cl, cd = foil_coefficients(alpha, aspect_ratio, cd0, oswald, alpha_sep_rad, cl_max)

    q = 0.5 * float(rho) * speed * speed * float(area)
    return np.asarray(q * (cd * drag_dir + cl * lift_dir), dtype=float)
