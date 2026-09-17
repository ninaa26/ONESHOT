"""Skin-friction laws a hull model can be built on.

The choice used to be an `if` inside `BasicHullModel`: anything that was not the
string `hughes` meant the flat coefficient, so `friction_model: hugues` ran the
model the config did not ask for and said nothing. Adding a third law meant
another branch in a method that is otherwise about hull geometry.

A law is a function of speed, waterline length and the hull's parameters, and
registers itself under the name a config names it by. Adding one is writing it
here; nothing else changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sailbench.sim.registry import register

if TYPE_CHECKING:
    from collections.abc import Mapping

PART = "friction"

# Below this the Hughes line is not valid and skin friction is not the story.
LAMINAR_RE = 1.0e4

# The boundary layer does not run the full waterline.
RE_LENGTH_FRACTION = 0.85


@register(PART, "flat")
def flat_plate(u: float, length: float, params: Mapping[str, Any]) -> float:
    """Return a constant coefficient, independent of speed.

    A plausible mid-range number for a hull this size, and what every config got
    before a friction law could be named. It does not vary with speed, which is
    the one thing skin friction is known to do.
    """
    del u, length, params
    return 0.004


@register(PART, "hughes")
def hughes(u: float, length: float, params: Mapping[str, Any]) -> float:
    """Return an ITTC-style Reynolds-dependent coefficient, times a form factor.

        Re = 0.85 * |u| * L / nu
        Cf = 0.066 / (log10(Re) - 2.03)^2
        ff = 1.05                     (a hull is not a flat plate)

    Falls back to the flat coefficient below Re 1e4, where the line does not hold.
    """
    nu = float(params.get("nu_water", 1.19e-6))  # [m^2/s] fresh water, ~15 C
    re = RE_LENGTH_FRACTION * abs(u) * length / nu
    if re < LAMINAR_RE:
        return flat_plate(u, length, params)
    cf = 0.066 / (np.log10(re) - 2.03) ** 2
    return float(cf * float(params.get("form_factor", 1.05)))
