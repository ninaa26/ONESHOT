"""Runge-Kutta 4th order solver."""

from collections.abc import Callable

import numpy as np


def rk4_step(
    f: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Perform one RK4 integration step."""
    k1 = f(state)
    k2 = f(state + dt * k1 / 2.0)
    k3 = f(state + dt * k2 / 2.0)
    k4 = f(state + dt * k3)

    return np.asarray(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)
