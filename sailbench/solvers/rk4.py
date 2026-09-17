"""Runge-Kutta 4th order solver."""

from collections.abc import Callable

import numpy as np


def rk4_step(
    f: Callable[[np.ndarray, float], np.ndarray],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Perform one RK4 integration step.

    `f` takes the state and the time offset within the step, and returns the
    state derivative. The offset matters whenever anything other than the state
    varies across the step -- here it is the actuators, which keep moving while
    the boat is being integrated. Evaluating them at the stage times is what lets
    a moving control surface be integrated at fourth order instead of being
    frozen at its value at the start of the step, which is only first order
    however exactly the actuator itself is solved.
    """
    k1 = f(state, 0.0)
    k2 = f(state + dt * k1 / 2.0, dt / 2.0)
    k3 = f(state + dt * k2 / 2.0, dt / 2.0)
    k4 = f(state + dt * k3, dt)

    return np.asarray(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)
