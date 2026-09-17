"""First-order servo model for a steered surface.

Separate from both the hub and the force models: an actuator produces no force,
it only decides where a control surface actually is, given where it was told to
go. The rudder and the sheet winch are the same kind of object with very
different constants, so the model is written once.

Why a first-order lag rather than a rate limit
----------------------------------------------
The previous model slewed toward the command at a fixed rate with forward Euler.
That is first order in dt, and measurement showed it had become the dominant
source of trajectory error in the whole simulator: with it slewing, RK4
converged at order 0.6-0.9, and with the rudder held constant, at 4.4 -- a factor
of 35 in error at dt = 0.02.

A first-order lag under a command held constant across the step has an exact
solution::

    theta(t + dt) = cmd + (theta - cmd) * exp(-dt / tau)

so it contributes no integration error at all, at any step size. That is the
point: the actuator stops being the accuracy bottleneck rather than merely
becoming a smaller one.

A rate limit is still available and still saturates, because a real servo has a
maximum slew rate. It is a clamp on top of the exact lag, not a substitute for it.
"""

from __future__ import annotations

import numpy as np


class Actuator:
    """A control surface that lags behind its command.

    Units are whatever the caller uses -- degrees for the rudder, radians for the
    sheet -- as long as `max_rate` and `deadband` use the same one.

    Args:
        tau_s: first-order lag time constant. 0 means the surface tracks its
            command instantly, subject only to the rate limit.
        max_rate: maximum slew per second. 0 means unlimited.
        deadband: commands smaller than this are treated as zero. A manual-helm
            aid; it corrupts action semantics for a learned policy, so leave it
            at 0 for RL.
        center_tau_s: when the command is zero, return to centre with this time
            constant instead of `tau_s`. Also a manual-helm aid, also 0 for RL.
        position: initial position.

    """

    def __init__(
        self,
        tau_s: float = 0.0,
        max_rate: float = 0.0,
        deadband: float = 0.0,
        center_tau_s: float = 0.0,
        position: float = 0.0,
    ) -> None:
        """Initialize the actuator."""
        self.tau_s = float(tau_s)
        self.max_rate = float(max_rate)
        self.deadband = float(deadband)
        self.center_tau_s = float(center_tau_s)
        self.position = float(position)

    def position_at(self, command: float, dt: float) -> float:
        """Where the surface will be `dt` into a step, without moving it.

        Lets an integrator sample the actuator at its stage times, so a surface
        that is still slewing is integrated at the solver's full order rather
        than being frozen at its value at the start of the step.
        """
        cmd = 0.0 if abs(float(command)) <= self.deadband else float(command)
        tau = self.center_tau_s if (cmd == 0.0 and self.center_tau_s > 0.0) else self.tau_s

        target = cmd if tau <= 0.0 else cmd + (self.position - cmd) * float(np.exp(-float(dt) / tau))

        if self.max_rate > 0.0:
            limit = self.max_rate * float(dt)
            target = self.position + float(np.clip(target - self.position, -limit, limit))
        return float(target)

    def advance(self, command: float, dt: float) -> float:
        """Advance one step against a command held constant across it.

        Call exactly once per step, outside the integrator: this is actuator
        state, not a function of the boat state, and slewing it once per RK4
        stage would multiply the effective rate by the number of stages.
        """
        self.position = self.position_at(command, dt)
        return self.position
