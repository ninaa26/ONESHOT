"""Boat3DOF

State: [u, v, r, x, y, psi]
  u, v : body surge/sway [m/s]
  r    : yaw rate [rad/s]
  x, y : world pos [m]
  psi  : heading [rad]

Models expected:
  hull  .compute(state)                         -> [X, Y, N]
  keel  .compute(state)                         -> [X, Y, N]
  rudder.compute(state, {"delta_rudder": rad})  -> [X, Y, N]
  sail  .compute(state, {"delta_sail":  rad}, env) -> [X, Y, N]

Use:
  boat = Boat3DOF(m, Iz, hull, keel, rudder, sail, env)
  boat.set_controls(delta_rudder=..., delta_sail=...)
  state = boat.step(dt, rk4_step)   # where rk4_step(f, state, dt) -> next_state
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Boat3DOF:
    m: float
    Iz: float
    hull: object
    keel: object
    rudder: object
    sail: object
    env: dict[str, float] = field(default_factory=lambda: {"wind_speed": 4.0, "wind_dir": 0.0})
    state: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    inputs: dict[str, float] = field(default_factory=lambda: {"delta_rudder": 0.0, "delta_sail": 0.0})
    use_coriolis: bool = True

    # --- small helpers -------------------------------------------------
    def reset(self, state: np.ndarray | None = None):
        self.state[:] = 0.0 if state is None else np.asarray(state, dtype=float)

    def set_controls(
        self,
        *,
        delta_rudder: float | None = None,
        delta_sail: float | None = None,
    ):
        if delta_rudder is not None:
            self.inputs["delta_rudder"] = float(delta_rudder)
        if delta_sail is not None:
            self.inputs["delta_sail"] = float(delta_sail)

    def set_environment(self, *, wind_speed: float | None = None, wind_dir: float | None = None):
        if wind_speed is not None:
            self.env["wind_speed"] = float(wind_speed)
        if wind_dir is not None:
            self.env["wind_dir"] = float(wind_dir)  # radians, FROM

    # --- physics core --------------------------------------------------
    def _forces(self, s: np.ndarray) -> tuple[float, float, float]:
        Xh, Yh, Nh = self.hull.compute(s)
        Xk, Yk, Nk = self.keel.compute(s)
        Xr, Yr, Nr = self.rudder.compute(s, {"delta_rudder": self.inputs["delta_rudder"]})
        Xs, Ys, Ns = self.sail.compute(s, {"delta_sail": self.inputs["delta_sail"]}, self.env)

        if self.use_coriolis:
            Cxu = self.m * s[1] * s[2]  # m * v * r
            Cyv = -self.m * s[0] * s[2]  # -m * u * r
        else:
            Cxu = Cyv = 0.0

        X = Xh + Xk + Xr + Xs + Cxu
        Y = Yh + Yk + Yr + Ys + Cyv
        N = Nh + Nk + Nr + Ns
        return X, Y, N

    def f(self, s: np.ndarray) -> np.ndarray:
        """State derivative sdot = f(s)."""
        u, v, r, _, _, psi = s
        X, Y, N = self._forces(s)
        du = X / self.m
        dv = Y / self.m
        dr = N / self.Iz
        dx = u * math.cos(psi) - v * math.sin(psi)
        dy = u * math.sin(psi) + v * math.cos(psi)
        dpsi = r
        return np.array([du, dv, dr, dx, dy, dpsi], dtype=float)

    # --- stepping via external solver ---------------------------------
    def step(
        self,
        dt: float,
        stepper: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, float], np.ndarray],
    ):
        """Advance by dt using an external stepper:
        next_state = stepper(self.f, self.state, dt)
        """
        self.state = stepper(self.f, self.state, dt)
        return self.state

    # --- factory ------------------------------------------
    @staticmethod
    def from_config(
        cfg: dict,
        *,
        hull_cls,
        keel_cls,
        rudder_cls,
        sail_cls,
        use_coriolis: bool = True,
    ) -> Boat3DOF:
        boat = cfg["boat"]
        hull = hull_cls(boat)
        keel = keel_cls(cfg["keel"])
        rudder = rudder_cls(cfg["rudder"])
        sail = sail_cls(cfg["sail"])
        envcfg = cfg.get("environment", {})
        env = {
            "wind_speed": float(envcfg.get("wind_speed", 4.0)),
            "wind_dir": math.radians(float(envcfg.get("wind_dir_deg", 0.0))),
        }
        return Boat3DOF(
            m=float(boat["m"]),
            Iz=float(boat["Iz"]),
            hull=hull,
            keel=keel,
            rudder=rudder,
            sail=sail,
            env=env,
            use_coriolis=use_coriolis,
        )
