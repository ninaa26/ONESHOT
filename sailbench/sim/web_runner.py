from __future__ import annotations

"""WebSocket-based simulation runner using SailboatHub.

This module keeps the physics core in Python (via SailboatHub) and exposes
the kinematic state over a WebSocket connection so a browser (e.g. three.js)
can render a 3D view and send rudder/sail controls.
"""

import argparse
import asyncio
import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO

from websockets.server import WebSocketServerProtocol, serve

from sailbench.models.model import State
from sailbench.rl.envs.waypoint_env import WaypointEnvConfig
from sailbench.sim.protocol import ControlInputs, make_state_message, parse_control_message
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step


@dataclass(slots=True)
class HubSimulationConfig:
    """Configuration for SailboatHub-based WebSocket runner."""

    config_file: str
    fps: float | None = None
    policy_model: str | None = None
    policy_config: str | None = None
    deterministic_policy: bool = True


@dataclass(slots=True)
class PolicyController:
    """Optional PPO policy controller for live simulation."""

    model: PPO
    cfg: WaypointEnvConfig
    waypoint: tuple[float, float]
    last_action: np.ndarray
    prev_distance: float

    @classmethod
    def from_files(cls, model_path: str, config_path: str | None) -> "PolicyController":
        full_cfg: dict[str, Any] = {}
        if config_path is not None:
            with Path(config_path).open(encoding="utf-8") as file:
                full_cfg = dict(yaml.safe_load(file) or {})
        env_cfg = WaypointEnvConfig(**full_cfg.get("env", {}))
        model = PPO.load(model_path)
        waypoint = (float(env_cfg.waypoint_min_radius_m), 0.0)
        return cls(
            model=model,
            cfg=env_cfg,
            waypoint=waypoint,
            last_action=np.zeros(2, dtype=np.float64),
            prev_distance=0.0,
        )

    def reset_waypoint(self, state: State, rng: np.random.Generator) -> None:
        theta = float(rng.uniform(-math.pi, math.pi))
        radius = float(rng.uniform(self.cfg.waypoint_min_radius_m, self.cfg.waypoint_max_radius_m))
        self.waypoint = (state.x + radius * math.cos(theta), state.y + radius * math.sin(theta))
        self.prev_distance = self.distance_to_waypoint(state)

    def distance_to_waypoint(self, state: State) -> float:
        dx = self.waypoint[0] - state.x
        dy = self.waypoint[1] - state.y
        return float(math.hypot(dx, dy))

    def _observation(self, hub: SailboatHub, state: State) -> np.ndarray:
        dx_world = self.waypoint[0] - state.x
        dy_world = self.waypoint[1] - state.y
        c, s = state.psi
        dx_boat = c * dx_world + s * dy_world
        dy_boat = -s * dx_world + c * dy_world
        distance = max(math.hypot(dx_world, dy_world), 1e-9)
        rel_dir_x = dx_boat / distance
        rel_dir_y = dy_boat / distance

        wind_speed = float(hub.sail_cfg.get("wind_speed", 0.0))
        wind_dir_deg = float(hub.sail_cfg.get("wind_dir_deg", 90.0))
        wind_dir_rad = math.radians(wind_dir_deg)
        wind_world_x = math.cos(wind_dir_rad)
        wind_world_y = math.sin(wind_dir_rad)
        wind_boat_x = c * wind_world_x + s * wind_world_y
        wind_boat_y = -s * wind_world_x + c * wind_world_y

        return np.array(
            [
                np.clip(dx_boat / self.cfg.waypoint_max_radius_m, -1.0, 1.0),
                np.clip(dy_boat / self.cfg.waypoint_max_radius_m, -1.0, 1.0),
                np.clip(distance / self.cfg.waypoint_max_radius_m, 0.0, 1.0),
                np.clip(rel_dir_x, -1.0, 1.0),
                np.clip(rel_dir_y, -1.0, 1.0),
                np.clip(state.u / self.cfg.speed_scale, -1.0, 1.0),
                np.clip(state.v / self.cfg.speed_scale, -1.0, 1.0),
                np.clip(state.r / self.cfg.yaw_rate_scale, -1.0, 1.0),
                np.clip(wind_boat_x, -1.0, 1.0),
                np.clip(wind_boat_y, -1.0, 1.0),
                np.clip(wind_speed / self.cfg.wind_speed_scale, 0.0, 1.0),
                float(self.last_action[0]),
                float(self.last_action[1]),
            ],
            dtype=np.float32,
        )

    def compute_controls(self, hub: SailboatHub, state: State, deterministic: bool) -> tuple[float, float]:
        obs = self._observation(hub, state)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        act = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.last_action = act
        rudder_deg = float(act[0] * self.cfg.max_rudder_deg)
        # Match training env scaling: action[1] in [-1, 1] maps to sheet limit [0, max].
        sail_cmd = 0.5 * (float(act[1]) + 1.0)
        sail_rad = float(sail_cmd * math.radians(self.cfg.max_sail_deg))
        return rudder_deg, sail_rad


@dataclass(slots=True)
class HubSimulation:
    """Wrapper around SailboatHub state and controls."""

    hub: SailboatHub
    state: State
    dt: float
    t: float = 0.0
    rudder_deg: float = 0.0
    sheet_limit_rad: float = 0.0
    paused: bool = False
    policy: PolicyController | None = None
    policy_deterministic: bool = True
    rng: np.random.Generator | None = None
    # Manual-control filtering (server-side safety net).
    # The browser now owns "snap back" behavior; this just smooths abrupt updates.
    target_rudder_deg: float = 0.0
    target_sail_rad: float = 0.0
    manual_rudder_rate_deg_s: float = 120.0

    def step(self) -> None:
        """Advance the simulation by one fixed step using current controls."""
        if self.paused:
            return

        if self.policy is not None:
            self.rudder_deg, requested_sheet_limit_rad = self.policy.compute_controls(
                self.hub,
                self.state,
                deterministic=self.policy_deterministic,
            )
            sheet_cap_rad = float(np.abs(self.target_sail_rad))
            if sheet_cap_rad <= 0.0:
                sheet_cap_rad = float(math.radians(self.policy.cfg.max_sail_deg))
            self.sheet_limit_rad = float(np.clip(np.abs(requested_sheet_limit_rad), 0.0, sheet_cap_rad))
        else:
            # Manual mode: rate-limit toward latest targets from the browser.
            max_step_deg = self.manual_rudder_rate_deg_s * float(self.dt)
            self.rudder_deg += float(
                np.clip(self.target_rudder_deg - self.rudder_deg, -max_step_deg, max_step_deg)
            )

            # Sail follows command immediately (no backend smoothing/rate limiting).
            self.sheet_limit_rad = float(self.target_sail_rad)

        self.state = self.hub.step(
            self.state,
            self.dt,
            solver=rk4_step,
            sail_angle=self.sheet_limit_rad,
            rudder_angle=self.rudder_deg,
        )
        self.t += self.dt

    def apply_controls(self, controls: ControlInputs) -> None:
        """Update control targets (rudder, sail, wind, pause/reset)."""
        # In RL mode, policy outputs own rudder/sheet commands every step.
        # Ignore manual helm inputs so they can never override policy intent.
        if self.policy is None and controls.rudder_deg is not None:
            self.target_rudder_deg = float(controls.rudder_deg)
        if controls.sail_deg is not None:
            self.target_sail_rad = math.radians(float(controls.sail_deg))

        if controls.wind_speed is not None:
            self.hub.sail_cfg["wind_speed"] = float(controls.wind_speed)
        if controls.wind_dir_deg is not None:
            self.hub.sail_cfg["wind_dir_deg"] = float(controls.wind_dir_deg)

        if controls.paused is not None:
            self.paused = controls.paused

        if controls.reset:
            self.state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=0.0, v=0.0, r=0.0)
            self.t = 0.0
            self.rudder_deg = 0.0
            self.sheet_limit_rad = 0.0
            self.target_rudder_deg = 0.0
            if self.policy is not None:
                self.target_sail_rad = math.radians(float(self.policy.cfg.max_sail_deg))
            else:
                self.target_sail_rad = 0.0
            if self.policy is not None and self.rng is not None:
                self.policy.reset_waypoint(self.state, self.rng)


async def hub_simulation_loop(
    ws: WebSocketServerProtocol,
    sim: HubSimulation,
) -> None:
    """Drive SailboatHub in a fixed-step loop and stream state to client."""
    controls = ControlInputs()

    async def control_task() -> None:
        nonlocal controls
        while True:
            raw_msg = await ws.recv()
            data = json.loads(raw_msg)
            controls = parse_control_message(data)
            sim.apply_controls(controls)

    ctrl_future = asyncio.create_task(control_task())

    try:
        while True:
            sim.step()
            if sim.policy is not None and sim.rng is not None:
                distance = sim.policy.distance_to_waypoint(sim.state)
                if distance <= sim.policy.cfg.success_radius_m:
                    sim.policy.reset_waypoint(sim.state, sim.rng)
            wind_speed = float(sim.hub.sail_cfg.get("wind_speed", 0.0))
            wind_dir_deg = float(sim.hub.sail_cfg.get("wind_dir_deg", 0.0))
            sail_force = getattr(sim.hub, "last_sail_force", (0.0, 0.0))
            forces = getattr(sim.hub, "last_forces", {})
            sail_angle_deg = math.degrees(float(getattr(sim.hub, "last_sail_angle_rad", 0.0)))
            msg = make_state_message(
                sim.state,
                sim.t,
                wind_speed=wind_speed,
                wind_dir_deg=wind_dir_deg,
                sail_force=sail_force,
                forces=forces,
                sail_angle_deg=sail_angle_deg,
                rudder_angle_deg=sim.rudder_deg,
                waypoint=sim.policy.waypoint if sim.policy is not None else None,
                control_mode="rl" if sim.policy is not None else "manual",
            )
            await ws.send(json.dumps(msg))
            await asyncio.sleep(sim.dt)
    finally:
        ctrl_future.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ctrl_future


async def handle_client(ws: WebSocketServerProtocol, path: str, cfg: HubSimulationConfig) -> None:
    """Handle a single WebSocket client using a dedicated SailboatHub instance."""
    del path  # unused

    hub = SailboatHub(config_file=cfg.config_file)
    dt_cfg = float(hub.simulation_cfg.get("dt", 0.02))
    fps = cfg.fps or (1.0 / dt_cfg if dt_cfg > 0 else 60.0)
    dt = 1.0 / fps

    policy = None
    rng = np.random.default_rng(7)
    if cfg.policy_model is not None:
        policy = PolicyController.from_files(
            model_path=cfg.policy_model,
            config_path=cfg.policy_config,
        )

    sim = HubSimulation(
        hub=hub,
        state=State(x=0.0, y=0.0, psi=(1.0, 0.0), u=0.0, v=0.0, r=0.0),
        dt=dt,
        policy=policy,
        policy_deterministic=cfg.deterministic_policy,
        rng=rng,
    )
    if sim.policy is not None:
        sim.target_sail_rad = math.radians(float(sim.policy.cfg.max_sail_deg))
        sim.policy.reset_waypoint(sim.state, rng)

    await hub_simulation_loop(ws, sim)


async def run_server(
    host: str,
    port: int,
    config_file: str,
    fps: float | None,
    policy_model: str | None,
    policy_config: str | None,
    deterministic_policy: bool,
) -> None:
    """Run a WebSocket server that exposes SailboatHub to browser clients."""
    cfg = HubSimulationConfig(
        config_file=config_file,
        fps=fps,
        policy_model=policy_model,
        policy_config=policy_config,
        deterministic_policy=deterministic_policy,
    )

    async with serve(
        lambda ws, path: handle_client(ws, path, cfg),
        host,
        port,
    ):
        await asyncio.Future()  # run forever


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SailBench SailboatHub WebSocket runner")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port to listen on (default: 8765)")
    parser.add_argument(
        "--config",
        default="basic_sailbot.yaml",
        help="YAML config filename (relative to configs/)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target simulation frames per second (overrides config dt if set)",
    )
    parser.add_argument(
        "--policy-model",
        default=None,
        help="Optional PPO model zip path to autopilot the boat.",
    )
    parser.add_argument(
        "--policy-config",
        default="configs/rl_waypoint_sb3.yaml",
        help="RL YAML config used for observation/action scaling.",
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions.",
    )

    args = parser.parse_args(argv)
    asyncio.run(
        run_server(
            args.host,
            args.port,
            args.config,
            args.fps,
            args.policy_model,
            args.policy_config,
            not args.stochastic_policy,
        ),
    )


if __name__ == "__main__":
    main()

