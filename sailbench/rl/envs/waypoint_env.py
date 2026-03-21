"""Gymnasium waypoint-navigation environment for SailBench."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step


@dataclass(slots=True)
class WaypointEnvConfig:
    """Configuration for waypoint navigation task."""

    simulator_config: str = "basic_sailbot.yaml"
    dt: float | None = None
    max_episode_steps: int = 1000
    max_rudder_deg: float = 35.0
    max_sail_deg: float = 85.0
    initial_heading_range_deg: float = 30.0
    initial_speed_max: float = 0.2
    spawn_radius_m: float = 10.0
    waypoint_min_radius_m: float = 12.0
    waypoint_max_radius_m: float = 25.0
    success_radius_m: float = 1.5
    fail_radius_m: float = 80.0
    speed_scale: float = 6.0
    yaw_rate_scale: float = 2.0
    wind_speed_scale: float = 15.0
    progress_reward_coeff: float = 3.0
    heading_reward_coeff: float = 0.05
    control_delta_penalty_coeff: float = 0.01
    control_effort_penalty_coeff: float = 0.002
    success_reward: float = 25.0
    failure_penalty: float = -10.0


class WaypointEnv(gym.Env[NDArray[np.float32], NDArray[np.float64]]):  # type: ignore[misc]
    """Continuous-control waypoint task built on SailboatHub dynamics."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, config: WaypointEnvConfig) -> None:
        super().__init__()
        self.cfg = config
        self.hub = SailboatHub(config_file=self.cfg.simulator_config)
        self.dt = float(self.cfg.dt or self.hub.simulation_cfg.get("dt", 0.02))
        self.max_sail_rad = math.radians(self.cfg.max_sail_deg)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)

        self.state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=0.0, v=0.0, r=0.0)
        self.waypoint = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.t = 0.0
        self.prev_distance = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.last_action = np.zeros(2, dtype=np.float64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment and sample start/goal states."""
        super().reset(seed=seed)
        del options

        self.hub = SailboatHub(config_file=self.cfg.simulator_config)
        self.steps = 0
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.last_action = np.zeros(2, dtype=np.float64)

        start = self._sample_start()
        self.state = start
        self.waypoint = self._sample_waypoint(start)
        self.prev_distance = self._distance_to_waypoint()

        return self._get_observation(), self._build_info(0.0, 0.0, 0.0, 0.0, False, False)

    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance simulation with normalized rudder/sail actions."""
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        rudder_deg = float(clipped[0] * self.cfg.max_rudder_deg)
        sail_rad = float((clipped[1] + 1.0) * 0.5 * self.max_sail_rad)

        self.state = self.hub.step(
            state=self.state,
            dt=self.dt,
            solver=rk4_step,
            sail_angle=sail_rad,
            rudder_angle=rudder_deg,
        )
        self.steps += 1
        self.t += self.dt

        distance = self._distance_to_waypoint()
        progress_reward = self.cfg.progress_reward_coeff * (self.prev_distance - distance)
        heading_alignment = self._heading_alignment()
        heading_reward = self.cfg.heading_reward_coeff * heading_alignment * math.exp(
            -distance / max(self.cfg.waypoint_max_radius_m, 1e-6),
        )
        control_delta_penalty = self.cfg.control_delta_penalty_coeff * float(np.sum(np.abs(clipped - self.prev_action)))
        control_effort_penalty = self.cfg.control_effort_penalty_coeff * float(np.sum(np.abs(clipped)))

        terminated = False
        truncated = self.steps >= self.cfg.max_episode_steps
        success = distance <= self.cfg.success_radius_m
        failure = distance >= self.cfg.fail_radius_m

        terminal_reward = 0.0
        if success:
            terminated = True
            terminal_reward = self.cfg.success_reward
        elif failure:
            terminated = True
            terminal_reward = self.cfg.failure_penalty

        reward = (
            progress_reward
            + heading_reward
            - control_delta_penalty
            - control_effort_penalty
            + terminal_reward
        )

        self.prev_distance = distance
        self.prev_action = clipped
        self.last_action = clipped

        info = self._build_info(
            progress_reward=progress_reward,
            heading_reward=heading_reward,
            control_delta_penalty=control_delta_penalty,
            control_effort_penalty=control_effort_penalty,
            success=success,
            failure=failure,
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _sample_start(self) -> State:
        theta = float(self.np_random.uniform(-math.pi, math.pi))
        radius = float(self.np_random.uniform(0.0, self.cfg.spawn_radius_m))
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        heading_jitter = math.radians(self.cfg.initial_heading_range_deg)
        heading = float(self.np_random.uniform(-heading_jitter, heading_jitter))
        u = float(self.np_random.uniform(0.0, self.cfg.initial_speed_max))
        v = float(self.np_random.uniform(-0.05, 0.05))
        r = float(self.np_random.uniform(-0.05, 0.05))
        return State(x=x, y=y, psi=(math.cos(heading), math.sin(heading)), u=u, v=v, r=r)

    def _sample_waypoint(self, start: State) -> NDArray[np.float64]:
        theta = float(self.np_random.uniform(-math.pi, math.pi))
        radius = float(self.np_random.uniform(self.cfg.waypoint_min_radius_m, self.cfg.waypoint_max_radius_m))
        return np.array(
            [
                start.x + radius * math.cos(theta),
                start.y + radius * math.sin(theta),
            ],
            dtype=np.float64,
        )

    def _distance_to_waypoint(self) -> float:
        dx = self.waypoint[0] - self.state.x
        dy = self.waypoint[1] - self.state.y
        return float(math.hypot(dx, dy))

    def _heading_alignment(self) -> float:
        dx = self.waypoint[0] - self.state.x
        dy = self.waypoint[1] - self.state.y
        dist = max(math.hypot(dx, dy), 1e-9)
        target_unit = np.array([dx / dist, dy / dist], dtype=np.float64)
        heading_unit = np.array([self.state.psi[0], self.state.psi[1]], dtype=np.float64)
        return float(np.sum(target_unit * heading_unit))

    def _relative_waypoint_boat_frame(self) -> NDArray[np.float64]:
        dx_world = self.waypoint[0] - self.state.x
        dy_world = self.waypoint[1] - self.state.y
        c = self.state.psi[0]
        s = self.state.psi[1]
        dx_boat = c * dx_world + s * dy_world
        dy_boat = -s * dx_world + c * dy_world
        return np.array([dx_boat, dy_boat], dtype=np.float64)

    def _get_observation(self) -> NDArray[np.float32]:
        rel_wp = self._relative_waypoint_boat_frame()
        distance = max(self._distance_to_waypoint(), 1e-9)
        rel_dir = rel_wp / distance

        wind_speed = float(self.hub.sail_cfg.get("wind_speed", 0.0))
        wind_dir_deg = float(self.hub.sail_cfg.get("wind_dir_deg", 90.0))
        wind_dir_rad = math.radians(wind_dir_deg)
        wind_world = np.array([math.cos(wind_dir_rad), math.sin(wind_dir_rad)], dtype=np.float64)
        c = self.state.psi[0]
        s = self.state.psi[1]
        wind_boat = np.array(
            [
                c * wind_world[0] + s * wind_world[1],
                -s * wind_world[0] + c * wind_world[1],
            ],
            dtype=np.float64,
        )

        observation = np.array(
            [
                np.clip(rel_wp[0] / self.cfg.waypoint_max_radius_m, -1.0, 1.0),
                np.clip(rel_wp[1] / self.cfg.waypoint_max_radius_m, -1.0, 1.0),
                np.clip(distance / self.cfg.waypoint_max_radius_m, 0.0, 1.0),
                np.clip(rel_dir[0], -1.0, 1.0),
                np.clip(rel_dir[1], -1.0, 1.0),
                np.clip(self.state.u / self.cfg.speed_scale, -1.0, 1.0),
                np.clip(self.state.v / self.cfg.speed_scale, -1.0, 1.0),
                np.clip(self.state.r / self.cfg.yaw_rate_scale, -1.0, 1.0),
                np.clip(wind_boat[0], -1.0, 1.0),
                np.clip(wind_boat[1], -1.0, 1.0),
                np.clip(wind_speed / self.cfg.wind_speed_scale, 0.0, 1.0),
                float(self.last_action[0]),
                float(self.last_action[1]),
            ],
            dtype=np.float32,
        )
        return observation

    def _build_info(
        self,
        progress_reward: float,
        heading_reward: float,
        control_delta_penalty: float,
        control_effort_penalty: float,
        success: bool,
        failure: bool,
    ) -> dict[str, Any]:
        return {
            "distance_to_waypoint": self._distance_to_waypoint(),
            "waypoint_x": float(self.waypoint[0]),
            "waypoint_y": float(self.waypoint[1]),
            "sim_time_s": self.t,
            "step_count": self.steps,
            "reward_progress": progress_reward,
            "reward_heading": heading_reward,
            "penalty_control_delta": control_delta_penalty,
            "penalty_control_effort": control_effort_penalty,
            "success": success,
            "failure": failure,
        }

