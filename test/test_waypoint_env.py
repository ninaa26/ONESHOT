"""Contract tests for waypoint Gym environment."""

from __future__ import annotations

import numpy as np

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig


def _make_env(max_episode_steps: int = 50) -> WaypointEnv:
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        max_episode_steps=max_episode_steps,
        waypoint_min_radius_m=8.0,
        waypoint_max_radius_m=12.0,
    )
    return WaypointEnv(config=cfg)


def test_reset_returns_valid_observation() -> None:
    env = _make_env()
    obs, _ = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_step_contract_and_action_clipping() -> None:
    env = _make_env()
    env.reset(seed=0)
    action = np.array([5.0, -7.0], dtype=np.float32)  # intentionally out of bounds
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert np.all(env.last_action <= 1.0)
    assert np.all(env.last_action >= -1.0)


def test_episode_truncates_at_horizon() -> None:
    env = _make_env(max_episode_steps=3)
    env.reset(seed=11)
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert truncated

