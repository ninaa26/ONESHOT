"""Smoke test for SB3 integration."""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig


def test_ppo_smoke_train_short_run() -> None:
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        max_episode_steps=50,
        waypoint_min_radius_m=8.0,
        waypoint_max_radius_m=12.0,
    )

    env = DummyVecEnv([lambda: Monitor(WaypointEnv(config=cfg))])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        learning_rate=3e-4,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=64)

