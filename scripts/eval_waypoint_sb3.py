"""Evaluate a trained PPO waypoint policy on SailBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure the repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO waypoint policy.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to checkpoint zip (e.g. runs/.../best_model/best_model.zip).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_waypoint_sb3.yaml"),
        help="Path to RL config YAML.",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=None,
        help="Optional path to VecNormalize stats file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (defaults to config).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    args = parser.parse_args(argv)

    full_cfg = _load_yaml(args.config)
    env_cfg = WaypointEnvConfig(**full_cfg.get("env", {}))
    eval_cfg = full_cfg.get("eval", {})
    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 20))
    deterministic = bool(eval_cfg.get("deterministic", True))

    base_env = DummyVecEnv([lambda: WaypointEnv(config=env_cfg)])
    if args.vecnormalize is not None:
        env = VecNormalize.load(str(args.vecnormalize), base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    model = PPO.load(str(args.model), env=env)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    success_flags: list[float] = []
    final_distances: list[float] = []
    path_efficiencies: list[float] = []
    times_to_goal: list[float] = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        initial_distance = None
        final_info: dict[str, Any] | None = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            episode_return += float(rewards[0])
            episode_length += 1
            final_info = infos[0]
            if initial_distance is None:
                initial_distance = float(final_info.get("distance_to_waypoint", 0.0))

        final_distance = float(final_info.get("distance_to_waypoint", 0.0)) if final_info else 0.0
        success = 1.0 if (final_info and bool(final_info.get("success", False))) else 0.0
        sim_time_s = float(final_info.get("sim_time_s", 0.0)) if final_info else 0.0
        if success > 0.0:
            times_to_goal.append(sim_time_s)

        if initial_distance is None:
            initial_distance = final_distance
        path_eff = max(initial_distance - final_distance, 0.0) / max(initial_distance, 1e-6)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        success_flags.append(success)
        final_distances.append(final_distance)
        path_efficiencies.append(path_eff)

    metrics = {
        "episodes": episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(success_flags)),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_path_efficiency": float(np.mean(path_efficiencies)),
        "mean_time_to_goal_s": float(np.mean(times_to_goal)) if times_to_goal else None,
    }
    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        with args.output_json.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()

