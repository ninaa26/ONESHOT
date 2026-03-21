"""Train a waypoint-navigation PPO policy on SailBench."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure the repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def _build_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"waypoint_ppo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO for SailBench waypoint navigation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_waypoint_sb3.yaml"),
        help="Path to RL training config YAML.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory for training outputs.",
    )
    args = parser.parse_args(argv)

    full_cfg = _load_yaml(args.config)
    env_cfg = WaypointEnvConfig(**full_cfg.get("env", {}))
    train_cfg = full_cfg.get("train", {})

    run_dir = _build_run_dir(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_logs_dir = run_dir / "eval_logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(full_cfg, file, sort_keys=False)

    seed = int(train_cfg.get("seed", 7))
    n_envs = int(train_cfg.get("n_envs", 4))
    total_timesteps = int(train_cfg.get("total_timesteps", 200_000))

    def make_env() -> Monitor:
        return Monitor(WaypointEnv(config=env_cfg))

    train_env = DummyVecEnv([make_env for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env])

    use_norm_obs = bool(train_cfg.get("normalize_observation", False))
    use_norm_reward = bool(train_cfg.get("normalize_reward", False))
    if use_norm_obs or use_norm_reward:
        train_env = VecNormalize(
            train_env,
            norm_obs=use_norm_obs,
            norm_reward=use_norm_reward,
            clip_obs=10.0,
            training=True,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=use_norm_obs,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

    tensorboard_log = train_cfg.get("tensorboard_log", "runs/tensorboard")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 1024)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_range=float(train_cfg.get("clip_range", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.0)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        use_sde=bool(train_cfg.get("use_sde", False)),
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(train_cfg.get("checkpoint_freq", 25_000)) // n_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix="ppo_waypoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_logs_dir),
        eval_freq=max(int(train_cfg.get("eval_freq", 10_000)) // n_envs, 1),
        n_eval_episodes=int(train_cfg.get("eval_episodes", 10)),
        deterministic=bool(train_cfg.get("deterministic_eval", True)),
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))
    if isinstance(train_env, VecNormalize):
        train_env.save(str(run_dir / "vecnormalize.pkl"))

    summary = {
        "run_dir": str(run_dir),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "final_model_path": str(final_model_path),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()

