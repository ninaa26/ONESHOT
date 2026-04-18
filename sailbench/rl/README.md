# SailBench RL Overview

This folder contains the RL task implementation used to train waypoint navigation policies with Stable-Baselines3 (PPO).

## What lives where

- `envs/waypoint_env.py`
  - Gymnasium environment (`WaypointEnv`) and config dataclass (`WaypointEnvConfig`).
  - Wraps the SailBench physics core (`SailboatHub`) into a standard RL step/reset API.
- `live_vis.py`
  - Lightweight websocket broadcaster used during training when `--watch-web` is enabled.
  - Streams training state to the existing browser visualizer protocol.
- `__init__.py`
  - Package marker and high-level module description.

Related scripts and config outside this folder:

- `scripts/train_waypoint_sb3.py` for training
- `scripts/eval_waypoint_sb3.py` for evaluation
- `configs/rl_waypoint_sb3.yaml` for environment + PPO hyperparameters
- `scripts/train_waypoint_ga.py` for genetic algorithm (Torch `MLGAgentPolicy` on `WaypointEnv`)
- `configs/rl_waypoint_ga.yaml` / `configs/rl_waypoint_ga_smoke.yaml` for GA hyperparameters

## Environment design (`WaypointEnv`)

`WaypointEnv` is a continuous control task:

- **Action space:** 2D continuous `[-1, 1]`
  - `action[0]` -> rudder command in degrees (`max_rudder_deg`)
  - `action[1]` -> sail sheet limit (mapped from `[-1,1]` to `[0, max_sail_deg]`)
- **Observation space:** 13D normalized vector
  - Relative waypoint position and direction in boat frame
  - Body velocities (`u`, `v`, `r`)
  - Wind direction in boat frame + wind speed
  - Previous action channels (helps policy smooth control)
- **Episode ends when:**
  - success: boat reaches `success_radius_m`
  - failure: boat exceeds `fail_radius_m`
  - truncation: `max_episode_steps`

## Reward structure

Current reward terms (see `step()` in `envs/waypoint_env.py`):

- Positive progress from velocity made good toward waypoint (`vmg_term`)
- Positive reward from distance reduction (`dist_term`)
- Penalty on action movement (`joint_penalty * sum(abs(delta_action))`)
- Constant per-step time penalty
- Terminal bonus/penalty on success/failure

All term values are controlled from `env` in `configs/rl_waypoint_sb3.yaml`.

## Training flow (`scripts/train_waypoint_sb3.py`)

1. Load YAML config.
2. Build `WaypointEnvConfig`.
3. Create vectorized training envs (`DummyVecEnv`) and one eval env.
4. Optionally wrap with `VecNormalize` (obs/reward normalization).
5. Train PPO with checkpoint + evaluation callbacks.
6. Save artifacts under `runs/waypoint_ppo_<timestamp>/`.

Key outputs:

- `final_model.zip`
- `best_model/best_model.zip`
- `checkpoints/ppo_waypoint_*_steps.zip`
- `summary.json`
- `config_used.yaml`
- `vecnormalize.pkl` (when normalization enabled)

## Evaluation flow (`scripts/eval_waypoint_sb3.py`)

Loads a trained model and runs N episodes, reporting:

- mean/std return
- mean episode length
- success rate
- mean final distance
- mean path efficiency
- mean time-to-goal (successful episodes only)

## Live visualization during training

Training can stream env state into the same browser UI used by the normal simulation.

- Enable with `--watch-web` in training script.
- `LiveTrainingVisServer` starts a websocket server (default `127.0.0.1:8765`).
- Only env-0 publishes state frames to avoid mixing multiple parallel env trajectories.
- Messages are built via `sailbench.sim.protocol.make_state_message`, so payload shape matches normal simulation.
- Control mode is tagged as `"training"` for HUD display.

## Typical commands

Train:

```bash
uv run python scripts/train_waypoint_sb3.py --config configs/rl_waypoint_sb3.yaml
```

Train with live web watch:

```bash
uv run python scripts/train_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --watch-web
```

Evaluate:

```bash
uv run python scripts/eval_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --model runs/<run_name>/best_model/best_model.zip
```

## Genetic algorithm (`scripts/train_waypoint_ga.py`)

Torch-based GA with `MLGAgentPolicy` (ReLU hidden stack, `tanh` actions) on the same `WaypointEnv` reward as PPO. Parallel rollout evaluation uses multiple processes on CUDA (see `train.n_workers` in YAML).

Train:

```bash
uv run --group rl python scripts/train_waypoint_ga.py --config configs/rl_waypoint_ga.yaml
```

Checkpoints (every `train.checkpoint_freq_gens` generations, `0` to disable):

- `runs/waypoint_ga_<timestamp>/checkpoints/ga_waypoint_gen_<G>.npz` — `generation`, `best_fitness`, `best_genome`, `obs_dim`, `act_dim`, `hidden_dim`, `num_hidden`, `seed`, `device`

Final artifact:

- `runs/waypoint_ga_<timestamp>/final_model.npz` — same fields as checkpoints for the best genome at end of training

See the repository root `README.md` for more GA options (`--device`, `--n-workers`, smoke config).

## Tuning notes

- If policy jitters controls, increase `env.joint_penalty`.
- If training is slow, reduce eval/checkpoint frequency and avoid `--watch-web` except when debugging.
- Keep config consistent when resuming runs, especially env settings and vectorization count. Weird stuff will happen if not...
