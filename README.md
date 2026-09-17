# sailbench
Sailbench is an end-to-end sailing physics simulator made by [Cornell Autonomous Sailboat Team](https://cusail.com/). It is to be used for testing and RL model training for autonomous sailboats.

<img width="755" height="454" alt="image" src="https://github.com/user-attachments/assets/8149ee33-4ae5-45b6-ae8e-e1e794e2095b" />

## First-Time Setup

`sailbench` uses the [uv project manager](https://docs.astral.sh/uv/) for dependency and environment management. Follow the steps below to set up a local development environment.

### Prerequisites
- Python 3.12+
- [UV project manager](https://docs.astral.sh/uv/)

### Setup Instructions

1. **Install Python**  
   If you don’t already have Python installed, download it from `https://www.python.org/downloads/`.

2. **Install `uv`**  
   Follow the installation instructions here: `https://docs.astral.sh/uv/getting-started/installation/`.

3. **Install project dependencies**

   In the terminal, navigate to ```sailbench/```

   For users who just want to run the simulation, run:
   ```bash
   uv sync
   ```

   For developers (linters/type-checking), install the `dev` dependency group:

   ```bash
   uv sync --group dev
   ```

   If you’re working on RL training/evaluation, also install the `rl` group:

   ```bash
   uv sync --group dev --group rl
   ```

5. **[Optional but recommended] Setup VSCode Extensions**   
    I would recommend utilizing VSCode for developing in this project. The two extensions to install are [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (Python formatter and linter) as well as [MyPy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) (Python type checker). This will help keep consistent code quality and style across sailbench.

## Running the Web Simulation

Once you've installed the packages, you can play sailbench with manual control with the following instructions.

1. **Start the sim backend** (from ```sailbench/```):
   ```bash
   uv run python -m sailbench.sim.web_runner --config basic_sailbot.yaml --fps 60
   ```

2. **Start the frontend** (in a separate terminal):
   ```bash
   cd web
   python -m http.server 8000
   ```

3. Open http://localhost:8000 in your browser.

The page talks to the backend on port 8765. To point it somewhere else -- a
second checkout running its own backend, say -- pass the port in the URL:
`http://localhost:8000/?port=8766`. `?host=` reaches another machine and `?ws=`
replaces the whole socket URL.

The page opens on the **shipyard**: pick a boat (any boat config under
`configs/`), then the sail, keel, rudder and hull models and who holds the helm
(you, or a trained policy found under `runs/`), and set sail. Arrow keys move
around, Enter launches, Esc while sailing brings you back to re-rig. Models a
boat's config cannot support are greyed out with the reason. `--config` and
`--policy-model` only set what is preselected.

### Watch a trained RL policy in the web simulation

To run a trained RL model in sailbench, perform the following.

1. Start the backend with an RL checkpoint:
   ```bash
   uv run python -m sailbench.sim.web_runner \
     --config basic_sailbot.yaml \
     --policy-model runs/<run_name>/best_model/best_model.zip \
     --policy-config configs/rl_waypoint_sb3.yaml
   ```
2. In a second terminal:
   ```bash
   cd web
   python -m http.server 8000
   ```
3. Open http://localhost:8000. The red/yellow marker shows the current waypoint.


## RL Training (Gymnasium + SB3)

SailBench includes a Gymnasium continuous-control task (`sailbench.rl.envs.WaypointEnv`) and Stable-Baselines3 scripts for training/evaluating PPO on waypoint navigation.

Install the optional RL dependencies:

```bash
uv sync --group rl
```

### Train (PPO)

Use the default waypoint RL config (`configs/rl_waypoint_sb3.yaml`):

```bash
uv run python scripts/train_waypoint_sb3.py --config configs/rl_waypoint_sb3.yaml
```

Each RL config names the boat it trains under `env.simulator_config`. The
default above trains `basic_sailbot.yaml` (0.75 m² keel, 3.0 m² sail). To train
the measured Flingo Floaty hull, pass the config that points at it:

```bash
uv run python scripts/train_waypoint_sb3.py --config configs/flingo_rl.yaml
```

To watch training live in the web visualizer, enable the training websocket stream:

```bash
uv run python scripts/train_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --watch-web
```

Then run the frontend in another terminal and open `http://localhost:8000`:

```bash
cd web
python -m http.server 8000
```

Notes:

- The live stream is served on `ws://127.0.0.1:8765/sim` by default (same frontend URL as the normal backend).
- Training visualization publishes from env-0 only, so it works with vectorized training (`train.n_envs > 1`).
- You can reduce browser update load with `--watch-stride <N>` (or `train.watch_stride` in YAML).

To continue a stopped run from a checkpoint:

```bash
uv run python scripts/train_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --resume-from runs/<run_name>/checkpoints/ppo_waypoint_<steps>_steps.zip
```

When `--resume-from` is provided, training continues from that model state and keeps timestep counting continuous.

Training outputs are written under `runs/` by default:

- `runs/waypoint_ppo_<timestamp>/best_model/best_model.zip`: best checkpoint per eval callback
- `runs/waypoint_ppo_<timestamp>/checkpoints/`: periodic checkpoints
- `runs/waypoint_ppo_<timestamp>/final_model.zip`: final model after training
- `runs/waypoint_ppo_<timestamp>/vecnormalize.pkl`: VecNormalize stats (only if enabled via config)
- `runs/waypoint_ppo_<timestamp>/config_used.yaml`: the exact config used for the run

Notes for resume:

- Keep `--config` consistent with the original training setup (especially env settings and `train.n_envs`).
- If normalization is enabled, the trainer automatically attempts to load checkpoint stats from the matching file `.../<checkpoint_stem>_vecnormalize.pkl`.

### TensorBoard

If you keep `train.tensorboard_log: runs/tensorboard` (the default), you can launch TensorBoard with:

```bash
uv run tensorboard --logdir runs/tensorboard
```

When training, logs will be written into subdirectories under `runs/tensorboard/`, one per run. You can view your training progress, hyperparameters, and evaluation metrics in TensorBoard at [http://localhost:6006](http://localhost:6006) after launching the command above.


### Evaluate a trained checkpoint

```bash
uv run python scripts/eval_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --model runs/<run_name>/best_model/best_model.zip \
  --vecnormalize runs/<run_name>/vecnormalize.pkl
```

Notes:

- Pass `--vecnormalize` only if your training run produced `vecnormalize.pkl` (i.e., you enabled observation/reward normalization in the training config).
- `eval_waypoint_sb3.py` prints a JSON blob of summary metrics to stdout; use `--output-json <path>` to save them.
