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

## Checks

Three commands, and CI runs the same three on every pull request. A change that fails
any of them cannot be merged, so it is worth running them before pushing rather than
finding out on GitHub.

```bash
uv run pytest -q                 # the suite
uv run ruff check .              # lint
uv run ruff format --check .     # formatting
```

`ruff check . --fix` and `ruff format .` fix most of what they find. The rule set lives
in `pyproject.toml` under `[tool.ruff.lint]`; where a rule is switched off there is a
comment saying why, so if one gets in your way read that first rather than reaching for
a blanket `noqa`.

Useful while working on one thing:

```bash
uv run pytest test/test_orc_sail.py -v    # one file
uv run pytest -k "keel" -v                # tests matching a name
uv run pytest --lf                        # only what failed last time
```

`FIXLIST.md` in the repo root is the standing list of what to do next and why.

## Adding a model

A boat is assembled from **parts**, and each part is a slot you can swap a model into.
Nothing central holds a list of what exists: a model registers itself, and a config names
the one it wants.

| part | models it currently offers |
| --- | --- |
| `sail` | `basic`, `hybrid`, `orc_main`, `orc_w_jib` |
| `keel` | `basic`, `finite_span` |
| `rudder` | `basic`, `finite_span` |
| `hull` | `basic`, `linear`, `quadratic` |
| `friction` | `flat`, `hughes` |

To add one, write the class and decorate it. There is no table to update:

```python
@register(
    "keel",
    "finite_span",
    name="Finite span",
    blurb="Lifting-line induced drag and post-stall blend",
)
class FiniteSpanKeel(BasicKeel):
    REQUIRES = (("span", "effective_aspect_ratio"),)   # one of each group must be present
    REFUSES = ()                                       # keys this model does not read
    DEFAULTS = {"alpha_sep_deg": 25.0}                 # physics constants it owns
```

`FiniteSpanRudder` next door is the same shape with `REFUSES = CLAMP_KEYS` — it blends
past stall, so the clamp keys the basic rudder uses would do nothing on it and are
rejected rather than silently ignored.

`name` and `blurb` are what the shipyard screen shows, so a new model arrives already
described. `REQUIRES` and `REFUSES` are what let the shipyard grey out a model a given
boat cannot run, *with the reason*, without building anything — and each model's
`_check_keys` enforces that same `REFUSES` list at construction, so the screen and the
boat cannot disagree.

Registration happens on import, so the module must be imported by
`sailbench/foils/__init__.py` or `sailbench/dynamics/__init__.py`. **A model in a file
nobody imports is a model the registry has never heard of.**

Prefer a new model over a subclass that hardcodes a combination. Skin friction is the
worked example: rather than a hull subclass that fixes the Hughes line and then has to
refuse a `friction_model` key, friction is its own part, so any friction law composes
with any hull and the shipyard can offer the choice.

### How a boat config names models

Shared keys apply to whichever model runs. Each model's own parameters go in a block it
owns, so no model is handed another's keys:

```yaml
keel:
  area: 0.1225          # shared: every keel model reads these
  x_pos: 0.184
  model: finite_span    # which one to build
  models:
    basic: {}           # empty is fine — a model with no parameters of its own
    finite_span:
      span: 0.700
      end_plate_factor: 2.0
```

A section with no `models:` block is read exactly as before, so older configs keep working
and a boat can be migrated one part at a time.

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

3. Open http://localhost:8000/ in your browser.

`web/` serves four separate front-ends over a shared core, and the page at the root is a
launcher linking to all of them:

| page | what it is | needs the backend? |
| --- | --- | --- |
| `/sim/` | the 3D simulator, sailing whatever `--config` was launched with | yes |
| `/shipyard/` | the same sim, with the boat-assembly screen in front of it | yes |
| `/game/` | the scored waypoint task — manual, autopilot or a trained policy | only for `backend` mode |
| `/physics/` | the physics lab, every model pulled apart | no |

The pages that talk to the backend use port 8765. To point one somewhere else -- a
second checkout running its own backend, say -- pass the port in the URL of the page
itself: `http://localhost:8000/sim/?port=8766`. `?host=` reaches another machine and
`?ws=` replaces the whole socket URL. The launcher does not forward these, so put them
on the app's own URL rather than on `/`.

`/shipyard/` opens on the **shipyard**: pick a boat (any boat config under
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
3. Open http://localhost:8000/sim/. The red/yellow marker shows the current waypoint.


## Test bench

`web/game/` is where a controller gets driven against the real physics. It runs the same
waypoint task policies are trained on — `sailbench/rl/envs/waypoint_env.py` — with the same
observation vector, the same normalised `[rudder, sheet]` action and the same termination rules,
ported to the browser in `web/game/task.js`.

```bash
cd web
python -m http.server 8000
```

Open <http://localhost:8000/game/>.

### Three drivers, scored the same way

| mode | key | what it is |
| --- | --- | --- |
| `manual` | `1` | you steer with the arrow keys — a human baseline |
| `autopilot` | `2` | the hand-written controller in `web/game/autopilot.js` |
| `backend` | `3` | mirrors `web_runner`, so a trained checkpoint runs here |

Backend mode attaches to the same WebSocket the 3D viewer uses, so a policy is tested by starting
the runner with a checkpoint:

```bash
uv run python -m sailbench.sim.web_runner \
  --config basic_sailbot.yaml \
  --policy-model runs/<run_name>/best_model/best_model.zip \
  --policy-config configs/rl_waypoint_sb3.yaml
```

Python owns the physics in that mode and the page mirrors it, but the episode is scored
identically, so a policy and the in-browser controllers are directly comparable.

### What it measures

Per episode: step count against the 1000-step budget, elapsed time, distance to goal, result
(`success` inside 1.5 m, `failure` beyond 80 m, `timeout`), and the number of tacks and gybes.
Across episodes, per driver: success rate, mean time on the runs that succeeded, and path
efficiency — straight-line distance over distance actually sailed. Leave `auto-next` on and a
controller will keep running episodes until the sample is worth something.

The **observation → action** panel shows all 13 observation elements and the 2 action elements as
live bars, so you can see what the controller is being fed and what it does with it.

### Using it

- Click the water to place a waypoint, `N` to sample a new one the way the env does.
- `←`/`→` rudder, `↑`/`↓` sheet when you are driving, `space` to pause, `R` to reset the boat.
- `F` force vectors, `Q` wind, `T` trail, `M` map, `O` auto-next.
- Wind speed and direction are adjustable live; the rose shows true and apparent wind against the
  bow with the no-go zone shaded.
- 400 × 400 m of open water, north-up, with the camera following the boat.
- The boat is whichever config you pick. Hull proportions, appendage sizes and mast position are
  read straight from `configs/*.yaml`, so Flingo Floty is drawn short, beamy and deep with her
  small rig and long rudder arm — because that is what her numbers say.

### Swapping in your own controller

Anything with `act(observation) -> [rudder, sheet]` works. `autopilot.js` is the worked example;
point `main.js` at yours instead:

```js
const controller = createMyController(config);
// ...
const action = controller.act(observation(state, waypoint, wind, lastAction));
```

Per-boat colours and sail marks live in `web/game/boats.js` and do not touch the physics.

## Physics Lab

`web/physics/` is a second, standalone UI that pulls the boat physics apart — one section
per model, each figure computed live by a JavaScript port of the real Python models.

It needs no backend. Serve `web/` and open it:

```bash
cd web
python -m http.server 8000
```

Then open <http://localhost:8000/physics/> (there is also a **Physics Lab →** link in the 3D
simulator's HUD).

**You can sail it.** The force board has a *sail it* mode: arrow keys steer exactly as they do in
the 3D view (`←`/`→` rudder, `↑`/`↓` sheet), the boat is integrated in the page with the same RK4
step and the same servo model, and the tells light up the moment each updated model engages —
sheet on the stop, sail luffing, rudder angle of attack clamped, keel stalled. Every other figure
on the page tracks the boat while you sail.

What it covers:

- **Geometric sheeting** and the **luff ramp** (`b522bcd`) — the sheet as a stop rather than a
  commanded angle, and lift fading smoothly instead of switching off
- **Geometry-derived hull drag** (`2741bc5`) — `L`/`B`/`T` sliders driving `k_u`, `k_v`, `k_r`
- **Rudder authority and the helm servo** — local inflow including yaw rate, the angle-of-attack
  and coefficient clamps, deadband / rate limit / self-centring
- **Keel, and the NeuralFoil polars** every foil actually evaluates
- **Flingo Floty** (`9c6442c`) — spec sheet, to-scale plan view and a parameter diff against
  `basic_sailbot`
- **VPP polars** — ground truth, generated by running the real simulator
- **Mirror check** — the page's models replayed against forces recorded from `SailboatHub._forces`
- **Live telemetry** — attaches to `web_runner` on `:8765` when it is running

### Regenerating the lab's data

`web/shared/physics/data.js` holds the NeuralFoil polars, the parsed configs, the VPP sweeps and the
validation cases. Regenerate it after changing a model or a config:

```bash
uv run python scripts/gen_physics_ui_data.py
```

The sweeps are the slow part (a few minutes); `--skip-polars` rebuilds everything else quickly,
and `--twa-step` / `--sail-step` trade resolution for time.

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

Then run the frontend in another terminal and open `http://localhost:8000/sim/`:

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

### Score the runs so the shipyard can name them

`eval_waypoint_sb3.py` answers a question about one checkpoint. `score_runs.py`
answers the question you have while picking one from a list of nine:

```bash
uv run python scripts/score_runs.py            # score whatever is not scored yet
uv run python scripts/score_runs.py --force    # score it all again
```

It writes `scorecard.json` into each run directory and the shipyard reads it, so
the helm row stops offering `waypoint_ppo_20260919_074021 (best)` and starts
offering `750 k · no-go 3 · best — 1.15 m/s, 37% pinch, trims`. The left of the
dash is a diff of the run's own `config_used.yaml` against the other runs on the
same boat; the right is what the scorecard measured. A star marks the fastest
policy that reaches the mark on each boat, and a checkpoint with no scorecard
says `unscored` rather than guessing.

Every policy is scored on the same fixed-seed episodes, with
`upwind_waypoint_bias` — the one knob that changes what a seed produces — forced
to a common value. Observation scales come from each run's own config, because
those are part of the encoding the policy was trained with.

What it measures, and why not success rate: every Flingo run on disk reaches the
mark essentially every time, so success rate ranks nothing. Mean speed, the
share of steps spent pinching inside 25° of the true wind, and whether the sheet
is pinned at an extreme are what separate these policies — `runs/README.md` has
the argument, and the note on `no_go_zone_penalty` in `configs/flingo_rl.yaml`
has the physics behind it. Episodes the integrator cannot finish are counted and
reported rather than averaged in; a policy that blows the solver up says so on
its own chip.
