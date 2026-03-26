# sailbench

## First-Time Setup (Developers)

`sailbench` uses the [uv project manager](https://docs.astral.sh/uv/) for dependency and environment management. Follow the steps below to set up a local development environment.

### Prerequisites
- Python 3.12+
- (Optional )XFoil (See below, XFoil is only necessary for generating polars)
  

### Setup Instructions

1. **Install Python**  
   If you don’t already have Python installed, download it from: https://www.python.org/downloads/

2. **Install `uv`**  
   Follow the installation instructions here: https://docs.astral.sh/uv/getting-started/installation/

3. **Install project dependencies**  
   From the root of the repository (`sailbench/`), run:
   ```bash
   uv sync

4. **[Optional but recommended] Setup VSCode Extensions**   
    I would recommend utilizing VSCode for developing in this project. The two extensions to install are [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (Python formatter and linter) as well as [MyPy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) (Python type checker). This will help keep consistent code quality and style across sailbench.

## Running the Web Simulation

1. **Start the sim backend** (from the repository root):
   ```bash
   uv run python -m sailbench.sim.web_runner --config basic_sailbot.yaml --fps 60
   ```

2. **Start the frontend** (in a separate terminal, from the `web` directory):
   ```bash
   cd web
   python -m http.server 8000
   ```

3. Open http://localhost:8000 in your browser.

### Watch a trained RL policy in the web simulation

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

### [DEPRECATED: sailbench now runs on neuralfoil] XFoil Integration (Instructions for Windows)
If you are developing new models with new foil types / characteristics, you may run into an issue with Aerosandbox generating new polars:

```Running XFoil to generate polars for Airfoil 'NACA0012'::   0%|          | 0/1 [00:00<?, ?it/s]```

This is because Aerosandbox requires the XFoil software to generate polars, to install head to this link: https://web.mit.edu/drela/Public/web/xfoil/ and download XFOIL6.99.zip. You then must add your XFoil folder to PATH.

## RL Training (Gymnasium + SB3)

Use the default waypoint RL config:

```bash
uv run python scripts/train_waypoint_sb3.py --config configs/rl_waypoint_sb3.yaml
```

Evaluate a trained checkpoint:

```bash
uv run python scripts/eval_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --model runs/<run_name>/best_model/best_model.zip \
  --vecnormalize runs/<run_name>/vecnormalize.pkl
```
