# sailbench

## First-Time Setup (Developers)

`sailbench` uses the [uv project manager](https://docs.astral.sh/uv/) for dependency and environment management. Follow the steps below to set up a local development environment.

### Prerequisites
- Python 3.12+

### Setup Instructions

1. **Install Python**  
   If you don’t already have Python installed, download it from: https://www.python.org/downloads/

2. **Install `uv`**  
   Follow the installation instructions here: https://docs.astral.sh/uv/getting-started/installation/

3. **Install project dependencies**  
   From the root of the repository (`sailbench/`), run:
   ```bash
   uv sync

3. Activate your python venv.  
    - On Mac and Linux: ```source .venv/bin/activate```
    - On Windows: ```.\.venv\Scripts\Activate.ps1```

