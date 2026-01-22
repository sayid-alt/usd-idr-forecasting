# USD–IDR Forecasting

Time series forecasting pipeline for Indonesian Rupiah (IDR) to US Dollar (USD) using TensorFlow (LSTM/GRU + Conv1D), Keras Tuner, and Weights & Biases (W&B). Data is sourced from Yahoo Finance (`IDR=X`).

## Requirements
- Python `>=3.11`.
- `uv` package manager installed.
- W&B account and API key.

## Install uv
```bash
# macOS (Homebrew)
brew install uv

# Or via the official installer
curl -Ls https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

## Setup with uv sync
```bash
# Clone the repository
git clone https://github.com/sayid-alt/usd-idr-forecasting.git
cd usd-idr-forecasting

# Sync dependencies (reads pyproject.toml / uv.lock, creates .venv)
uv sync

# Optionally activate the venv
source .venv/bin/activate
```
- `uv sync` installs the exact dependency set into `.venv` using `pyproject.toml` and `uv.lock`.
- You can run scripts without activating the venv using `uv run ...`.

## Environment Variables
Create a `.env` file in the project root:
```bash
PROJECT_WORKING_DIR=/absolute/path/to/usd-idr-forecasting
WANDB_IDRX_FORECAST_KEY=your_wandb_api_key
```
- `PROJECT_WORKING_DIR` is used to save datasets/models.
- `WANDB_IDRX_FORECAST_KEY` enables W&B login.

## Configuration
Edit `project_configs.yaml` for window size, epochs, tuner ranges, and dataset period. The loader reads it via `ProjectConfig.from_yaml(...)`.


## Notes
- Console script in `pyproject.toml` maps `usd-idr-forecasting = usd_idr_forecasting:main`, but no `main` is implemented in the package. Use `uv run python ...` commands above.
- If dependency resolution fails during `uv sync`, ensure Python version is `>=3.11` and dependency names in `pyproject.toml` are valid.

## Project Structure
```
src/usd_idr_forecasting/
  data/               # Data loading and W&B artifact helpers
  configs.py          # YAML-backed configuration loader
  models.py           # Model builders and hypermodel
  trainers.py         # Cold start and retraining flows
  tuner.py            # Tuner wrapper
  evaluators.py       # Evaluation utilities
  processors.py       # Split, scaling, windowing pipeline
script/
  prepare_data.py     # Fetch and split data
  preprocessing.py    # Preprocessing CLI stub
datasets/             # Local outputs (originals, splits, testing)
project_configs.yaml  # Main configuration file
pyproject.toml        # Project metadata and dependencies
uv.lock               # Locked dependency set
```