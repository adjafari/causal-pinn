# causal-pinn

[![Build Status](https://github.com/adjafari/causal-pinn/actions/workflows/ci.yml/badge.svg )](https://github.com/adjafari/causal-pinn/actions/workflows/ci.yml )

A lightweight demonstration of causal physics-informed reconstruction for a
one-dimensional heat-equation field. The example generates sparse noisy
measurements, reconstructs the full field with several assumptions, and plots
side-by-side qualitative and quantitative comparisons.

## What is visualized

The plotting workflow compares:

1. Ground-truth synthetic data.
2. Sparse noisy observations.
3. Reconstruction without a physics constraint (`Data only`).
4. Reconstruction with a physics constraint (`Physics constrained`).
5. Reconstruction with causal time weighting but without physics (`Causal data`).
6. Reconstruction with both causal weighting and a physics constraint (`Causal PINN`).

The generated figure also includes bar charts for RMSE, relative L2 error, and
heat-equation residual so the reconstruction performance can be compared at a
glance.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/plot_causal_pinn_demo.py
```

The visualization is written to:

```text
outputs/causal_pinn_comparison.svg
```

## Development checks

```bash
python -m compileall causal_pinn scripts tests
pytest
```
