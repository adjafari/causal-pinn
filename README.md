# causal-pinn

[![Build Status](https://github.com/adjafari/causal-pinn/actions/workflows/ci.yml/badge.svg )](https://github.com/adjafari/causal-pinn/actions/workflows/ci.yml )

A lightweight demonstration of causal physics-informed reconstruction with both
scripted SVG output and a modern Tkinter learning GUI. The examples generate
sparse noisy measurements, reconstruct fields with several assumptions, and
visualize Causal-PINN ideas across 1D signals, 2D image slices, and 3D volume
slices.

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
python -m causal_pinn.gui
```

The first command writes the static visualization to:

```text
outputs/causal_pinn_comparison.svg
```

## Tkinter learning GUI

Launch the interactive explorer with:

```bash
python -m causal_pinn.gui
```

The GUI includes:

- A 1D heat-field example for sparse sensor measurements over space and time.
- A 2D medical-style image-slice inpainting example.
- A 3D volume mid-slice example that mimics tomographic sampling bands.
- Layer toggles for ground truth, sparse observations, and Causal-PINN reconstruction.
- Metric readouts and teaching notes for causal weighting, physics residuals, and collocation points.
- An SVG export action for the original static comparison figure.

## Development checks

```bash
python -m compileall causal_pinn scripts tests
pytest
```
