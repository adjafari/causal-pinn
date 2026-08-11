# causal-pinn

[![Build Status](https://github.com/adjafari/causal-pinn/actions/workflows/ci.yml/badge.svg )](https://github.com/adjafari/causal-pinn/actions/workflows/ci.yml )

A lightweight teaching toolkit for causal and Bayesian physics-informed
reconstruction in one-dimensional heat-transfer problems. The examples generate
sparse noisy measurements, reconstruct hidden temperature fields, and visualize
how data terms, physics residuals, causal time weighting, and Bayesian
uncertainty estimates change inverse-problem behavior.

The modern Bayesian example is framed as an infrared-imaging inverse problem:
a sparse IR camera observes noisy radiance, while the learner infers the
underlying temperature field, thermal diffusivity, emissivity, and uncertainty.

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

## Bayesian infrared inverse-problem lesson

The Bayesian PINN-style workflow in `causal_pinn.demo` is designed for teaching
rather than heavy optimization. It uses a grid of diffusivity hypotheses as a
small ensemble of physics-informed models. Each candidate reconstructs the
temperature field from sparse radiance pixels, receives a likelihood from the
infrared data mismatch and heat-equation residual, and contributes to a
posterior mean and uncertainty map. The resulting SVG shows:

1. The hidden temperature contrast.
2. Sparse noisy infrared radiance measurements.
3. The posterior mean temperature reconstruction.
4. The posterior standard deviation map.
5. Posterior weights over candidate thermal diffusivities.
6. Estimated emissivity, MAP diffusivity, RMSE, and credible-interval coverage.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/plot_causal_pinn_demo.py
```

The visualizations are written to:

```text
outputs/causal_pinn_comparison.svg
outputs/bayesian_infrared_pinn.svg
```

## Development checks

```bash
python -m compileall causal_pinn scripts tests
pytest
```
