# CMS phase-field inversion core code package

This package contains only the reusable core implementation of the phase-field inversion method.

## Included

- `src/cms_pf/solver.py`: constant-mobility Cahn--Hilliard solver.
- `src/cms_pf/data.py`: trajectory dataset and grouped split utilities.
- `src/cms_pf/descriptors.py`: morphology descriptor extraction.
- `src/cms_pf/model.py`: compact temporal-fusion posterior model.
- `src/cms_pf/model_sota.py`: proposed multiscale temporal encoder and conditional normalizing-flow posterior.
- `tests/test_solver.py`: solver conservation, energy, and symmetry test.
- `environment.yml`: minimal environment specification for the core modules.

## Not included

Paper plotting code, experiment/training/evaluation scripts, baselines, experiment configurations and protocols, raw or processed datasets, checkpoints, numerical results, generated figures, manuscript files, caches, temporary environments, and third-party repositories are intentionally excluded.

## Basic setup

```bash
conda env create -f environment.yml
conda activate cms-pf-core
export PYTHONPATH="$PWD/src"
pytest -q tests/test_solver.py
```

Install the PyTorch build appropriate for the target CPU, CUDA, or Ascend environment before importing the model modules.
