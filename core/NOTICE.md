# Third-party code notice

The modules in this folder are adapted from:

> Wen, X., & Wang, Y. (2024). *Code for: Seismic wave propagation simulation with quantum computing* [Software]. Zenodo. https://doi.org/10.5281/zenodo.14868887

which accompanies:

> Wen, X., & Wang, Y. (2025). Seismic wave propagation simulation with quantum computing. *Journal of Geophysical Research: Machine Learning and Computation*, 2(2), e2024JH000473. https://doi.org/10.1029/2024JH000473

That code is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The files in this folder stay under CC BY 4.0. The MIT license in the repository root covers the rest of the repository.

## What changed

| File | Status |
|---|---|
| `stagger_FD_coeff.py` | Unchanged. Builds the dispersion-preserving linear systems (1D/2D/3D), the objective Φ and the dispersion ratio σ. |
| `utils.py` | Unchanged. Helpers, including `cholesky_solver`, the classical least-squares solver behind the Wang baseline. |
| `QUBO_solvers.py` | Modified. See below. |
| `__init__.py` | Added. The Zenodo release is a flat set of files, but its notebooks import from a `core` package. This file makes the folder that package. |

Changes to `QUBO_solvers.py`, all in `IterSampleSolver`:

1. **`ensemble_k`** (default `1`). If greater than 1, the update direction is the Boltzmann-weighted average of the `K` lowest-energy samples instead of the single best sample. Used for the ESE ablation.
2. **`momentum_beta`** (default `0.0`). If positive, the step size L is replaced by an exponential moving average of past step sizes. Used for the momentum ablation.
3. **`greedy_revert`** (default `False`). If enabled, an iteration that increases the objective is discarded and the solver restarts from the best iterate so far. Used for the greedy-revert ablation.
4. The best iterate is stored with `.copy()`. The original rebinds rather than mutates arrays, so this does not change its results. It makes sure the rollback in (3) restores an independent array.

With the defaults, the solver behaves exactly like the original.

## Known upstream quirks

- `stagger_FD_coeff.py` uses `np.math.factorial`, which NumPy 2.0 removed. `requirements.txt` therefore pins `numpy<2`.
- `QUBO_solvers.py` imports `dwave.system` at module level, so `dwave-system` must be installed even though only the simulated-annealing sampler is used here.
