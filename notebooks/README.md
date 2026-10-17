# Upstream notebooks

These are the seven notebooks from Wen & Wang's Zenodo release ([doi:10.5281/zenodo.14868887](https://doi.org/10.5281/zenodo.14868887), CC BY 4.0). They are included so the original experiments can be re-run against the `core/` package in this repository. The Zenodo release is a flat set of files, so its notebooks cannot import `core` as distributed.

The only changes are: outputs cleared, and a provenance note plus a one-line path-setup cell added at the top of each notebook. The code is otherwise identical to the Zenodo version. All seven use the classical simulated-annealing sampler (`neal`) by default. The D-Wave QPU sampler is left commented out, as in the original.

| Notebook | What it does | Our re-run (in `figures/reproduction/`) |
|---|---|---|
| `1D_StaggerGrid_FD.ipynb` | 1D low- and high-velocity media (M = 4, 6, 8; also 10, 16, 20); QA vs least squares | `1d_low_velocity.png`, `1d_high_velocity.png`, `1d_qa_vs_least_squares.png` |
| `2D_StaggerGrid_FD.ipynb` | 2D low- and high-velocity media (M = 10) | `2d_low_velocity.png`, `2d_high_velocity.png` |
| `3D_StaggerGrid_FD.ipynb` | 3D medium | `3d_dispersion.png` |
| `Algorithm_Optimization.ipynb` | Effect of Tikhonov regularization; the new vs. the original iterative QA update; adding the nonlinear objective | `regularization_convergence.png` |
| `Algorithm_Efficiency_eval.ipynb` | Runtime and iteration count versus stencil length | n/a |
| `Comparsion_of_global_convergence_ability.ipynb` | Optimiser trajectories on a 2-D multimodal test function (MATLAB `peaks`) | n/a |
| `Iteration_compare_disp.ipynb` | Objective value and dispersion curves across iterations | n/a |

## Running

From the repository root, after `pip install -r requirements.txt` and `pip install jupyterlab`:

```bash
cd notebooks
jupyter lab
```

The notebooks save their figures and logs into `notebooks/`. Those files are git-ignored.

**Note:** the last cell of `Algorithm_Optimization.ipynb` reads `Inf_reg_202406171657.txt`, the log file from the original authors' run. When you re-run the notebook, change that filename to the `log_filename` created by the earlier cells.
