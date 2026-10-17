#!/usr/bin/env python
"""
In-Band vs Out-of-Band Dispersion Analysis for DOPP
===================================================
sigma_MAE averages |sigma - 1| over the full range kh in (0, pi], but Wang and
QA only optimise the design band kh <= kh_max (set by freq_max). This script
splits the error of each coefficient set into the two regions, and in 3D also
evaluates it on directions that DOPP never optimised:

  phi             objective minimised by Wang/QA (design band, all design directions)
  sigma_mae       full-band error on the evaluation directions (as in the other scripts)
  mae_in_band     mean |sigma - 1| for kh <= kh_max
  max_in_band     max  |sigma - 1| for kh <= kh_max
  mae_out_band    mean |sigma - 1| for kh >  kh_max
  bandwidth_1e-3  largest kh with |sigma - 1| <= 1e-3 for every smaller kh
  heldout_*       3D only: the same metrics on 76 held-out (theta, phi) directions

Wang and DOPP(Wang) are deterministic; the full pipeline is a single stochastic run.

Usage:
    python experiments/Band_Analysis.py              # 1D and 3D
    python experiments/Band_Analysis.py --dim 1      # 1D only

Outputs:
    results/band_analysis.csv
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from experiments_common import PI, RESULTS_DIR, _direction_grid, solve_taylor, solve_wang, write_csv
from Dispersion_Optimized_Novelty import CONFIGS, run_dopp, run_full_pipeline

KH = np.linspace(PI / 1000, PI, 1000)


def _silent(func, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


def _abs_errors(dim, cfg, linear_sys, coeff, directions):
    """|sigma - 1| with shape (len(directions), len(KH))."""
    r = cfg["v"] * cfg["tao"] / cfg["h"]
    rows = []
    for theta, phi in directions:
        if dim == 1:
            sigma = linear_sys.cal_sigma(r, KH, coeff)
        elif dim == 2:
            sigma = linear_sys.cal_sigma(r, KH, coeff, theta=theta)
        else:
            sigma = linear_sys.cal_sigma(r, KH, coeff, theta=theta, phi=phi)
        rows.append(np.abs(np.asarray(sigma).reshape(-1) - 1.0))
    return np.vstack(rows)


def _bandwidth(err, tol=1e-3):
    bad = np.nonzero(err.max(axis=0) > tol)[0]
    if bad.size == 0:
        return float(KH[-1])
    return float(KH[bad[0] - 1]) if bad[0] > 0 else 0.0


def _heldout_directions(dim):
    if dim != 3:
        return []
    grid = np.linspace(0, PI / 2, 9)
    trained = {(round(t, 6), round(p, 6)) for t, p in _direction_grid(3)}
    return [(t, p) for t in grid for p in grid if (round(t, 6), round(p, 6)) not in trained]


def run(dims):
    rows = []
    for dim in dims:
        cfg = CONFIGS[dim]
        max_iter = 30 if dim == 3 else 20
        taylor = solve_taylor(cfg["M"])
        print(f"[{dim}D] Wang, DOPP(Wang) and one full-pipeline run (pool size {cfg['pool_size']})...")
        wang = _silent(solve_wang, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                       freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter,
                       alpha=1.0, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
                       coeff_init=taylor["coeff"])
        dopp_wang = _silent(run_dopp, dim, cfg, wang["coeff"], wang["linear_sys"], "Wang")
        full = _silent(run_full_pipeline, dim, cfg, pool_size=cfg["pool_size"])

        linear_sys = wang["linear_sys"]
        kh_max = cfg["freq_max"] / (cfg["v"] / (2 * cfg["h"])) * PI
        in_band = KH <= kh_max
        heldout = _heldout_directions(dim)

        for method, coeff in [("Taylor", taylor["coeff"]), ("Wang", wang["coeff"]),
                              ("DOPP(Wang)", dopp_wang["coeff"]), ("Full_Pipeline", full["coeff"])]:
            coeff = np.asarray(coeff).reshape(-1, 1)
            err = _abs_errors(dim, cfg, linear_sys, coeff, _direction_grid(dim))
            row = {
                "dim": dim, "method": method, "kh_max": kh_max,
                "phi": float(linear_sys.cal_Phi(coeff)),
                "sigma_mae": float(err.mean()),
                "mae_in_band": float(err[:, in_band].mean()),
                "max_in_band": float(err[:, in_band].max()),
                "mae_out_band": float(err[:, ~in_band].mean()),
                "bandwidth_1e-3": _bandwidth(err),
                "heldout_mae": "", "heldout_mae_in_band": "",
            }
            if heldout:
                h_err = _abs_errors(dim, cfg, linear_sys, coeff, heldout)
                row["heldout_mae"] = float(h_err.mean())
                row["heldout_mae_in_band"] = float(h_err[:, in_band].mean())
            rows.append(row)
            print(f"  {method:14s} phi={row['phi']:.3e}  sigma_mae={row['sigma_mae']:.2e}  "
                  f"in-band={row['mae_in_band']:.2e}  out-of-band={row['mae_out_band']:.2e}  "
                  f"bandwidth={row['bandwidth_1e-3']:.3f}")

    out_path = RESULTS_DIR / "band_analysis.csv"
    write_csv(out_path, rows)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, nargs="+", default=[1, 3], choices=[1, 2, 3])
    args = parser.parse_args()
    run(args.dim)
