#!/usr/bin/env python
"""
Dispersion-Optimized Novelty Studies
=====================================

Key insight: Wang and QA both minimize Φ = ||Ax-b||² (linearized system).
But the REAL metric that matters for seismic simulation is σ_MAE —
the dispersion error |σ(θ,kh) - 1|. Minimizing Φ ≈ minimizing σ_MAE,
but NOT perfectly. A solution with slightly higher Φ can have LOWER σ_MAE.

Three genuinely new approaches:

1. QA Diversity Pool (QDP):
   Run QA N times independently. Each run finds a DIFFERENT approximate
   solution (QA is stochastic). Evaluate ALL solutions on σ_MAE. Pick best.
   QA's "weakness" (stochasticity) becomes a STRENGTH (diversity).

2. Dispersion-Optimized Post-Processing (DOPP):
   Take Wang/QA solution as starting point. Run Nelder-Mead optimization
   DIRECTLY on σ_MAE (not Φ). This is a novel contribution: nobody in the
   paper optimizes σ_MAE directly.

3. Full Pipeline (QDP + DOPP):
   Generate diverse candidates via QA pool → pick best by σ_MAE → polish
   with Nelder-Mead on σ_MAE. This is a complete hybrid pipeline.

Usage:
    python experiments/Dispersion_Optimized_Novelty.py              # all dims
    python experiments/Dispersion_Optimized_Novelty.py --dim 1       # 1D only
    python experiments/Dispersion_Optimized_Novelty.py --dim 1 2 3   # specific dims
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize as scipy_minimize

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, FIGURES_DIR, RESULTS_DIR, SolverSettings,
    aggregate_mean_std, compute_metrics, make_linear_system,
    solve_taylor, solve_wang, solve_quantum, write_csv,
    _direction_grid,
)


CONFIGS = {
    1: {"tag": "1d", "v": 4500, "h": 10, "tao": 0.001,
        "freq_max": 174, "M": 8, "epsilon": 1e-8,
        "pool_size": 10, "repeat": 5},
    2: {"tag": "2d", "v": 4500, "h": 20, "tao": 0.001,
        "freq_max": 90, "M": 10, "epsilon": 1e-10,
        "pool_size": 8, "repeat": 3},
    3: {"tag": "3d", "v": 2500, "h": 10, "tao": 0.001,
        "freq_max": 100, "M": 10, "epsilon": 1e-10,
        "pool_size": 6, "repeat": 3},
}


def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


def _eval(dim, cfg, coeff, linear_sys):
    """Compute all metrics for a coefficient vector."""
    return compute_metrics(
        dim=dim, linear_sys=linear_sys, coeff=coeff,
        v=cfg["v"], h=cfg["h"], tao=cfg["tao"])


def _sigma_mae_objective(coeff_flat, linear_sys, dim, cfg):
    """
    Compute σ_MAE directly from coefficients.
    This is what we optimize with Nelder-Mead.
    """
    coeff = coeff_flat.reshape(-1, 1)
    kh = np.linspace(PI / 1000, PI, 500)  # fewer points for speed
    r = cfg["v"] * cfg["tao"] / cfg["h"]

    sigma_errors = []
    for theta, phi in _direction_grid(dim):
        if dim == 1:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff)).reshape(-1)
        elif dim == 2:
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
        else:
            sigma = np.asarray(
                linear_sys.cal_sigma(r, kh, coeff, theta=theta, phi=phi)
            ).reshape(-1)
        sigma_errors.append(np.abs(sigma - 1.0))

    # Check for NaN/Inf (unstable coefficients)
    all_err = np.concatenate(sigma_errors)
    if np.any(np.isnan(all_err)) or np.any(np.isinf(all_err)):
        return 1e10
    return float(np.mean(all_err))


# ═══════════════════════════════════════════════════════════════════
# Method 1: QA Diversity Pool (QDP)
# ═══════════════════════════════════════════════════════════════════
def run_qa_diversity_pool(dim, cfg, pool_size=10):
    """
    Run QA 'pool_size' times independently, evaluate each on σ_MAE,
    pick the best. QA's stochasticity generates diverse candidates.
    """
    taylor = solve_taylor(cfg["M"])
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    max_iter = 30 if dim == 3 else 20

    linear_sys = make_linear_system(
        dim=dim, v=cfg["v"], h=cfg["h"],
        tao=cfg["tao"], freq_max=cfg["freq_max"])

    best_coeff = None
    best_mae = np.inf
    best_obj = np.inf
    all_maes = []
    total_runtime = 0.0

    start = time.perf_counter()
    for i in range(pool_size):
        qa = _silent(
            solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"],
            tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
            max_iter_num=max_iter, alpha=qa_alpha, alpha_decay=0.1,
            beta=1.0, epsilon=cfg["epsilon"],
            coeff_init=taylor["coeff"],
            settings=SolverSettings(
                num_sample=100, R=10, fixed_point=1, inner_maxiter=3))

        m = _eval(dim, cfg, qa["coeff"], qa["linear_sys"])
        all_maes.append(m["sigma_mae"])

        if m["sigma_mae"] < best_mae:
            best_mae = m["sigma_mae"]
            best_obj = m["final_obj"]
            best_coeff = qa["coeff"]
            linear_sys = qa["linear_sys"]

    runtime = time.perf_counter() - start

    return {
        "coeff": best_coeff, "linear_sys": linear_sys,
        "runtime_sec": runtime, "method": "QDP",
        "best_mae": best_mae, "best_obj": best_obj,
        "all_maes": all_maes, "pool_size": pool_size,
    }


# ═══════════════════════════════════════════════════════════════════
# Method 2: Dispersion-Optimized Post-Processing (DOPP)
# ═══════════════════════════════════════════════════════════════════
def run_dopp(dim, cfg, start_coeff, linear_sys, label="Wang"):
    """
    Take an initial coefficient vector and optimize σ_MAE directly
    using Nelder-Mead. This exploits the gap between Φ and σ_MAE.
    """
    start = time.perf_counter()

    x0 = start_coeff.flatten()

    result = scipy_minimize(
        _sigma_mae_objective,
        x0=x0,
        args=(linear_sys, dim, cfg),
        method="Nelder-Mead",
        options={
            "maxiter": 500,
            "maxfev": 2000,
            "xatol": 1e-10,
            "fatol": 1e-10,
            "adaptive": True,
        },
    )

    runtime = time.perf_counter() - start
    optimized_coeff = result.x.reshape(-1, 1)

    return {
        "coeff": optimized_coeff, "linear_sys": linear_sys,
        "runtime_sec": runtime, "method": f"DOPP({label})",
        "nfev": result.nfev, "success": result.success,
    }


# ═══════════════════════════════════════════════════════════════════
# Method 3: Full Pipeline (QDP + DOPP)
# ═══════════════════════════════════════════════════════════════════
def run_full_pipeline(dim, cfg, pool_size=10):
    """
    Complete pipeline:
    1. Generate diverse QA candidates (pool)
    2. Pick best by σ_MAE
    3. Polish with Nelder-Mead on σ_MAE
    """
    # Step 1+2: QDP
    qdp = run_qa_diversity_pool(dim, cfg, pool_size=pool_size)

    # Step 3: DOPP on the best QDP candidate
    dopp = run_dopp(dim, cfg, qdp["coeff"], qdp["linear_sys"], label="QDP")

    return {
        "coeff": dopp["coeff"], "linear_sys": dopp["linear_sys"],
        "runtime_sec": qdp["runtime_sec"] + dopp["runtime_sec"],
        "method": "Full_Pipeline",
    }


# ═══════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════
def run_experiment(dims=None):
    if dims is None:
        dims = [1, 2, 3]

    rows = []

    for dim in dims:
        cfg = CONFIGS[dim]
        taylor = solve_taylor(cfg["M"])
        max_iter = 30 if dim == 3 else 20
        qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims
        n_repeat = cfg["repeat"]
        pool_size = cfg["pool_size"]

        print(f"\n{'='*70}")
        print(f"  DIMENSION {dim}D — {n_repeat} repeats, pool_size={pool_size}")
        print(f"{'='*70}")

        # ── Wang baseline ──
        print(f"  [Wang] baseline...", end=" ", flush=True)
        wang = _silent(solve_wang, dim=dim, v=cfg["v"], h=cfg["h"],
                       tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
                       max_iter_num=max_iter, alpha=1.0, alpha_decay=0.1,
                       beta=1.0, epsilon=cfg["epsilon"],
                       coeff_init=taylor["coeff"])
        wang_m = _eval(dim, cfg, wang["coeff"], wang["linear_sys"])
        print(f"Φ={wang_m['final_obj']:.4e}  σ_MAE={wang_m['sigma_mae']:.6f}")
        rows.append({
            "dim": dim, "method": "Wang", "repeat": 0,
            "final_obj": wang_m["final_obj"],
            "sigma_mae": wang_m["sigma_mae"],
            "sigma_max": wang_m["sigma_max_abs"],
            "runtime_sec": wang["runtime_sec"],
        })

        # ── DOPP on Wang ── (single run — deterministic)
        print(f"  [DOPP(Wang)] polishing Wang with Nelder-Mead on σ_MAE...",
              end=" ", flush=True)
        dopp_wang = run_dopp(dim, cfg, wang["coeff"], wang["linear_sys"], "Wang")
        dopp_wang_m = _eval(dim, cfg, dopp_wang["coeff"], dopp_wang["linear_sys"])
        improvement = (wang_m["sigma_mae"] - dopp_wang_m["sigma_mae"]) / wang_m["sigma_mae"] * 100
        print(f"σ_MAE={dopp_wang_m['sigma_mae']:.6f} "
              f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.1f}%)")
        rows.append({
            "dim": dim, "method": "DOPP(Wang)", "repeat": 0,
            "final_obj": dopp_wang_m["final_obj"],
            "sigma_mae": dopp_wang_m["sigma_mae"],
            "sigma_max": dopp_wang_m["sigma_max_abs"],
            "runtime_sec": dopp_wang["runtime_sec"] + wang["runtime_sec"],
        })

        for rep in range(n_repeat):
            print(f"\n  --- Repeat {rep+1}/{n_repeat} ---")

            # ── QA baseline ──
            print(f"    [QA] baseline...", end=" ", flush=True)
            qa = _silent(
                solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"],
                tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter, alpha=qa_alpha, alpha_decay=0.1,
                beta=1.0, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
                settings=SolverSettings(
                    num_sample=100, R=10, fixed_point=1, inner_maxiter=3))
            qa_m = _eval(dim, cfg, qa["coeff"], qa["linear_sys"])
            print(f"Φ={qa_m['final_obj']:.4e}  σ_MAE={qa_m['sigma_mae']:.6f}")
            rows.append({
                "dim": dim, "method": "QA_baseline", "repeat": rep,
                "final_obj": qa_m["final_obj"],
                "sigma_mae": qa_m["sigma_mae"],
                "sigma_max": qa_m["sigma_max_abs"],
                "runtime_sec": qa["runtime_sec"],
            })

            # ── DOPP on QA ──
            print(f"    [DOPP(QA)] polishing QA...", end=" ", flush=True)
            dopp_qa = run_dopp(dim, cfg, qa["coeff"], qa["linear_sys"], "QA")
            dopp_qa_m = _eval(dim, cfg, dopp_qa["coeff"], dopp_qa["linear_sys"])
            improvement = (qa_m["sigma_mae"] - dopp_qa_m["sigma_mae"]) / qa_m["sigma_mae"] * 100
            print(f"σ_MAE={dopp_qa_m['sigma_mae']:.6f} "
                  f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.1f}%)")
            rows.append({
                "dim": dim, "method": "DOPP(QA)", "repeat": rep,
                "final_obj": dopp_qa_m["final_obj"],
                "sigma_mae": dopp_qa_m["sigma_mae"],
                "sigma_max": dopp_qa_m["sigma_max_abs"],
                "runtime_sec": dopp_qa["runtime_sec"] + qa["runtime_sec"],
            })

            # ── QA Diversity Pool ──
            print(f"    [QDP] diversity pool (N={pool_size})...", end=" ", flush=True)
            qdp = run_qa_diversity_pool(dim, cfg, pool_size=pool_size)
            qdp_m = _eval(dim, cfg, qdp["coeff"], qdp["linear_sys"])
            print(f"σ_MAE={qdp_m['sigma_mae']:.6f}  "
                  f"(pool range: {min(qdp['all_maes']):.6f}–{max(qdp['all_maes']):.6f})")
            rows.append({
                "dim": dim, "method": "QDP", "repeat": rep,
                "final_obj": qdp_m["final_obj"],
                "sigma_mae": qdp_m["sigma_mae"],
                "sigma_max": qdp_m["sigma_max_abs"],
                "runtime_sec": qdp["runtime_sec"],
            })

            # ── Full Pipeline (QDP + DOPP) ──
            print(f"    [Full] QDP + DOPP...", end=" ", flush=True)
            full = run_full_pipeline(dim, cfg, pool_size=pool_size)
            full_m = _eval(dim, cfg, full["coeff"], full["linear_sys"])
            vs_wang = (wang_m["sigma_mae"] - full_m["sigma_mae"]) / wang_m["sigma_mae"] * 100
            print(f"σ_MAE={full_m['sigma_mae']:.6f}  "
                  f"(vs Wang: {'↓' if vs_wang > 0 else '↑'}{abs(vs_wang):.1f}%)")
            rows.append({
                "dim": dim, "method": "Full_Pipeline", "repeat": rep,
                "final_obj": full_m["final_obj"],
                "sigma_mae": full_m["sigma_mae"],
                "sigma_max": full_m["sigma_max_abs"],
                "runtime_sec": full["runtime_sec"],
            })

    # Save results. Rows for dimensions not re-run in this invocation are kept,
    # so `--dim 1` followed by `--dim 3` does not overwrite the 1D results.
    raw_path = RESULTS_DIR / "dispersion_novelty_raw.csv"
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if int(r["dim"]) not in dims]
        rows = sorted(kept + rows, key=lambda r: int(r["dim"]))
    write_csv(raw_path, rows)

    summary = aggregate_mean_std(
        rows, group_keys=["dim", "method"],
        value_keys=["final_obj", "sigma_mae", "sigma_max", "runtime_sec"])
    summary_path = RESULTS_DIR / "dispersion_novelty_summary.csv"
    write_csv(summary_path, summary)

    # ── Print results table ──
    print(f"\n{'='*100}")
    print(f"{'Dim':>3} | {'Method':<18} | {'Φ (mean)':>12} | "
          f"{'σ_MAE (mean)':>14} | {'σ_MAX (mean)':>14} | {'Runtime':>10}")
    print("-" * 100)
    for r in sorted(summary,
                    key=lambda x: (int(x["dim"]), float(x["sigma_mae_mean"]))):
        print(f"{r['dim']:>3} | {r['method']:<18} | "
              f"{float(r['final_obj_mean']):>12.4e} | "
              f"{float(r['sigma_mae_mean']):>14.8f} | "
              f"{float(r['sigma_max_mean']):>14.8f} | "
              f"{float(r['runtime_sec_mean']):>8.1f}s")
    print("=" * 100)

    # ── Generate figures ──
    for dim in dims:
        dim_summary = [r for r in summary if int(r["dim"]) == dim]
        dim_summary = sorted(dim_summary,
                            key=lambda x: float(x["sigma_mae_mean"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

        methods = [r["method"] for r in dim_summary]
        mae_means = [float(r["sigma_mae_mean"]) for r in dim_summary]
        mae_stds = [float(r["sigma_mae_std"]) for r in dim_summary]
        obj_means = [float(r["final_obj_mean"]) for r in dim_summary]

        colors = []
        for m in methods:
            if "Wang" in m and "DOPP" not in m and "QDP" not in m:
                colors.append("#4CAF50")
            elif m == "QA_baseline":
                colors.append("#2196F3")
            elif "DOPP" in m:
                colors.append("#E91E63")
            elif m == "QDP":
                colors.append("#FF9800")
            elif m == "Full_Pipeline":
                colors.append("#9C27B0")
            else:
                colors.append("#666")

        # σ_MAE comparison
        bars = ax1.bar(range(len(methods)), mae_means, yerr=mae_stds,
                       capsize=4, color=colors, edgecolor="black",
                       linewidth=0.6, alpha=0.9)
        wang_mae = next((float(r["sigma_mae_mean"]) for r in dim_summary
                        if r["method"] == "Wang"), None)
        if wang_mae:
            ax1.axhline(wang_mae, color="#4CAF50", linestyle="--",
                       linewidth=2, label=f"Wang σ_MAE={wang_mae:.6f}",
                       alpha=0.8)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax1.set_ylabel("σ_MAE (lower is better)", fontsize=10)
        ax1.set_title(f"{dim}D: Dispersion Error (σ_MAE)", fontsize=12,
                     fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

        # Φ comparison
        dim_summary_by_obj = sorted(dim_summary,
                                    key=lambda x: float(x["final_obj_mean"]))
        methods2 = [r["method"] for r in dim_summary_by_obj]
        obj_means2 = [float(r["final_obj_mean"]) for r in dim_summary_by_obj]
        colors2 = []
        for m in methods2:
            if "Wang" in m and "DOPP" not in m and "QDP" not in m:
                colors2.append("#4CAF50")
            elif m == "QA_baseline":
                colors2.append("#2196F3")
            elif "DOPP" in m:
                colors2.append("#E91E63")
            elif m == "QDP":
                colors2.append("#FF9800")
            elif m == "Full_Pipeline":
                colors2.append("#9C27B0")
            else:
                colors2.append("#666")

        ax2.bar(range(len(methods2)), obj_means2, color=colors2,
                edgecolor="black", linewidth=0.6, alpha=0.9)
        ax2.set_xticks(range(len(methods2)))
        ax2.set_xticklabels(methods2, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("Final Objective Φ (log, lower is better)", fontsize=10)
        ax2.set_yscale("log")
        ax2.set_title(f"{dim}D: Objective Value (Φ)", fontsize=12,
                     fontweight="bold")
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

        fig.suptitle(f"Dimension {dim}D: Novel Dispersion-Optimized Results",
                    fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"dispersion_novelty_dim{dim}.png",
                   dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: figures/dispersion_novelty_dim{dim}.png")

    print(f"\nDone! Results in: {raw_path.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, nargs="+", default=None,
                        choices=[1, 2, 3])
    args = parser.parse_args()
    run_experiment(dims=args.dim)
