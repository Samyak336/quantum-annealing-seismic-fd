#!/usr/bin/env python
"""
Improved CNAR: Per-Iteration Adaptive Condition-Number-Aware Regularisation
===========================================================================

The original CNAR used a threshold of log10(κ) > 6.0, but actual condition
numbers are 4.4–5.7, so α was always 0 (CNAR never activated).

This script implements three improved strategies:
  A) Lower threshold (4.0) — so CNAR activates in all dimensions
  B) Per-iteration adaptive — recompute κ at each outer iteration
  C) Soft proportional — α = c × log10(κ)/10 (always active, no threshold)
  D) Combined: per-iteration adaptive + ESE K=3

Usage:
    python experiments/CNAR_Improved_Study.py                     # full run
    python experiments/CNAR_Improved_Study.py --quick             # fast test
    python experiments/CNAR_Improved_Study.py --strategy B        # specific strategy
    python experiments/CNAR_Improved_Study.py --strategy all      # all strategies

Outputs:
    results/cnar_improved_raw.csv
    results/cnar_improved_summary.csv
    figures/cnar_improved_obj_dim{1,2,3}.png
    figures/cnar_improved_dispersion_dim{1,2,3}.png
"""

from __future__ import annotations

import argparse, io, sys, time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from neal import SimulatedAnnealingSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, FIGURES_DIR, RESULTS_DIR, SolverSettings,
    aggregate_mean_std, compute_metrics, make_linear_system,
    solve_taylor, solve_wang, write_csv,
)
from core.QUBO_solvers import IterSampleSolver
from core.utils import cholesky_solver

CONFIGS = {
    1: {"tag": "1d_ref", "v": 4500, "h": 10, "tao": 0.001,
        "freq_max": 174, "M": 8, "epsilon": 1e-8, "repeat": 3},
    2: {"tag": "2d_ref", "v": 4500, "h": 20, "tao": 0.001,
        "freq_max": 90, "M": 10, "epsilon": 1e-10, "repeat": 3},
    3: {"tag": "3d_ref", "v": 2500, "h": 10, "tao": 0.001,
        "freq_max": 100, "M": 10, "epsilon": 1e-10, "repeat": 3},
}


def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


# ═══════════════════════════════════════════════════════════════════
# Per-iteration adaptive CNAR solver
# ═══════════════════════════════════════════════════════════════════
def solve_quantum_cnar_adaptive(
    *,
    dim, v, h, tao, freq_max, M,
    max_iter_num, epsilon, coeff_init,
    num_sample=100, R=10, fixed_point=1, inner_maxiter=3,
    strategy="B",
    threshold=4.0, c_factor=0.1,
    ensemble_k=1,
):
    """
    Custom outer loop with per-iteration condition-number-aware
    regularisation.

    Strategies:
      A: Lower threshold (compute κ once, use threshold=4.0)
      B: Per-iteration adaptive (recompute κ each iteration)
      C: Soft proportional (α = c × log10(κ)/10, no threshold)
      D: Per-iteration adaptive + ESE K=3
    """
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao,
                                     freq_max=freq_max)
    coeff_vec = coeff_init.copy()
    if coeff_vec.ndim == 1 or coeff_vec.shape[1] != 1:
        coeff_vec = coeff_vec.reshape(-1, 1)

    history = []
    kappa_history = []

    # Create QA solver
    sampler = SimulatedAnnealingSampler()
    x0 = np.ones(M).reshape(-1, 1) / 100

    ens_k = 3 if strategy == "D" else ensemble_k

    sis = IterSampleSolver(
        sampler=sampler, x0=x0, num_sample=num_sample,
        R=R, fixed_point=fixed_point, maxiter=inner_maxiter,
        codingMod="pn", resultMod="lowest_energy",
        ensemble_k=ens_k,
    )

    # For strategy A: compute κ once
    if strategy == "A":
        A_init, b_init = linear_sys.construct(M, coeff_vec)
        kappa_init = np.linalg.cond(A_init.T @ A_init)
        fixed_alpha = c_factor * max(0, np.log10(kappa_init) - threshold)

    start_time = time.perf_counter()

    for iter_num in range(1, max_iter_num + 1):
        obj = linear_sys.cal_Phi(coeff_vec)
        history.append(float(obj))

        if obj < epsilon:
            break

        # Construct Jacobian at current coefficients
        A, b = linear_sys.construct(M, coeff_vec)
        kappa = np.linalg.cond(A.T @ A)
        log_kappa = np.log10(max(kappa, 1.0))
        kappa_history.append(float(kappa))

        # Compute adaptive alpha based on strategy
        if strategy == "A":
            alpha = fixed_alpha
        elif strategy in ("B", "D"):
            alpha = c_factor * max(0, log_kappa - threshold)
        elif strategy == "C":
            alpha = c_factor * log_kappa / 10.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Decay alpha slightly with iteration (gentler than original 0.1)
        alpha_effective = alpha * (0.7 ** (iter_num - 1))

        # Solve the sub-problem using QA
        # The IterSampleSolver.solve handles the inner iterative loop
        try:
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                coeff_delta = sis.solve(A, b, alpha=alpha_effective)
        except Exception:
            # Fallback: no regularization
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                coeff_delta = sis.solve(A, b, alpha=0.0)

        # Update coefficients
        coeff_vec_new = coeff_vec + coeff_delta

        # Check for stalling
        if np.linalg.norm((coeff_vec - coeff_vec_new) /
                          (np.abs(coeff_vec) + 1e-15)) < epsilon:
            coeff_vec = coeff_vec_new
            break

        coeff_vec = coeff_vec_new

    runtime = time.perf_counter() - start_time

    return {
        "coeff": coeff_vec,
        "runtime_sec": runtime,
        "history": history,
        "kappa_history": kappa_history,
        "iterations": len(history),
        "linear_sys": linear_sys,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════
STRATEGY_COLORS = {
    "wang_only": "forestgreen",
    "qa_baseline": "royalblue",
    "original_cnar": "gray",
    "A_lower_thresh": "#E91E63",
    "B_per_iter": "#FF5722",
    "C_soft": "#9C27B0",
    "D_per_iter_ese": "#00BCD4",
}

STRATEGY_LABELS = {
    "wang_only": "Wang",
    "qa_baseline": "QA (baseline)",
    "original_cnar": "CNAR (orig, α=0)",
    "A_lower_thresh": "CNAR-A (thresh=4)",
    "B_per_iter": "CNAR-B (per-iter)",
    "C_soft": "CNAR-C (soft)",
    "D_per_iter_ese": "CNAR-D (per-iter+ESE)",
}


def plot_cnar_comparison(dim, summary_rows, wang_obj, out_path):
    """Bar chart comparing all CNAR strategies."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)

    names = [STRATEGY_LABELS.get(r["strategy"], r["strategy"])
             for r in summary_rows]
    means = [float(r["final_obj_mean"]) for r in summary_rows]
    stds = [float(r["final_obj_std"]) for r in summary_rows]
    colors = [STRATEGY_COLORS.get(r["strategy"], "#666") for r in summary_rows]

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(wang_obj, color="forestgreen", linestyle="--", linewidth=2,
               label=f"Wang ({wang_obj:.2e})", alpha=0.9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Final Objective Φ (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"{dim}D: Improved CNAR Comparison", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_facecolor("#f8f8f8")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cnar_dispersion(dim, cfg, method_coeffs, out_path):
    """Directional dispersion comparison for improved CNAR methods."""
    kh = np.linspace(PI / 1000, PI, 1000)
    r = cfg["v"] * cfg["tao"] / cfg["h"]

    n = len(method_coeffs)
    fig, axes = plt.subplots(1, min(n, 6), figsize=(5.5 * min(n, 6), 5),
                             dpi=200, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    theta_vals = [0.0, PI/8, PI/4, 3*PI/8]
    phi_vals = [0.0, PI/8, PI/4]
    styles = ["-", "--", ":", "-."]
    angle_colors = ["maroon", "midnightblue", "purple", "teal"]

    for ax, (method_name, data) in zip(axes, method_coeffs.items()):
        ls = data["linear_sys"]
        coeff = data["coeff"]
        base_color = STRATEGY_COLORS.get(method_name, "black")

        if dim == 1:
            sigma = ls.cal_sigma(r, kh, coeff)
            ax.plot(kh, sigma, color=base_color, linewidth=1.8)
        elif dim == 2:
            for theta, sty in zip(theta_vals, styles):
                sigma = ls.cal_sigma(r, kh, coeff, theta=theta)
                ax.plot(kh, sigma, color=base_color, linestyle=sty,
                        linewidth=1.3)
        else:
            for theta, sty in zip(theta_vals, styles):
                for phi, col in zip(phi_vals, angle_colors):
                    sigma = ls.cal_sigma(r, kh, coeff, theta=theta, phi=phi)
                    ax.plot(kh, sigma, color=col, linestyle=sty,
                            linewidth=1.2)

        obj_val = ls.cal_Phi(coeff)
        label = STRATEGY_LABELS.get(method_name, method_name)
        ax.set_title(f"{label}\nΦ={float(obj_val):.2e}", fontsize=9)
        ax.set_xlim([0, PI])
        ax.set_ylim([0.90, 1.05])
        ax.set_xlabel("kh", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_facecolor("#f5f5f5")

    axes[0].set_ylabel("δ(θ,φ)", fontsize=11)
    fig.suptitle(f"{dim}D Dispersion: Improved CNAR Strategies",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════
def run_improved_cnar(strategies=None, repeat_override=None,
                      max_iter=20, num_sample=100, quick=False):
    """Run all improved CNAR strategies and generate comparison figures."""
    if strategies is None:
        strategies = ["A", "B", "C", "D"]

    if quick:
        repeat_override = repeat_override or 1
        max_iter = min(max_iter, 10)
        num_sample = min(num_sample, 50)

    rows = []
    dispersion_data = {}  # dim -> {strategy: {coeff, linear_sys}}

    for dim, cfg in CONFIGS.items():
        taylor = solve_taylor(cfg["M"])
        qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
        max_iter_dim = 30 if dim == 3 and not quick else max_iter
        n_repeat = repeat_override or cfg["repeat"]
        dispersion_data[dim] = {}

        print(f"\n{'='*60}")
        print(f"  Dimension {dim}D")
        print(f"{'='*60}")

        # 1) Wang baseline
        print(f"  Wang baseline...", end=" ", flush=True)
        wang = _silent(solve_wang, dim=dim, v=cfg["v"], h=cfg["h"],
                       tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
                       max_iter_num=max_iter_dim, alpha=1.0, alpha_decay=0.1,
                       beta=1.0, epsilon=cfg["epsilon"],
                       coeff_init=taylor["coeff"])
        wang_m = compute_metrics(dim=dim, linear_sys=wang["linear_sys"],
                                 coeff=wang["coeff"], v=cfg["v"],
                                 h=cfg["h"], tao=cfg["tao"])
        print(f"Φ={wang_m['final_obj']:.4e}")
        wang_obj = wang_m["final_obj"]
        dispersion_data[dim]["wang_only"] = {
            "coeff": wang["coeff"], "linear_sys": wang["linear_sys"]}

        # 2) QA baseline (original, no CNAR)
        for rep in range(n_repeat):
            print(f"  QA baseline rep {rep+1}...", end=" ", flush=True)
            from experiments_common import solve_quantum
            qa_base = _silent(
                solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"],
                tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter_dim, alpha=qa_alpha, alpha_decay=0.1,
                beta=1.0, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
                settings=SolverSettings(num_sample=num_sample, R=10,
                                        fixed_point=1, inner_maxiter=3))
            qa_m = compute_metrics(dim=dim, linear_sys=qa_base["linear_sys"],
                                   coeff=qa_base["coeff"], v=cfg["v"],
                                   h=cfg["h"], tao=cfg["tao"])
            print(f"Φ={qa_m['final_obj']:.4e}")
            rows.append({
                "dim": dim, "strategy": "qa_baseline", "repeat": rep,
                "final_obj": qa_m["final_obj"],
                "sigma_mae": qa_m["sigma_mae"],
                "runtime_sec": qa_base["runtime_sec"],
            })
            if rep == 0:
                dispersion_data[dim]["qa_baseline"] = {
                    "coeff": qa_base["coeff"],
                    "linear_sys": qa_base["linear_sys"]}

        # 3) Original CNAR (threshold=6.0, always zero)
        ls_check = make_linear_system(dim=dim, v=cfg["v"], h=cfg["h"],
                                       tao=cfg["tao"],
                                       freq_max=cfg["freq_max"])
        A_check, _ = ls_check.construct(cfg["M"], taylor["coeff"])
        kappa_init = np.linalg.cond(A_check.T @ A_check)
        orig_alpha = 0.1 * max(0, np.log10(kappa_init) - 6.0)
        print(f"  Original CNAR: κ={kappa_init:.2e}, "
              f"log10(κ)={np.log10(kappa_init):.2f}, α={orig_alpha:.4f}")

        # 4) Improved CNAR strategies
        for strat in strategies:
            for rep in range(n_repeat):
                strat_name = {
                    "A": "A_lower_thresh", "B": "B_per_iter",
                    "C": "C_soft", "D": "D_per_iter_ese"
                }[strat]
                print(f"  Strategy {strat} ({strat_name}) "
                      f"rep {rep+1}...", end=" ", flush=True)

                result = solve_quantum_cnar_adaptive(
                    dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                    freq_max=cfg["freq_max"], M=cfg["M"],
                    max_iter_num=max_iter_dim, epsilon=cfg["epsilon"],
                    coeff_init=taylor["coeff"],
                    num_sample=num_sample, R=10, fixed_point=1,
                    inner_maxiter=3, strategy=strat,
                    threshold=4.0, c_factor=0.1,
                )
                m = compute_metrics(
                    dim=dim, linear_sys=result["linear_sys"],
                    coeff=result["coeff"], v=cfg["v"],
                    h=cfg["h"], tao=cfg["tao"])
                print(f"Φ={m['final_obj']:.4e} ({result['runtime_sec']:.1f}s)")

                rows.append({
                    "dim": dim, "strategy": strat_name, "repeat": rep,
                    "final_obj": m["final_obj"],
                    "sigma_mae": m["sigma_mae"],
                    "runtime_sec": result["runtime_sec"],
                })
                if rep == 0:
                    dispersion_data[dim][strat_name] = {
                        "coeff": result["coeff"],
                        "linear_sys": result["linear_sys"]}

    # Save results
    raw_path = RESULTS_DIR / "cnar_improved_raw.csv"
    write_csv(raw_path, rows)
    print(f"\nSaved raw results: {raw_path}")

    summary = aggregate_mean_std(
        rows, group_keys=["dim", "strategy"],
        value_keys=["final_obj", "sigma_mae", "runtime_sec"])
    summary_path = RESULTS_DIR / "cnar_improved_summary.csv"
    write_csv(summary_path, summary)
    print(f"Saved summary: {summary_path}")

    # Generate figures
    for dim in CONFIGS:
        dim_rows = [r for r in summary if int(r["dim"]) == dim]
        order = ["qa_baseline", "original_cnar",
                 "A_lower_thresh", "B_per_iter", "C_soft", "D_per_iter_ese"]
        dim_rows = sorted(dim_rows,
                          key=lambda r: order.index(r["strategy"])
                          if r["strategy"] in order else 99)

        wang_obj_val = compute_metrics(
            dim=dim, linear_sys=dispersion_data[dim]["wang_only"]["linear_sys"],
            coeff=dispersion_data[dim]["wang_only"]["coeff"],
            v=CONFIGS[dim]["v"], h=CONFIGS[dim]["h"],
            tao=CONFIGS[dim]["tao"])["final_obj"]

        plot_cnar_comparison(
            dim, dim_rows, wang_obj_val,
            FIGURES_DIR / f"cnar_improved_obj_dim{dim}.png")

        if dim in dispersion_data and len(dispersion_data[dim]) >= 2:
            plot_cnar_dispersion(
                dim, CONFIGS[dim], dispersion_data[dim],
                FIGURES_DIR / f"cnar_improved_dispersion_dim{dim}.png")

    # Print summary table
    print(f"\n{'='*85}")
    print(f"{'Dim':>3} | {'Strategy':<22} | {'Obj (mean±std)':>26} | "
          f"{'Runtime':>10}")
    print("-" * 85)
    for r in sorted(summary, key=lambda x: (x["dim"], x["final_obj_mean"])):
        print(f"{r['dim']:>3} | {r['strategy']:<22} | "
              f"{r['final_obj_mean']:.4e} ± {r['final_obj_std']:.4e} | "
              f"{r['runtime_sec_mean']:>8.1f}s")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run improved CNAR study")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["A", "B", "C", "D", "all"],
                        help="Which strategy to run (default: all)")
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--num-sample", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    strats = ["A", "B", "C", "D"] if args.strategy == "all" else [args.strategy]

    run_improved_cnar(
        strategies=strats,
        repeat_override=args.repeat,
        max_iter=args.max_iter,
        num_sample=args.num_sample,
        quick=args.quick,
    )
