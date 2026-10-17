#!/usr/bin/env python
"""
ESE & CNAR: Forward-Simulation Dispersion Figures + Full Rerun

Generates publication-quality directional dispersion plots
(σ(θ,φ) vs kh) comparing:
  - Wang-only (baseline)
  - QA K=1 (original single-sample)
  - QA ESE K=5 (ensemble average — our novelty)

Also plots CNAR vs fixed-reg vs no-reg dispersion curves.

These are the most convincing figures for the paper because they show
the physical effect (dispersion error) rather than just a scalar objective.

Usage:
    python experiments/ESE_CNAR_Forward_Simulation.py                     # full run
    python experiments/ESE_CNAR_Forward_Simulation.py --dims 1 3          # specific dims
    python experiments/ESE_CNAR_Forward_Simulation.py --quick              # fast test
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from experiments_common import (
    PI,
    SolverSettings,
    compute_metrics,
    solve_quantum,
    solve_taylor,
    solve_wang,
)

ROOT = Path(__file__).resolve().parent.parent  # repository root
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SIM_CONFIGS = {
    1: {"v": 4500, "h": 10, "tao": 0.001, "freq_max": 174, "M": 8, "epsilon": 1e-8},
    2: {"v": 4500, "h": 20, "tao": 0.001, "freq_max": 90, "M": 10, "epsilon": 1e-10},
    3: {"v": 2500, "h": 10, "tao": 0.001, "freq_max": 100, "M": 10, "epsilon": 1e-10},
}


def _silent_solve(func, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kwargs)


def _angle_text(x):
    if np.isclose(x, 0.0):
        return "0"
    if np.isclose(x, PI / 8):
        return "π/8"
    if np.isclose(x, PI / 4):
        return "π/4"
    if np.isclose(x, 3 * PI / 8):
        return "3π/8"
    return f"{x:.2f}"


# ═══════════════════════════════════════════════════════════════════════
# Run solvers to get coefficients for each method
# ═══════════════════════════════════════════════════════════════════════
def simulate_ese_coefficients(dim, cfg, max_iter_num=20, num_sample=100):
    """Run Wang, QA K=1, QA K=5 for ESE comparison."""
    taylor = solve_taylor(cfg["M"])["coeff"]
    # Use paper-correct params per dimension
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    max_iter_dim = 30 if dim == 3 else max_iter_num

    # Wang baseline
    wang = _silent_solve(
        solve_wang, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=1.0, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
    )

    # QA K=1 (original single-sample)
    qa_k1 = _silent_solve(
        solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=qa_alpha, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
        settings=SolverSettings(num_sample=num_sample, R=10, fixed_point=1,
                                inner_maxiter=3, ensemble_k=1),
    )

    # QA K=5 (ESE ensemble)
    qa_k5 = _silent_solve(
        solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=qa_alpha, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
        settings=SolverSettings(num_sample=num_sample, R=10, fixed_point=1,
                                inner_maxiter=3, ensemble_k=5),
    )

    return {
        "Wang": {"sys": wang["linear_sys"], "coeff": wang["coeff"]},
        "QA (K=1)": {"sys": qa_k1["linear_sys"], "coeff": qa_k1["coeff"]},
        "QA ESE (K=5)": {"sys": qa_k5["linear_sys"], "coeff": qa_k5["coeff"]},
    }


def simulate_cnar_coefficients(dim, cfg, max_iter_num=20, num_sample=100):
    """Run Wang, QA no_reg, QA fixed_reg, QA CNAR for comparison."""
    taylor = solve_taylor(cfg["M"])["coeff"]
    # Use paper-correct params per dimension
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    max_iter_dim = 30 if dim == 3 else max_iter_num
    from experiments_common import make_linear_system
    ls = make_linear_system(dim=dim, v=cfg["v"], h=cfg["h"],
                            tao=cfg["tao"], freq_max=cfg["freq_max"])
    A, b = ls.construct(cfg["M"], taylor)
    kappa = np.linalg.cond(A.T @ A)
    cnar_alpha = 0.1 * max(0, np.log10(kappa) - 6.0)

    wang = _silent_solve(
        solve_wang, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=1.0, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
    )

    qa_noreg = _silent_solve(
        solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=0.0, alpha_decay=1.0, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
        settings=SolverSettings(num_sample=num_sample, R=10, fixed_point=1,
                                inner_maxiter=3),
    )

    qa_fixed = _silent_solve(
        solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=qa_alpha, alpha_decay=0.1, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
        settings=SolverSettings(num_sample=num_sample, R=10, fixed_point=1,
                                inner_maxiter=3),
    )

    qa_cnar = _silent_solve(
        solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"], max_iter_num=max_iter_dim,
        alpha=cnar_alpha, alpha_decay=0.5, beta=1.0, epsilon=cfg["epsilon"],
        coeff_init=taylor,
        settings=SolverSettings(num_sample=num_sample, R=10, fixed_point=1,
                                inner_maxiter=3),
    )

    return {
        "Wang": {"sys": wang["linear_sys"], "coeff": wang["coeff"]},
        "QA (no reg)": {"sys": qa_noreg["linear_sys"], "coeff": qa_noreg["coeff"]},
        "QA (fixed α)": {"sys": qa_fixed["linear_sys"], "coeff": qa_fixed["coeff"]},
        "QA (CNAR)": {"sys": qa_cnar["linear_sys"], "coeff": qa_cnar["coeff"]},
    }


# ═══════════════════════════════════════════════════════════════════════
# Plotting (dispersion curves at multiple angles)
# ═══════════════════════════════════════════════════════════════════════
METHOD_COLORS = {
    "Wang": "forestgreen",
    "QA (K=1)": "royalblue",
    "QA ESE (K=5)": "crimson",
    "QA (no reg)": "royalblue",
    "QA (fixed α)": "darkorange",
    "QA (CNAR)": "crimson",
}


def plot_dispersion(dim, cfg, method_runs, out_name, title_prefix=""):
    """
    Generate the directional dispersion figure: one subplot per method,
    σ(θ,φ) vs kh at multiple angles.
    """
    kh = np.linspace(PI / 1000, PI, 1000)
    r = cfg["v"] * cfg["tao"] / cfg["h"]

    n_methods = len(method_runs)
    fig, axes = plt.subplots(1, n_methods, figsize=(6.0 * n_methods, 5.2),
                             dpi=220, sharex=True, sharey=True)
    if n_methods == 1:
        axes = [axes]

    style_set = ["-", "--", ":", "-."]
    angle_colors = ["maroon", "midnightblue", "purple", "teal"]
    theta_vals = [0.0, PI / 8, PI / 4, 3 * PI / 8]
    phi_vals = [0.0, PI / 8, PI / 4]

    for ax, (method_name, data) in zip(axes, method_runs.items()):
        linear_sys = data["sys"]
        coeff = data["coeff"]
        base_color = METHOD_COLORS.get(method_name, "black")

        if dim == 1:
            sigma = linear_sys.cal_sigma(r, kh, coeff)
            ax.plot(kh, sigma, color=base_color, linewidth=1.8, label=method_name)

        elif dim == 2:
            for theta, ls in zip(theta_vals, style_set):
                sigma = linear_sys.cal_sigma(r, kh, coeff, theta=theta)
                ax.plot(kh, sigma, color=base_color, linestyle=ls,
                        linewidth=1.4, label=f"θ={_angle_text(theta)}")

        else:  # 3D
            for theta, ls in zip(theta_vals, style_set):
                for phi, col in zip(phi_vals, angle_colors):
                    sigma = linear_sys.cal_sigma(r, kh, coeff, theta=theta, phi=phi)
                    ax.plot(kh, sigma, color=col, linestyle=ls,
                            linewidth=1.35,
                            label=f"θ={_angle_text(theta)}, φ={_angle_text(phi)}")

        # Print objective value in the title
        m = compute_metrics(dim=dim, linear_sys=linear_sys, coeff=coeff,
                            v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
        obj_str = f"{m['final_obj']:.2e}"

        ax.set_facecolor("#f5f5f5")
        ax.set_xlim([0.0, PI])
        ax.set_ylim([0.90, 1.05])
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_yticks([0.90, 0.95, 1.00, 1.05])
        ax.set_xlabel("kh", fontsize=12)
        ax.tick_params(top=True, right=True, which="major", direction="in",
                       length=4, width=1.0)
        ax.tick_params(top=True, right=True, which="minor", direction="in",
                       length=2, width=0.8)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.grid(visible=True, axis="both", which="major", linestyle="--",
                linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
        ax.set_title(f"{method_name}\nΦ = {obj_str}", fontsize=11)

        if dim < 3:
            ax.legend(fontsize=7, loc="lower left", frameon=True,
                      edgecolor="black", borderpad=0.4, labelspacing=0.25,
                      handlelength=2.8)
        else:
            ax.legend(fontsize=5.5, loc="lower left", frameon=True,
                      edgecolor="black", borderpad=0.3, labelspacing=0.2,
                      handlelength=2.4, ncol=2)

    axes[0].set_ylabel("$\\delta(\\theta, \\phi)$", fontsize=13)
    fig.suptitle(f"{title_prefix}{dim}D directional dispersion", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer iterations/samples for testing")
    parser.add_argument("--ese-only", action="store_true")
    parser.add_argument("--cnar-only", action="store_true")
    args = parser.parse_args()

    max_iter = 10 if args.quick else 20
    num_sample = 50 if args.quick else 100

    for dim in args.dims:
        cfg = SIM_CONFIGS[dim]

        if not args.cnar_only:
            print(f"\n{'='*60}")
            print(f"ESE Forward Simulation: {dim}D")
            print(f"{'='*60}")
            runs_ese = simulate_ese_coefficients(dim, cfg, max_iter, num_sample)
            plot_dispersion(dim, cfg, runs_ese,
                            f"ese_dispersion_dim{dim}.png",
                            title_prefix="ESE: ")

        if not args.ese_only:
            print(f"\n{'='*60}")
            print(f"CNAR Forward Simulation: {dim}D")
            print(f"{'='*60}")
            runs_cnar = simulate_cnar_coefficients(dim, cfg, max_iter, num_sample)
            plot_dispersion(dim, cfg, runs_cnar,
                            f"cnar_dispersion_dim{dim}.png",
                            title_prefix="CNAR: ")

    print(f"\n{'='*60}")
    print("All forward simulation figures generated!")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"{'='*60}")
