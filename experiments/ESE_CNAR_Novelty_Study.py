#!/usr/bin/env python
"""
Novelty Study: Ensemble Sample Exploitation (ESE) & Condition-Number-Aware
Regularisation (CNAR) for QA-based staggered-grid FD coefficient optimisation.

ESE:  Use top-K Boltzmann-weighted SA samples instead of the single lowest-energy
      sample. Tests K ∈ {1, 3, 5, 10} across 1D, 2D, 3D.

CNAR: At each outer iteration, compute cond(A^T A) and set regularisation
      proportionally. Tested across 1D, 2D, 3D.

Safe to run — estimated runtime: ~20-30 min total.
"""

from __future__ import annotations

import numpy as np


def _suffix_name(base: str, suffix: str) -> str:
    suffix = suffix.strip()
    if not suffix:
        return base
    return f"{base}_{suffix}"

from experiments_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SolverSettings,
    aggregate_mean_std,
    compute_metrics,
    plot_bar,
    plot_line,
    solve_quantum,
    solve_taylor,
    solve_wang,
    write_csv,
)

# ── Reference configs (same as Novelty_QA_to_Wang_Stabilization.py) ──────────
CONFIGS = {
    1: {
        "tag": "1d_ref", "v": 4500, "h": 10, "tao": 0.001,
        "freq_max": 174, "M": 8, "epsilon": 1e-8, "repeat": 3,
    },
    2: {
        "tag": "2d_ref", "v": 4500, "h": 20, "tao": 0.001,
        "freq_max": 90, "M": 10, "epsilon": 1e-10, "repeat": 3,
    },
    3: {
        "tag": "3d_ref", "v": 2500, "h": 10, "tao": 0.001,
        "freq_max": 100, "M": 10, "epsilon": 1e-10, "repeat": 3,
    },
}


def _eval(dim, cfg, run_result):
    return compute_metrics(
        dim=dim, linear_sys=run_result["linear_sys"],
        coeff=run_result["coeff"], v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
    )


# ═════════════════════════════════════════════════════════════════════════
# Novelty 1: ESE — Ensemble Sample Exploitation
# ═════════════════════════════════════════════════════════════════════════
def run_ese_study(
    *,
    repeat_override: int | None = None,
    max_iter_num: int = 20,
    num_sample: int = 100,
    inner_maxiter: int = 3,
    output_suffix: str = "",
) -> None:
    """Test ESE with K ∈ {1, 3, 5, 10} across all dimensions."""
    K_values = [1, 3, 5, 10]
    rows = []

    for dim, cfg in CONFIGS.items():
        taylor = solve_taylor(cfg["M"])
        n_repeat = cfg["repeat"] if repeat_override is None else repeat_override
        for K in K_values:
            for rep in range(n_repeat):
                print(f"[ESE] dim={dim}, K={K}, rep={rep+1}/{n_repeat}")
                settings = SolverSettings(
                    num_sample=num_sample, R=10, fixed_point=1,
                    inner_maxiter=inner_maxiter, coding_mod="pn",
                    result_mod="lowest_energy", ensemble_k=K,
                )
                qa = solve_quantum(
                    dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                    freq_max=cfg["freq_max"], M=cfg["M"],
                    max_iter_num=max_iter_num, alpha=1.0, alpha_decay=0.1,
                    beta=1.0, epsilon=cfg["epsilon"],
                    coeff_init=taylor["coeff"], settings=settings,
                )
                m = _eval(dim, cfg, qa)
                rows.append({
                    "dim": dim, "tag": cfg["tag"], "K": K,
                    "repeat": rep,
                    "runtime_sec": float(qa["runtime_sec"]),
                    "iterations": int(qa["iterations"]),
                    "final_obj": m["final_obj"],
                    "sigma_mae": m["sigma_mae"],
                    "sigma_max_abs": m["sigma_max_abs"],
                })

    # Save raw + summary
    write_csv(RESULTS_DIR / f"{_suffix_name('ese_raw', output_suffix)}.csv", rows)
    summary = aggregate_mean_std(
        rows, group_keys=["dim", "tag", "K"],
        value_keys=["runtime_sec", "iterations", "final_obj", "sigma_mae", "sigma_max_abs"],
    )
    write_csv(RESULTS_DIR / f"{_suffix_name('ese_summary', output_suffix)}.csv", summary)

    # Plot per dimension
    for dim in CONFIGS:
        dim_rows = sorted(
            [r for r in summary if int(r["dim"]) == dim],
            key=lambda r: int(r["K"]),
        )
        plot_bar(
            categories=[f"K={r['K']}" for r in dim_rows],
            values=[float(r["final_obj_mean"]) for r in dim_rows],
            title=f"Dim {dim}: ESE objective vs ensemble size K",
            y_label="Final objective (mean)",
            out_path=FIGURES_DIR / f"{_suffix_name(f'ese_obj_dim{dim}', output_suffix)}.png",
        )
        plot_bar(
            categories=[f"K={r['K']}" for r in dim_rows],
            values=[float(r["final_obj_std"]) for r in dim_rows],
            title=f"Dim {dim}: ESE variance vs ensemble size K",
            y_label="Objective std",
            out_path=FIGURES_DIR / f"{_suffix_name(f'ese_std_dim{dim}', output_suffix)}.png",
        )
    print(f"ESE study complete. Results in {RESULTS_DIR}")


# ═════════════════════════════════════════════════════════════════════════
# Novelty 2: CNAR — Condition-Number-Aware Regularisation
# ═════════════════════════════════════════════════════════════════════════
def run_cnar_study(
    *,
    repeat_override: int | None = None,
    max_iter_num: int = 20,
    num_sample: int = 100,
    inner_maxiter: int = 3,
    output_suffix: str = "",
    kappa_threshold: float = 6.0,
    c_factor: float = 0.1,
) -> None:
    """
    Test condition-number-aware regularisation.
    For each dim, run QA with:
      - no_reg (α=0)
      - fixed (α=1, decay=0.1)
      - CNAR: α_k = c × max(0, log10(κ_k) - threshold)
    """
    from experiments_common import make_linear_system

    strategies = {
        "no_reg":    {"alpha": 0.0, "alpha_decay": 1.0},
        "fixed":     {"alpha": 1.0, "alpha_decay": 0.1},
        # CNAR uses dynamic α — we'll implement it inline
    }

    rows = []

    for dim, cfg in CONFIGS.items():
        taylor = solve_taylor(cfg["M"])
        n_repeat = cfg["repeat"] if repeat_override is None else repeat_override

        # First, compute condition numbers for this dimension
        linear_sys = make_linear_system(
            dim=dim, v=cfg["v"], h=cfg["h"],
            tao=cfg["tao"], freq_max=cfg["freq_max"],
        )
        A, b = linear_sys.construct(cfg["M"], taylor["coeff"])
        kappa = np.linalg.cond(A.T @ A)
        print(f"[CNAR] dim={dim}: initial cond(A^T A) = {kappa:.2e}")

        # CNAR adaptive α based on condition number
        cnar_alpha = c_factor * max(0, np.log10(kappa) - kappa_threshold)
        print(f"[CNAR] dim={dim}: CNAR α = {cnar_alpha:.4f}")

        strategies_full = {
            **strategies,
            "cnar": {"alpha": cnar_alpha, "alpha_decay": 0.5},
        }

        for sname, s in strategies_full.items():
            for rep in range(n_repeat):
                print(f"[CNAR] dim={dim}, strategy={sname}, rep={rep+1}")
                settings = SolverSettings(
                    num_sample=num_sample, R=10, fixed_point=1,
                    inner_maxiter=inner_maxiter, coding_mod="pn",
                    result_mod="lowest_energy",
                )
                qa = solve_quantum(
                    dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                    freq_max=cfg["freq_max"], M=cfg["M"],
                    max_iter_num=max_iter_num, alpha=s["alpha"],
                    alpha_decay=s["alpha_decay"], beta=1.0,
                    epsilon=cfg["epsilon"],
                    coeff_init=taylor["coeff"], settings=settings,
                )
                m = _eval(dim, cfg, qa)
                rows.append({
                    "dim": dim, "tag": cfg["tag"],
                    "strategy": sname,
                    "initial_cond_number": float(kappa),
                    "cnar_alpha": float(cnar_alpha) if sname == "cnar" else 0.0,
                    "repeat": rep,
                    "runtime_sec": float(qa["runtime_sec"]),
                    "iterations": int(qa["iterations"]),
                    "final_obj": m["final_obj"],
                    "sigma_mae": m["sigma_mae"],
                    "sigma_max_abs": m["sigma_max_abs"],
                })

    write_csv(RESULTS_DIR / f"{_suffix_name('cnar_raw', output_suffix)}.csv", rows)
    summary = aggregate_mean_std(
        rows, group_keys=["dim", "tag", "strategy"],
        value_keys=["runtime_sec", "iterations", "final_obj", "sigma_mae", "sigma_max_abs"],
    )
    write_csv(RESULTS_DIR / f"{_suffix_name('cnar_summary', output_suffix)}.csv", summary)

    for dim in CONFIGS:
        dim_rows = [r for r in summary if int(r["dim"]) == dim]
        order = ["no_reg", "fixed", "cnar"]
        dim_rows = sorted(dim_rows, key=lambda r: order.index(r["strategy"]))
        plot_bar(
            categories=[r["strategy"] for r in dim_rows],
            values=[float(r["final_obj_mean"]) for r in dim_rows],
            title=f"Dim {dim}: CNAR regularisation comparison",
            y_label="Final objective (mean)",
            out_path=FIGURES_DIR / f"{_suffix_name(f'cnar_obj_dim{dim}', output_suffix)}.png",
        )
    print(f"CNAR study complete. Results in {RESULTS_DIR}")


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ESE and/or CNAR novelty studies")
    parser.add_argument("--ese", action="store_true", help="Run ESE study")
    parser.add_argument("--cnar", action="store_true", help="Run CNAR study")
    parser.add_argument("--all", action="store_true", help="Run all studies")
    parser.add_argument("--repeat", type=int, default=None, help="Override repeats per config")
    parser.add_argument("--max-iter", type=int, default=20, help="Outer iterations per run")
    parser.add_argument("--num-sample", type=int, default=100, help="Sampler num_reads")
    parser.add_argument("--inner-maxiter", type=int, default=3, help="Inner iterative solve steps")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output CSV/PNG names")
    parser.add_argument("--cnar-kappa-threshold", type=float, default=6.0, help="CNAR threshold on log10(cond)")
    parser.add_argument("--cnar-c-factor", type=float, default=0.1, help="CNAR scaling constant")
    parser.add_argument("--quick", action="store_true", help="Fast pass: repeat=1, max_iter=10, num_sample=50")
    args = parser.parse_args()

    if args.quick:
        if args.repeat is None:
            args.repeat = 1
        if args.max_iter == 20:
            args.max_iter = 10
        if args.num_sample == 100:
            args.num_sample = 50

    common_kwargs = {
        "repeat_override": args.repeat,
        "max_iter_num": args.max_iter,
        "num_sample": args.num_sample,
        "inner_maxiter": args.inner_maxiter,
        "output_suffix": args.output_suffix,
    }

    cnar_kwargs = {
        **common_kwargs,
        "kappa_threshold": args.cnar_kappa_threshold,
        "c_factor": args.cnar_c_factor,
    }

    if args.all or (not args.ese and not args.cnar):
        run_ese_study(**common_kwargs)
        run_cnar_study(**cnar_kwargs)
    else:
        if args.ese:
            run_ese_study(**common_kwargs)
        if args.cnar:
            run_cnar_study(**cnar_kwargs)
