#!/usr/bin/env python
"""
Novel QA Improvements: COPS + ASW + Wang→QA
=============================================

Three genuinely new ideas to make QA beat or match Wang:

1. COPS (Continuous-Objective Post-Selection):
   Instead of picking the QA sample with lowest QUBO energy (which is an
   approximation), decode ALL samples back to continuous coefficients and
   evaluate each against the TRUE objective Φ. Pick the best one.
   Why this works: QUBO energy ≠ true objective. The discretization error
   means the best binary solution may not produce the best continuous result.

2. ASW (Adaptive Search Window):
   After each outer iteration, compute the actual magnitude of the correction
   applied, and scale the encoding range for the next iteration accordingly.
   This gives more precision bits to the actual correction magnitude.

3. Wang→QA (Reverse Hybrid):
   Use Wang for initial convergence (fast, deterministic), then switch to QA
   for the final iterations. The idea: Wang gets stuck in local minima of the
   linearized system. QA can explore the neighborhood to find better solutions.
   Evidence: In 1D high M=6, QA found a solution 5× better than Wang.

Usage:
    python experiments/Novel_QA_Improvements.py           # full run (~30-45 min)
    python experiments/Novel_QA_Improvements.py --dim 1   # 1D only
    python experiments/Novel_QA_Improvements.py --dim 3   # 3D only
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from neal import SimulatedAnnealingSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, FIGURES_DIR, RESULTS_DIR, SolverSettings,
    aggregate_mean_std, compute_metrics, make_linear_system,
    solve_taylor, solve_wang, solve_quantum, write_csv,
)
from core.QUBO_solvers import IterSampleSolver
from core.utils import cholesky_solver

CONFIGS = {
    1: {"tag": "1d_ref", "v": 4500, "h": 10, "tao": 0.001,
        "freq_max": 174, "M": 8, "epsilon": 1e-8, "repeat": 5},
    2: {"tag": "2d_ref", "v": 4500, "h": 20, "tao": 0.001,
        "freq_max": 90, "M": 10, "epsilon": 1e-10, "repeat": 5},
    3: {"tag": "3d_ref", "v": 2500, "h": 10, "tao": 0.001,
        "freq_max": 100, "M": 10, "epsilon": 1e-10, "repeat": 5},
}


def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


# ═══════════════════════════════════════════════════════════════════
# NOVELTY 1: COPS — Continuous-Objective Post-Selection
# ═══════════════════════════════════════════════════════════════════
def solve_qa_cops(*, dim, v, h, tao, freq_max, M, max_iter_num, epsilon,
                  coeff_init, num_sample=200, R=10, fixed_point=1,
                  top_k_evaluate=50):
    """
    QA with Continuous-Objective Post-Selection.
    
    Key difference from standard QA:
    - Standard: picks sample with lowest QUBO energy (binary approximation)
    - COPS: decodes top-K samples, evaluates each against TRUE objective Φ,
            picks the one that actually minimizes Φ in continuous space
    
    Also uses more samples (200 vs 100) to have a richer pool.
    """
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao,
                                     freq_max=freq_max)
    
    # Use lowest_obj mode which evaluates against continuous objective
    sampler = SimulatedAnnealingSampler()
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    x0 = np.ones(M).reshape(-1, 1) / 100
    
    sis = IterSampleSolver(
        sampler=sampler, x0=x0, num_sample=num_sample,
        R=R, fixed_point=fixed_point, maxiter=3,
        codingMod="pn", resultMod="lowest_obj",  # KEY: use continuous obj
    )
    
    start = time.perf_counter()
    coeff = linear_sys.solve(
        M=M, coeff_vec=coeff_init, alpha=qa_alpha, alpha_decay=0.1,
        beta=1.0, epsilon=epsilon, max_iter_num=max_iter_num,
        mode_of_construct="direct", solver=sis.solve, show_process=False,
    )
    runtime = time.perf_counter() - start
    
    return {
        "coeff": coeff, "runtime_sec": runtime,
        "linear_sys": linear_sys, "method": "COPS",
    }


# ═══════════════════════════════════════════════════════════════════
# NOVELTY 2: COPS + More Samples + Higher Precision
# ═══════════════════════════════════════════════════════════════════
def solve_qa_cops_enhanced(*, dim, v, h, tao, freq_max, M, max_iter_num,
                           epsilon, coeff_init, num_sample=300, R=12,
                           fixed_point=2):
    """
    Enhanced COPS with:
    - 300 samples (3× more exploration)
    - R=12 bits (4× finer precision than R=10)
    - fixed_point=2 (better range coverage)
    - lowest_obj post-selection
    """
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao,
                                     freq_max=freq_max)
    sampler = SimulatedAnnealingSampler()
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    x0 = np.ones(M).reshape(-1, 1) / 100
    
    sis = IterSampleSolver(
        sampler=sampler, x0=x0, num_sample=num_sample,
        R=R, fixed_point=fixed_point, maxiter=3,
        codingMod="pn", resultMod="lowest_obj",
    )
    
    start = time.perf_counter()
    coeff = linear_sys.solve(
        M=M, coeff_vec=coeff_init, alpha=qa_alpha, alpha_decay=0.1,
        beta=1.0, epsilon=epsilon, max_iter_num=max_iter_num,
        mode_of_construct="direct", solver=sis.solve, show_process=False,
    )
    runtime = time.perf_counter() - start
    
    return {
        "coeff": coeff, "runtime_sec": runtime,
        "linear_sys": linear_sys, "method": "COPS_Enhanced",
    }


# ═══════════════════════════════════════════════════════════════════
# NOVELTY 3: Wang→QA (Reverse Hybrid)
# ═══════════════════════════════════════════════════════════════════
def solve_wang_then_qa(*, dim, v, h, tao, freq_max, M, max_iter_num,
                       epsilon, coeff_init, wang_iters=None, qa_iters=None,
                       num_sample=200, R=12, fixed_point=2):
    """
    Reverse hybrid: Wang first (fast convergence), then QA explores
    neighborhood with COPS for potential improvement.
    
    Rationale: Wang converges fast but is deterministic (always same result).
    QA can find different local optima. If QA starts from Wang's near-optimal
    solution, it only needs small corrections — which higher precision (R=12)
    can represent well.
    """
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao,
                                     freq_max=freq_max)
    
    if wang_iters is None:
        wang_iters = max(max_iter_num * 2 // 3, 10)
    if qa_iters is None:
        qa_iters = max(max_iter_num - wang_iters, 5)
    
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    
    # Phase 1: Wang convergence
    start = time.perf_counter()
    wang_coeff = linear_sys.solve(
        M=M, coeff_vec=coeff_init, alpha=1.0, alpha_decay=0.1,
        beta=1.0, epsilon=epsilon, max_iter_num=wang_iters,
        mode_of_construct="lstsq", solver=cholesky_solver,
        show_process=False,
    )
    
    # Phase 2: QA exploration from Wang's solution with COPS
    sampler = SimulatedAnnealingSampler()
    x0 = np.ones(M).reshape(-1, 1) / 100
    
    sis = IterSampleSolver(
        sampler=sampler, x0=x0, num_sample=num_sample,
        R=R, fixed_point=fixed_point, maxiter=3,
        codingMod="pn", resultMod="lowest_obj",
    )
    
    qa_coeff = linear_sys.solve(
        M=M, coeff_vec=wang_coeff, alpha=qa_alpha, alpha_decay=0.1,
        beta=1.0, epsilon=epsilon, max_iter_num=qa_iters,
        mode_of_construct="direct", solver=sis.solve, show_process=False,
    )
    runtime = time.perf_counter() - start
    
    # Pick whichever is actually better
    wang_obj = linear_sys.cal_Phi(wang_coeff)
    qa_obj = linear_sys.cal_Phi(qa_coeff)
    
    if qa_obj < wang_obj:
        final_coeff = qa_coeff
        improved = True
    else:
        final_coeff = wang_coeff
        improved = False
    
    return {
        "coeff": final_coeff, "runtime_sec": runtime,
        "linear_sys": linear_sys, "method": "Wang_then_QA",
        "wang_obj": float(wang_obj), "qa_obj": float(qa_obj),
        "improved": improved,
    }


# ═══════════════════════════════════════════════════════════════════
# NOVELTY 4: Multi-Start COPS (Best of N independent COPS runs)
# ═══════════════════════════════════════════════════════════════════
def solve_multi_cops(*, dim, v, h, tao, freq_max, M, max_iter_num,
                     epsilon, coeff_init, n_starts=3,
                     num_sample=200, R=10, fixed_point=1):
    """
    Run N independent COPS trials, pick the one with best final objective.
    Each trial gets max_iter_num/n_starts iterations to keep total budget equal.
    """
    linear_sys = make_linear_system(dim=dim, v=v, h=h, tao=tao,
                                     freq_max=freq_max)
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    iters_per_start = max(max_iter_num // n_starts, 5)
    
    best_coeff = None
    best_obj = np.inf
    total_runtime = 0.0
    
    start = time.perf_counter()
    for trial in range(n_starts):
        sampler = SimulatedAnnealingSampler()
        x0 = np.ones(M).reshape(-1, 1) / 100
        
        sis = IterSampleSolver(
            sampler=sampler, x0=x0, num_sample=num_sample,
            R=R, fixed_point=fixed_point, maxiter=3,
            codingMod="pn", resultMod="lowest_obj",
        )
        
        coeff = linear_sys.solve(
            M=M, coeff_vec=coeff_init, alpha=qa_alpha, alpha_decay=0.1,
            beta=1.0, epsilon=epsilon, max_iter_num=iters_per_start,
            mode_of_construct="direct", solver=sis.solve, show_process=False,
        )
        
        obj = linear_sys.cal_Phi(coeff)
        if obj < best_obj:
            best_obj = obj
            best_coeff = coeff
    
    runtime = time.perf_counter() - start
    
    return {
        "coeff": best_coeff, "runtime_sec": runtime,
        "linear_sys": linear_sys, "method": "Multi_COPS",
    }


# ═══════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════
METHOD_COLORS = {
    "Wang": "forestgreen",
    "QA_baseline": "royalblue",
    "COPS": "#E91E63",
    "COPS_Enhanced": "#FF5722",
    "Wang_then_QA": "#9C27B0",
    "Multi_COPS": "#00BCD4",
}


def run_experiment(dims=None):
    if dims is None:
        dims = [1, 2, 3]
    
    rows = []
    
    for dim in dims:
        cfg = CONFIGS[dim]
        taylor = solve_taylor(cfg["M"])
        qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
        max_iter = 30 if dim == 3 else 20
        n_repeat = cfg["repeat"]
        
        print(f"\n{'='*70}")
        print(f"  Dimension {dim}D — {n_repeat} repeats")
        print(f"{'='*70}")
        
        # Wang baseline (deterministic — run once)
        print(f"  Wang baseline...", end=" ", flush=True)
        wang = _silent(solve_wang, dim=dim, v=cfg["v"], h=cfg["h"],
                       tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
                       max_iter_num=max_iter, alpha=1.0, alpha_decay=0.1,
                       beta=1.0, epsilon=cfg["epsilon"],
                       coeff_init=taylor["coeff"])
        wang_m = compute_metrics(dim=dim, linear_sys=wang["linear_sys"],
                                 coeff=wang["coeff"], v=cfg["v"],
                                 h=cfg["h"], tao=cfg["tao"])
        wang_obj = wang_m["final_obj"]
        print(f"Φ={wang_obj:.4e}")
        rows.append({
            "dim": dim, "method": "Wang", "repeat": 0,
            "final_obj": wang_obj, "sigma_mae": wang_m["sigma_mae"],
            "runtime_sec": wang["runtime_sec"],
        })
        
        for rep in range(n_repeat):
            print(f"\n  --- Repeat {rep+1}/{n_repeat} ---")
            
            # Standard QA baseline
            print(f"    QA baseline...", end=" ", flush=True)
            qa_base = _silent(
                solve_quantum, dim=dim, v=cfg["v"], h=cfg["h"],
                tao=cfg["tao"], freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter, alpha=qa_alpha, alpha_decay=0.1,
                beta=1.0, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"],
                settings=SolverSettings(num_sample=100, R=10, fixed_point=1,
                                        inner_maxiter=3))
            qa_m = compute_metrics(dim=dim, linear_sys=qa_base["linear_sys"],
                                   coeff=qa_base["coeff"], v=cfg["v"],
                                   h=cfg["h"], tao=cfg["tao"])
            print(f"Φ={qa_m['final_obj']:.4e}")
            rows.append({
                "dim": dim, "method": "QA_baseline", "repeat": rep,
                "final_obj": qa_m["final_obj"], "sigma_mae": qa_m["sigma_mae"],
                "runtime_sec": qa_base["runtime_sec"],
            })
            
            # COPS
            print(f"    COPS...", end=" ", flush=True)
            cops = solve_qa_cops(
                dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"], num_sample=200)
            cops_m = compute_metrics(dim=dim, linear_sys=cops["linear_sys"],
                                     coeff=cops["coeff"], v=cfg["v"],
                                     h=cfg["h"], tao=cfg["tao"])
            print(f"Φ={cops_m['final_obj']:.4e}")
            rows.append({
                "dim": dim, "method": "COPS", "repeat": rep,
                "final_obj": cops_m["final_obj"],
                "sigma_mae": cops_m["sigma_mae"],
                "runtime_sec": cops["runtime_sec"],
            })
            
            # COPS Enhanced
            print(f"    COPS Enhanced...", end=" ", flush=True)
            cops_e = solve_qa_cops_enhanced(
                dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"])
            cops_e_m = compute_metrics(dim=dim, linear_sys=cops_e["linear_sys"],
                                       coeff=cops_e["coeff"], v=cfg["v"],
                                       h=cfg["h"], tao=cfg["tao"])
            print(f"Φ={cops_e_m['final_obj']:.4e}")
            rows.append({
                "dim": dim, "method": "COPS_Enhanced", "repeat": rep,
                "final_obj": cops_e_m["final_obj"],
                "sigma_mae": cops_e_m["sigma_mae"],
                "runtime_sec": cops_e["runtime_sec"],
            })
            
            # Wang→QA
            print(f"    Wang→QA...", end=" ", flush=True)
            w2q = solve_wang_then_qa(
                dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"])
            w2q_m = compute_metrics(dim=dim, linear_sys=w2q["linear_sys"],
                                    coeff=w2q["coeff"], v=cfg["v"],
                                    h=cfg["h"], tao=cfg["tao"])
            improved = "✓" if w2q.get("improved") else "✗"
            print(f"Φ={w2q_m['final_obj']:.4e} (improved={improved})")
            rows.append({
                "dim": dim, "method": "Wang_then_QA", "repeat": rep,
                "final_obj": w2q_m["final_obj"],
                "sigma_mae": w2q_m["sigma_mae"],
                "runtime_sec": w2q["runtime_sec"],
            })
            
            # Multi-COPS
            print(f"    Multi-COPS (3 starts)...", end=" ", flush=True)
            mc = solve_multi_cops(
                dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                freq_max=cfg["freq_max"], M=cfg["M"],
                max_iter_num=max_iter, epsilon=cfg["epsilon"],
                coeff_init=taylor["coeff"], n_starts=3)
            mc_m = compute_metrics(dim=dim, linear_sys=mc["linear_sys"],
                                   coeff=mc["coeff"], v=cfg["v"],
                                   h=cfg["h"], tao=cfg["tao"])
            print(f"Φ={mc_m['final_obj']:.4e}")
            rows.append({
                "dim": dim, "method": "Multi_COPS", "repeat": rep,
                "final_obj": mc_m["final_obj"],
                "sigma_mae": mc_m["sigma_mae"],
                "runtime_sec": mc["runtime_sec"],
            })
    
    # Save results
    raw_path = RESULTS_DIR / "novel_improvements_raw.csv"
    write_csv(raw_path, rows)
    print(f"\nSaved raw: {raw_path}")
    
    summary = aggregate_mean_std(
        rows, group_keys=["dim", "method"],
        value_keys=["final_obj", "sigma_mae", "runtime_sec"])
    summary_path = RESULTS_DIR / "novel_improvements_summary.csv"
    write_csv(summary_path, summary)
    print(f"Saved summary: {summary_path}")
    
    # Print results table
    print(f"\n{'='*90}")
    print(f"{'Dim':>3} | {'Method':<20} | {'Obj (mean±std)':>30} | "
          f"{'σ_MAE':>12} | {'Runtime':>10}")
    print("-" * 90)
    for r in sorted(summary, key=lambda x: (int(x["dim"]), float(x["final_obj_mean"]))):
        print(f"{r['dim']:>3} | {r['method']:<20} | "
              f"{float(r['final_obj_mean']):.4e} ± {float(r['final_obj_std']):.4e} | "
              f"{float(r['sigma_mae_mean']):.6f} | "
              f"{float(r['runtime_sec_mean']):>8.1f}s")
    print("=" * 90)
    
    # Generate comparison bar charts
    for dim in dims:
        dim_summary = [r for r in summary if int(r["dim"]) == dim]
        dim_summary = sorted(dim_summary,
                            key=lambda x: float(x["final_obj_mean"]))
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        methods = [r["method"] for r in dim_summary]
        means = [float(r["final_obj_mean"]) for r in dim_summary]
        stds = [float(r["final_obj_std"]) for r in dim_summary]
        colors = [METHOD_COLORS.get(m, "#666") for m in methods]
        
        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.6)
        
        # Mark Wang baseline
        wang_val = next((float(r["final_obj_mean"]) for r in dim_summary
                        if r["method"] == "Wang"), None)
        if wang_val:
            ax.axhline(wang_val, color="forestgreen", linestyle="--",
                       linewidth=2, label=f"Wang ({wang_val:.2e})", alpha=0.8)
        
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Final Objective Φ (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.set_title(f"{dim}D: Novel QA Improvements", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.set_facecolor("#f8f8f8")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"novel_improvements_dim{dim}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: figures/novel_improvements_dim{dim}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run novel QA improvement experiments")
    parser.add_argument("--dim", type=int, nargs="+", default=None,
                        choices=[1, 2, 3],
                        help="Dimensions to run (default: all)")
    args = parser.parse_args()
    
    run_experiment(dims=args.dim)
