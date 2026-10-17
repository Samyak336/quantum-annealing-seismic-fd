#!/usr/bin/env python
"""
Novelty Ablation Experiments: Multi-Restart, Momentum L-Update, Greedy Revert
==============================================================================
Tests three ideas — individually and combined — across 1D, 2D, 3D.

Usage:
    python experiments/Novelty_Ablation_Experiments.py

Outputs:
    results/novelty_ablation_raw.csv        — per-trial results
    results/novelty_ablation_summary.csv    — aggregated mean±std
    figures/novelty_ablation_obj_dim*.png    — bar chart per dimension
    figures/novelty_ablation_runtime_dim*.png
    figures/novelty_dispersion_dim*.png      — forward simulation dispersion
"""

from __future__ import annotations

import copy
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_common import (
    SolverSettings,
    make_linear_system,
    solve_quantum,
    solve_wang,
    compute_metrics,
    write_csv,
    aggregate_mean_std,
    DimType,
    RESULTS_DIR,
    FIGURES_DIR,
)
from core import stagger_FD_coeff as sfci

PI = np.pi

# ─── Configuration ────────────────────────────────────────────────────────────

N_TRIALS = 5  # statistical trials per variant

# Physical parameters per dimension
DIM_CONFIGS = {
    1: dict(v=4500, h=10, tao=0.001, freq_max=174, M=8,
            max_iter_num=20, alpha=1e-3, alpha_decay=0.1, beta=1.0, epsilon=1e-10),
    2: dict(v=4500, h=20, tao=0.001, freq_max=90, M=10,
            max_iter_num=20, alpha=1e-3, alpha_decay=0.1, beta=1.0, epsilon=1e-10),
    3: dict(v=2500, h=10, tao=0.001, freq_max=100, M=10,
            max_iter_num=20, alpha=1e-3, alpha_decay=0.1, beta=1.0, epsilon=1e-10),
}

# Best known encodings per dimension (from our ablation study)
BEST_ENCODINGS = {
    1: dict(R=12, fixed_point=2),
    2: dict(R=10, fixed_point=1),
    3: dict(R=12, fixed_point=2),
}


@dataclass
class ExperimentVariant:
    name: str
    momentum_beta: float = 0.0
    greedy_revert: bool = False
    n_restarts: int = 1        # 1 = no multi-restart
    iters_per_restart: int = 20  # total outer iters per restart


# ─── Experiment Variants ──────────────────────────────────────────────────────
VARIANTS = [
    # Baseline
    ExperimentVariant(name="baseline"),

    # Individual ideas
    ExperimentVariant(name="momentum_0.5", momentum_beta=0.5),
    ExperimentVariant(name="momentum_0.7", momentum_beta=0.7),
    ExperimentVariant(name="momentum_0.9", momentum_beta=0.9),
    ExperimentVariant(name="greedy_revert", greedy_revert=True),

    # Multi-restart (total effort = 20 iters)
    ExperimentVariant(name="restart_2x10", n_restarts=2, iters_per_restart=10),
    ExperimentVariant(name="restart_4x5", n_restarts=4, iters_per_restart=5),
    ExperimentVariant(name="restart_5x4", n_restarts=5, iters_per_restart=4),

    # Combined: best of each
    ExperimentVariant(name="momentum_0.7+greedy", momentum_beta=0.7, greedy_revert=True),
    ExperimentVariant(name="restart_4x5+greedy", n_restarts=4, iters_per_restart=5, greedy_revert=True),
    ExperimentVariant(name="restart_4x5+momentum_0.7", n_restarts=4, iters_per_restart=5, momentum_beta=0.7),
    ExperimentVariant(name="full_combo", n_restarts=4, iters_per_restart=5,
                      momentum_beta=0.7, greedy_revert=True),
]


# ─── Helper: Multi-restart wrapper ───────────────────────────────────────────

def run_single_qa(dim: int, cfg: dict, settings: SolverSettings) -> dict:
    """Run a single QA solve and return results dict."""
    coeff_init = sfci.Taylor_Coef(cfg["M"])
    return solve_quantum(
        dim=dim,
        v=cfg["v"], h=cfg["h"], tao=cfg["tao"], freq_max=cfg["freq_max"],
        M=cfg["M"],
        max_iter_num=cfg["max_iter_num"],
        alpha=cfg["alpha"], alpha_decay=cfg["alpha_decay"],
        beta=cfg["beta"], epsilon=cfg["epsilon"],
        coeff_init=coeff_init,
        settings=settings,
    )


def run_multi_restart_qa(dim: int, cfg: dict, variant: ExperimentVariant) -> dict:
    """Run QA with optional multi-restart. Returns best result."""
    enc = BEST_ENCODINGS[dim]
    settings = SolverSettings(
        num_sample=100,
        R=enc["R"],
        fixed_point=enc["fixed_point"],
        inner_maxiter=3,
        momentum_beta=variant.momentum_beta,
        greedy_revert=variant.greedy_revert,
    )

    if variant.n_restarts <= 1:
        # Standard single run
        cfg_copy = dict(cfg)
        cfg_copy["max_iter_num"] = variant.iters_per_restart
        return run_single_qa(dim, cfg_copy, settings)
    else:
        # Multi-restart: run N times with fewer iters, pick the best
        best_result = None
        best_obj = np.inf
        total_runtime = 0.0

        for r in range(variant.n_restarts):
            cfg_copy = dict(cfg)
            cfg_copy["max_iter_num"] = variant.iters_per_restart
            result = run_single_qa(dim, cfg_copy, settings)

            total_runtime += result["runtime_sec"]

            # Compute objective for this restart
            metrics = compute_metrics(
                dim=dim,
                linear_sys=result["linear_sys"],
                coeff=result["coeff"],
                v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
            )
            obj = metrics["final_obj"]

            if obj < best_obj:
                best_obj = obj
                best_result = result

        # Override runtime to be total across restarts
        best_result["runtime_sec"] = total_runtime
        return best_result


# ─── Wang baseline (run once per dim) ────────────────────────────────────────

def run_wang_baseline(dim: int, cfg: dict) -> dict:
    coeff_init = sfci.Taylor_Coef(cfg["M"])
    return solve_wang(
        dim=dim,
        v=cfg["v"], h=cfg["h"], tao=cfg["tao"], freq_max=cfg["freq_max"],
        M=cfg["M"],
        max_iter_num=cfg["max_iter_num"],
        alpha=cfg["alpha"], alpha_decay=cfg["alpha_decay"],
        beta=cfg["beta"], epsilon=cfg["epsilon"],
        coeff_init=coeff_init,
    )


# ─── Forward simulation (directional dispersion) ─────────────────────────────

def direction_grid(dim):
    if dim == 1:
        return [(None, None)]
    elif dim == 2:
        return [(0.0, None), (PI/8, None), (PI/4, None), (3*PI/8, None)]
    else:
        return [
            (0.0, 0.0), (PI/8, 0.0), (PI/8, PI/8),
            (PI/4, PI/8), (PI/4, PI/4),
        ]


def compute_dispersion_curves(dim, linear_sys, coeff, v, h, tao, n_points=1000):
    kh = np.linspace(PI/1000, PI, n_points)
    r = v * tao / h
    curves = []
    for theta, phi in direction_grid(dim):
        if dim == 1:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff)).reshape(-1)
        elif dim == 2:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff, theta=theta)).reshape(-1)
        else:
            sigma = np.asarray(linear_sys.cal_sigma(r, kh, coeff, theta=theta, phi=phi)).reshape(-1)
        curves.append(sigma)
    return kh, np.array(curves)


def plot_dispersion_comparison(dim, kh, labeled_curve_sets, out_path):
    """Plot directional dispersion for multiple methods."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#795548']
    for i, (label, curves) in enumerate(labeled_curve_sets):
        color = colors[i % len(colors)]
        for j, sigma in enumerate(curves):
            lbl = label if j == 0 else None
            ax.plot(kh, sigma, color=color, alpha=0.7, linewidth=1.2, label=lbl)

    ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.8)
    ax.set_xlim(0, PI)
    ax.set_ylim(0.9, 1.05)
    ax.set_xlabel('kh', fontsize=12)
    ax.set_ylabel('σ (normalized phase velocity)', fontsize=12)
    ax.set_title(f'{dim}D Directional Dispersion: Novelty Methods', fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Grouped bar chart ───────────────────────────────────────────────────────

def plot_grouped_bars(dim, summary_rows, wang_obj, metric_key, ylabel, title_suffix, out_path):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    names = [r["variant"] for r in summary_rows]
    means = [r[f"{metric_key}_mean"] for r in summary_rows]
    stds = [r[f"{metric_key}_std"] for r in summary_rows]

    colors = []
    for name in names:
        if name == "baseline":
            colors.append('#78909C')
        elif "restart" in name and "momentum" not in name and "greedy" not in name:
            colors.append('#2196F3')
        elif "momentum" in name and "restart" not in name and "greedy" not in name:
            colors.append('#4CAF50')
        elif name == "greedy_revert":
            colors.append('#FF9800')
        else:
            colors.append('#9C27B0')  # combined

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor='white', linewidth=0.5)

    # Wang reference line
    ax.axhline(wang_obj, color='red', linestyle='--', linewidth=1.5,
               label=f'Wang ({wang_obj:.2e})', alpha=0.9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{dim}D {title_suffix}', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NOVELTY ABLATION EXPERIMENTS")
    print(f"Variants: {len(VARIANTS)}, Trials: {N_TRIALS}, Dims: 1D, 2D, 3D")
    print(f"Total runs: {len(VARIANTS) * N_TRIALS * 3}")
    print("=" * 70)

    all_raw_rows = []
    wang_cache = {}  # dim -> (obj, coeff, linear_sys)

    for dim in [1, 2, 3]:
        cfg = DIM_CONFIGS[dim]
        print(f"\n{'='*50}")
        print(f"  DIMENSION {dim}D")
        print(f"{'='*50}")

        # Wang baseline (once)
        if dim not in wang_cache:
            print(f"  Computing Wang baseline...")
            wang_result = run_wang_baseline(dim, cfg)
            wang_metrics = compute_metrics(
                dim=dim, linear_sys=wang_result["linear_sys"],
                coeff=wang_result["coeff"],
                v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
            )
            wang_cache[dim] = (wang_metrics["final_obj"], wang_result["coeff"],
                               wang_result["linear_sys"], wang_result["runtime_sec"])
            print(f"  Wang obj: {wang_metrics['final_obj']:.4e} "
                  f"(runtime: {wang_result['runtime_sec']:.2f}s)")

        wang_obj, wang_coeff, wang_linsys, wang_runtime = wang_cache[dim]

        # Run all variants
        for variant in VARIANTS:
            for trial in range(N_TRIALS):
                print(f"  [{dim}D] {variant.name} trial {trial+1}/{N_TRIALS} ...", end=" ", flush=True)
                t0 = time.time()

                try:
                    result = run_multi_restart_qa(dim, cfg, variant)

                    # Metrics
                    metrics = compute_metrics(
                        dim=dim, linear_sys=result["linear_sys"],
                        coeff=result["coeff"],
                        v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                    )

                    row = {
                        "dim": dim,
                        "variant": variant.name,
                        "trial": trial + 1,
                        "final_obj": metrics["final_obj"],
                        "sigma_mae": metrics["sigma_mae"],
                        "runtime_sec": result["runtime_sec"],
                        "wang_obj": wang_obj,
                        "improvement_pct": (1 - metrics["final_obj"] / wang_obj) * 100
                            if wang_obj > 0 else 0,
                    }
                    all_raw_rows.append(row)
                    elapsed = time.time() - t0
                    print(f"obj={metrics['final_obj']:.4e} ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_raw_rows.append({
                        "dim": dim, "variant": variant.name, "trial": trial + 1,
                        "final_obj": float('inf'), "sigma_mae": float('inf'),
                        "runtime_sec": 0.0, "wang_obj": wang_obj, "improvement_pct": -999,
                    })

    # ─── Save raw results ─────────────────────────────────────────────────
    raw_csv = RESULTS_DIR / "novelty_ablation_raw.csv"
    write_csv(raw_csv, all_raw_rows)
    print(f"\nSaved raw results: {raw_csv}")

    # ─── Aggregate summary ────────────────────────────────────────────────
    summary_rows = aggregate_mean_std(
        all_raw_rows,
        group_keys=["dim", "variant"],
        value_keys=["final_obj", "sigma_mae", "runtime_sec"],
    )
    summary_csv = RESULTS_DIR / "novelty_ablation_summary.csv"
    write_csv(summary_csv, summary_rows)
    print(f"Saved summary: {summary_csv}")

    # ─── Generate figures ─────────────────────────────────────────────────
    for dim in [1, 2, 3]:
        dim_rows = [r for r in summary_rows if r["dim"] == dim]
        wang_obj = wang_cache[dim][0]

        # Objective bar chart
        plot_grouped_bars(
            dim, dim_rows, wang_obj,
            metric_key="final_obj",
            ylabel="Dispersion Objective Φ (log scale)",
            title_suffix="Novelty Ablation — Objective",
            out_path=FIGURES_DIR / f"novelty_ablation_obj_dim{dim}.png",
        )

        # Runtime bar chart
        plot_grouped_bars(
            dim, dim_rows, wang_cache[dim][3],
            metric_key="runtime_sec",
            ylabel="Runtime (seconds, log scale)",
            title_suffix="Novelty Ablation — Runtime",
            out_path=FIGURES_DIR / f"novelty_ablation_runtime_dim{dim}.png",
        )

    # ─── Forward simulation dispersion for best variant per dim ───────────
    print("\n--- Generating dispersion figures ---")
    for dim in [1, 2, 3]:
        cfg = DIM_CONFIGS[dim]
        wang_obj, wang_coeff, wang_linsys, _ = wang_cache[dim]

        # Find the best variant for this dim
        dim_summary = [r for r in summary_rows if r["dim"] == dim]
        dim_summary.sort(key=lambda r: r["final_obj_mean"])
        best_variant_name = dim_summary[0]["variant"]
        best_variant = [v for v in VARIANTS if v.name == best_variant_name][0]

        print(f"  [{dim}D] Best variant: {best_variant_name} "
              f"(obj={dim_summary[0]['final_obj_mean']:.4e})")

        # Also get baseline
        baseline_variant = VARIANTS[0]

        # Run once for dispersion curves
        baseline_result = run_multi_restart_qa(dim, cfg, baseline_variant)
        best_result = run_multi_restart_qa(dim, cfg, best_variant)

        # Compute dispersion curves
        kh, wang_curves = compute_dispersion_curves(
            dim, wang_linsys, wang_coeff, cfg["v"], cfg["h"], cfg["tao"])
        _, baseline_curves = compute_dispersion_curves(
            dim, baseline_result["linear_sys"], baseline_result["coeff"],
            cfg["v"], cfg["h"], cfg["tao"])
        _, best_curves = compute_dispersion_curves(
            dim, best_result["linear_sys"], best_result["coeff"],
            cfg["v"], cfg["h"], cfg["tao"])

        labeled = [
            (f"Wang (Φ={wang_obj:.2e})", wang_curves),
            (f"QA baseline (Φ={compute_metrics(dim=dim, linear_sys=baseline_result['linear_sys'], coeff=baseline_result['coeff'], v=cfg['v'], h=cfg['h'], tao=cfg['tao'])['final_obj']:.2e})", baseline_curves),
        ]
        if best_variant_name != "baseline":
            best_obj_val = compute_metrics(
                dim=dim, linear_sys=best_result["linear_sys"],
                coeff=best_result["coeff"],
                v=cfg["v"], h=cfg["h"], tao=cfg["tao"])["final_obj"]
            labeled.append(
                (f"{best_variant_name} (Φ={best_obj_val:.2e})", best_curves))

        plot_dispersion_comparison(
            dim, kh, labeled,
            out_path=FIGURES_DIR / f"novelty_dispersion_dim{dim}.png")

    # ─── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Dim':>3} | {'Variant':<28} | {'Obj (mean±std)':>24} | {'Wang Obj':>12} | {'vs Wang':>8}")
    print("-" * 90)
    for r in sorted(summary_rows, key=lambda x: (x["dim"], x["final_obj_mean"])):
        wang = wang_cache[r["dim"]][0]
        ratio = r["final_obj_mean"] / wang if wang > 0 else float('inf')
        marker = " ✓" if ratio < 1.5 else " ✗" if ratio > 10 else ""
        print(f"{r['dim']:>3} | {r['variant']:<28} | "
              f"{r['final_obj_mean']:.4e} ± {r['final_obj_std']:.4e} | "
              f"{wang:.4e} | {ratio:>6.1f}×{marker}")
    print("=" * 90)


if __name__ == "__main__":
    main()
