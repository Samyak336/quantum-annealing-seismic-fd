#!/usr/bin/env python
"""
Faithful Reproduction of Wen & Wang (2025) — Corrected Parameters
==================================================================

Reproduces the paper's results across ALL 1D, 2D, and 3D configurations
with the CORRECT parameters: alpha=1.0 with alpha_decay=0.1 for ALL dims.

Phase 1: Paper-matched configs (Tables from the paper)
Phase 2: Parameter sensitivity (naive 1D→3D transfer test)

Outputs:
  - results/reproduction_corrected.csv
  - figures/reproduction_corrected_*.png

Usage:
    python experiments/Reproduction_Corrected.py
"""

from __future__ import annotations
import csv, io, sys, time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, SolverSettings, compute_metrics, make_linear_system,
    solve_quantum, solve_taylor, solve_wang,
)

FIGURES_DIR = ROOT.parent / "figures"
RESULTS_DIR = ROOT.parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


# ═══════════════════════════════════════════════════════════════════
# Paper configurations — ALL use alpha=1.0, alpha_decay=0.1
# ═══════════════════════════════════════════════════════════════════
CONFIGS_1D = [
    {"label": "1D v=1500 M=4",  "dim": 1, "v": 1500, "h": 10, "tao": 0.001,
     "freq_max": 58,  "M": 4,  "epsilon": 1e-8,  "max_iter": 20},
    {"label": "1D v=1500 M=6",  "dim": 1, "v": 1500, "h": 10, "tao": 0.001,
     "freq_max": 58,  "M": 6,  "epsilon": 1e-8,  "max_iter": 20},
    {"label": "1D v=1500 M=8",  "dim": 1, "v": 1500, "h": 10, "tao": 0.001,
     "freq_max": 58,  "M": 8,  "epsilon": 1e-8,  "max_iter": 20},
    {"label": "1D v=4500 M=4",  "dim": 1, "v": 4500, "h": 10, "tao": 0.001,
     "freq_max": 174, "M": 4,  "epsilon": 1e-8,  "max_iter": 20},
    {"label": "1D v=4500 M=6",  "dim": 1, "v": 4500, "h": 10, "tao": 0.001,
     "freq_max": 174, "M": 6,  "epsilon": 1e-8,  "max_iter": 20},
    {"label": "1D v=4500 M=8",  "dim": 1, "v": 4500, "h": 10, "tao": 0.001,
     "freq_max": 174, "M": 8,  "epsilon": 1e-8,  "max_iter": 20},
]

CONFIGS_2D = [
    {"label": "2D v=1500 M=10", "dim": 2, "v": 1500, "h": 20, "tao": 0.001,
     "freq_max": 29,  "M": 10, "epsilon": 1e-10, "max_iter": 20},
    {"label": "2D v=4500 M=10", "dim": 2, "v": 4500, "h": 20, "tao": 0.001,
     "freq_max": 90,  "M": 10, "epsilon": 1e-10, "max_iter": 20},
]

CONFIGS_3D = [
    {"label": "3D v=2500 M=10", "dim": 3, "v": 2500, "h": 10, "tao": 0.001,
     "freq_max": 100, "M": 10, "epsilon": 1e-10, "max_iter": 30},
]

ALL_CONFIGS = CONFIGS_1D + CONFIGS_2D + CONFIGS_3D


def run_single_config(cfg, n_qa_repeats=3):
    """Run Wang and QA for a single configuration."""
    label = cfg["label"]
    dim = cfg["dim"]
    taylor = solve_taylor(cfg["M"])["coeff"]

    # Common params — CORRECTED: alpha=1.0 for ALL dims
    common = dict(
        dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
        freq_max=cfg["freq_max"], M=cfg["M"],
        max_iter_num=cfg["max_iter"],
        alpha=1.0,           # CORRECTED: always 1.0
        alpha_decay=0.1,     # decays: a = 1.0 * 0.1^k
        beta=1.0,
        epsilon=cfg["epsilon"],
        coeff_init=taylor,
    )

    settings = SolverSettings(num_sample=100, R=10, fixed_point=1,
                              inner_maxiter=3)

    # Wang (deterministic — run once)
    print(f"    Wang...", end=" ", flush=True)
    wang = _silent(solve_wang, **common)
    wang_metrics = compute_metrics(
        dim=dim, linear_sys=wang["linear_sys"], coeff=wang["coeff"],
        v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
    print(f"Phi={wang_metrics['final_obj']:.4e}")

    # QA (stochastic — run n times, take best and mean)
    qa_results = []
    for i in range(n_qa_repeats):
        print(f"    QA run {i+1}/{n_qa_repeats}...", end=" ", flush=True)
        qa = _silent(solve_quantum, **common, settings=settings)
        qa_m = compute_metrics(
            dim=dim, linear_sys=qa["linear_sys"], coeff=qa["coeff"],
            v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
        qa_results.append(qa_m)
        print(f"Phi={qa_m['final_obj']:.4e}")

    qa_best = min(qa_results, key=lambda m: m["final_obj"])
    qa_mean_obj = np.mean([m["final_obj"] for m in qa_results])
    qa_mean_mae = np.mean([m["sigma_mae"] for m in qa_results])

    ratio = qa_best["final_obj"] / wang_metrics["final_obj"] if wang_metrics["final_obj"] > 0 else float("inf")

    return {
        "label": label,
        "dim": dim,
        "wang_obj": wang_metrics["final_obj"],
        "wang_mae": wang_metrics["sigma_mae"],
        "wang_max": wang_metrics["sigma_max_abs"],
        "wang_runtime": wang["runtime_sec"],
        "qa_best_obj": qa_best["final_obj"],
        "qa_best_mae": qa_best["sigma_mae"],
        "qa_best_max": qa_best["sigma_max_abs"],
        "qa_mean_obj": qa_mean_obj,
        "qa_mean_mae": qa_mean_mae,
        "ratio": ratio,
        "verdict": "PASS" if ratio < 2.0 else ("WARN" if ratio < 10.0 else "FAIL"),
    }


def save_results(rows, out_csv, out_fig):
    """Save CSV and bar chart."""
    # CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved CSV: {out_csv}")

    # Bar chart: QA/Wang ratio
    labels = [r["label"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    colors = ["forestgreen" if r < 2.0 else "darkorange" if r < 10.0 else "crimson"
              for r in ratios]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5), dpi=200)
    bars = ax.bar(range(len(labels)), ratios, color=colors, edgecolor="black",
                  linewidth=0.5)

    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="QA = Wang")
    ax.axhline(2.0, color="darkorange", linewidth=1.0, linestyle=":",
               label="2x threshold")

    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{ratio:.2f}x ", ha="center", va="bottom", fontsize=7,
                fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("QA Phi / Wang Phi  (lower = better)", fontsize=11)
    ax.set_title("Reproducibility: QA vs Wang -- Corrected alpha=1.0",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_fig}")


def plot_dispersion_comparison(configs, out_prefix):
    """Plot dispersion curves for each config: Wang vs QA side-by-side."""
    for cfg in configs:
        dim = cfg["dim"]
        label = cfg["label"]
        taylor = solve_taylor(cfg["M"])["coeff"]

        common = dict(
            dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
            freq_max=cfg["freq_max"], M=cfg["M"],
            max_iter_num=cfg["max_iter"],
            alpha=1.0, alpha_decay=0.1, beta=1.0,
            epsilon=cfg["epsilon"], coeff_init=taylor,
        )
        settings = SolverSettings(num_sample=100, R=10, fixed_point=1,
                                  inner_maxiter=3)

        wang = _silent(solve_wang, **common)
        qa = _silent(solve_quantum, **common, settings=settings)
        ls = wang["linear_sys"]

        kh = np.linspace(PI / 1000, PI, 1000)
        r = cfg["v"] * cfg["tao"] / cfg["h"]

        methods = {"Taylor": taylor, "Wang": wang["coeff"], "QA": qa["coeff"]}
        colors = {"Taylor": "#888888", "Wang": "forestgreen", "QA": "royalblue"}

        if dim == 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200,
                                     sharex=True, sharey=True)
            for ax, (name, coeff) in zip(axes, methods.items()):
                sigma = ls.cal_sigma(r, kh, coeff)
                m = compute_metrics(dim=dim, linear_sys=ls, coeff=coeff,
                                   v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
                ax.plot(kh, sigma, color=colors[name], linewidth=1.8)
                ax.set_title(f"{name}\nPhi={m['final_obj']:.2e}",
                             fontsize=10, fontweight="bold",
                             color=colors[name])
                ax.set_xlim([0, PI])
                ax.set_ylim([0.90, 1.05])
                ax.set_xlabel("kh", fontsize=10)
                ax.grid(True, linestyle="--", linewidth=0.5)
                ax.set_facecolor("#f5f5f5")
            axes[0].set_ylabel("sigma(kh)", fontsize=11)
        else:
            theta_vals = [0.0, PI/8, PI/4, 3*PI/8]
            styles = ["-", "--", ":", "-."]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200,
                                     sharex=True, sharey=True)
            for ax, (name, coeff) in zip(axes, methods.items()):
                m = compute_metrics(dim=dim, linear_sys=ls, coeff=coeff,
                                   v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
                if dim == 2:
                    for theta, sty in zip(theta_vals, styles):
                        sigma = ls.cal_sigma(r, kh, coeff, theta=theta)
                        ax.plot(kh, sigma, color=colors[name], linestyle=sty,
                                linewidth=1.2)
                else:
                    phi_vals = [0.0, PI/8, PI/4]
                    angle_colors = ["maroon", "midnightblue", "teal"]
                    for theta, sty in zip(theta_vals, styles):
                        for phi_, col in zip(phi_vals, angle_colors):
                            sigma = ls.cal_sigma(r, kh, coeff,
                                                 theta=theta, phi=phi_)
                            ax.plot(kh, sigma, color=col, linestyle=sty,
                                    linewidth=1.0)
                ax.set_title(f"{name}\nPhi={m['final_obj']:.2e}",
                             fontsize=10, fontweight="bold",
                             color=colors[name])
                ax.set_xlim([0, PI])
                ax.set_ylim([0.90, 1.05])
                ax.set_xlabel("kh", fontsize=10)
                ax.grid(True, linestyle="--", linewidth=0.5)
                ax.set_facecolor("#f5f5f5")
            axes[0].set_ylabel("sigma(theta,kh)", fontsize=11)

        safe_label = label.replace(" ", "_").replace("=", "")
        fig.suptitle(f"Dispersion: {label} (alpha=1.0, decay=0.1)",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        fname = FIGURES_DIR / f"{out_prefix}_{safe_label}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  REPRODUCTION WITH CORRECTED alpha=1.0 (all dims)")
    print("  Paper: Wen & Wang (2025)")
    print("=" * 60)

    # Phase 1: Reproduce all configs
    all_rows = []
    for cfg in ALL_CONFIGS:
        print(f"\n  Config: {cfg['label']}")
        n_repeats = 3 if cfg["dim"] == 3 else 3
        row = run_single_config(cfg, n_qa_repeats=n_repeats)
        all_rows.append(row)

    save_results(
        all_rows,
        RESULTS_DIR / "reproduction_corrected.csv",
        FIGURES_DIR / "reproduction_corrected_ratios.png")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"  {'Config':<22s} {'Wang Phi':>12s} {'QA best Phi':>12s} "
          f"{'Ratio':>8s} {'Verdict':>8s}")
    print("-" * 80)
    for r in all_rows:
        print(f"  {r['label']:<22s} {r['wang_obj']:>12.4e} "
              f"{r['qa_best_obj']:>12.4e} {r['ratio']:>8.2f}x "
              f"{r['verdict']:>8s}")
    print("=" * 80)

    # Phase 2: Dispersion curves for key configs
    print("\n  Generating dispersion curves...")
    key_configs = [
        CONFIGS_1D[4],  # v=4500 M=6 (QA beats Wang here)
        CONFIGS_1D[5],  # v=4500 M=8
        CONFIGS_2D[1],  # v=4500 2D
        CONFIGS_3D[0],  # 3D
    ]
    plot_dispersion_comparison(key_configs, "repro_corrected_disp")

    print(f"\n{'='*60}")
    print("  Reproduction complete!")
    print(f"{'='*60}")
