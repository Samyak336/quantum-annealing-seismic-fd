#!/usr/bin/env python
"""
DOPP/QDP Novelty: Forward Simulation + Waveform Figures
========================================================

Generates publication-quality dispersion curves and waveform figures
for our novel methods (DOPP, QDP, Full Pipeline) vs Wang and QA baseline.

Produces:
  1. Directional dispersion curves (σ vs kh) for each method
  2. 2D wavefield snapshots comparing all methods
  3. 1D seismogram overlays
  4. Wavefield error maps vs Wang reference

Usage:
    python experiments/DOPP_Forward_Waveform.py            # full run
    python experiments/DOPP_Forward_Waveform.py --dim 1    # 1D seismograms only
    python experiments/DOPP_Forward_Waveform.py --dim 2    # 2D snapshots only
"""

from __future__ import annotations
import argparse, io, sys, time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, SolverSettings, compute_metrics, make_linear_system,
    solve_quantum, solve_taylor, solve_wang, _direction_grid,
)

FIGURES_DIR = ROOT.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SIM_CONFIGS = {
    1: {"v": 4500, "h": 10, "tao": 0.001, "freq_max": 174, "M": 8,
        "epsilon": 1e-8},
    2: {"v": 4500, "h": 20, "tao": 0.001, "freq_max": 90, "M": 10,
        "epsilon": 1e-10},
    3: {"v": 2500, "h": 10, "tao": 0.001, "freq_max": 100, "M": 10,
        "epsilon": 1e-10},
}

METHOD_COLORS = {
    "Taylor": "#888888",
    "Wang": "forestgreen",
    "QA baseline": "royalblue",
    "DOPP(Wang)": "#E91E63",
    "QDP": "#FF9800",
    "Full Pipeline": "#9C27B0",
}

def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


def _sigma_mae_obj(coeff_flat, linear_sys, dim, cfg):
    coeff = coeff_flat.reshape(-1, 1)
    kh = np.linspace(PI / 1000, PI, 500)
    r = cfg["v"] * cfg["tao"] / cfg["h"]
    errs = []
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
        errs.append(np.abs(sigma - 1.0))
    all_err = np.concatenate(errs)
    if np.any(np.isnan(all_err)) or np.any(np.isinf(all_err)):
        return 1e10
    return float(np.mean(all_err))


def run_dopp(coeff_start, linear_sys, dim, cfg):
    result = scipy_minimize(
        _sigma_mae_obj, x0=coeff_start.flatten(),
        args=(linear_sys, dim, cfg), method="Nelder-Mead",
        options={"maxiter": 500, "maxfev": 2000, "xatol": 1e-10,
                 "fatol": 1e-10, "adaptive": True})
    return result.x.reshape(-1, 1)


# ═══════════════════════════════════════════════════════════════════
# Get coefficients for all methods
# ═══════════════════════════════════════════════════════════════════
def get_novelty_coefficients(dim, cfg):
    taylor = solve_taylor(cfg["M"])["coeff"]
    qa_alpha = 1.0  # Paper uses alpha=1.0 for ALL dims, alpha_decay=0.1
    max_iter = 30 if dim == 3 else 20
    pool_size = 6 if dim == 3 else 10

    common_wang = dict(dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                       freq_max=cfg["freq_max"], M=cfg["M"],
                       max_iter_num=max_iter, alpha=1.0, alpha_decay=0.1,
                       beta=1.0, epsilon=cfg["epsilon"], coeff_init=taylor)

    settings = SolverSettings(num_sample=100, R=10, fixed_point=1,
                              inner_maxiter=3)

    methods = {}

    print("  [1/6] Taylor...")
    methods["Taylor"] = taylor

    print("  [2/6] Wang...")
    wang = _silent(solve_wang, **common_wang)
    methods["Wang"] = wang["coeff"]
    wang_sys = wang["linear_sys"]

    print("  [3/6] QA baseline...")
    qa = _silent(solve_quantum, **{**common_wang, "alpha": qa_alpha},
                 settings=settings)
    methods["QA baseline"] = qa["coeff"]

    print("  [4/6] DOPP(Wang) — Nelder-Mead on σ_MAE...")
    dopp_coeff = run_dopp(wang["coeff"], wang_sys, dim, cfg)
    methods["DOPP(Wang)"] = dopp_coeff

    print(f"  [5/6] QDP — diversity pool (N={pool_size})...")
    best_mae = np.inf
    best_qa_coeff = None
    best_qa_sys = None
    for i in range(pool_size):
        q = _silent(solve_quantum, **{**common_wang, "alpha": qa_alpha},
                    settings=settings)
        m = compute_metrics(dim=dim, linear_sys=q["linear_sys"],
                           coeff=q["coeff"], v=cfg["v"], h=cfg["h"],
                           tao=cfg["tao"])
        if m["sigma_mae"] < best_mae:
            best_mae = m["sigma_mae"]
            best_qa_coeff = q["coeff"]
            best_qa_sys = q["linear_sys"]
    methods["QDP"] = best_qa_coeff

    print("  [6/6] Full Pipeline — QDP + DOPP...")
    full_coeff = run_dopp(best_qa_coeff, best_qa_sys, dim, cfg)
    methods["Full Pipeline"] = full_coeff

    # Print metrics
    print("\n  Metrics:")
    for name, coeff in methods.items():
        m = compute_metrics(dim=dim, linear_sys=wang_sys, coeff=coeff,
                           v=cfg["v"], h=cfg["h"], tao=cfg["tao"])
        print(f"    {name:20s}: Φ={m['final_obj']:.4e}  "
              f"σ_MAE={m['sigma_mae']:.6f}  σ_MAX={m['sigma_max_abs']:.6f}")

    return methods, wang_sys


# ═══════════════════════════════════════════════════════════════════
# Dispersion curve plotting
# ═══════════════════════════════════════════════════════════════════
def _angle_text(x):
    if np.isclose(x, 0.0): return "0"
    if np.isclose(x, PI / 8): return "π/8"
    if np.isclose(x, PI / 4): return "π/4"
    if np.isclose(x, 3 * PI / 8): return "3π/8"
    return f"{x:.2f}"


def plot_dispersion_novelty(dim, cfg, methods, linear_sys, out_name):
    kh = np.linspace(PI / 1000, PI, 1000)
    r = cfg["v"] * cfg["tao"] / cfg["h"]

    # Skip Taylor for cleaner comparison
    plot_methods = {k: v for k, v in methods.items() if k != "Taylor"}
    n = len(plot_methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), dpi=220,
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    theta_vals = [0.0, PI / 8, PI / 4, 3 * PI / 8]
    phi_vals = [0.0, PI / 8, PI / 4]
    styles = ["-", "--", ":", "-."]
    angle_colors = ["maroon", "midnightblue", "purple", "teal"]

    for ax, (name, coeff) in zip(axes, plot_methods.items()):
        base_color = METHOD_COLORS.get(name, "black")
        m = compute_metrics(dim=dim, linear_sys=linear_sys, coeff=coeff,
                           v=cfg["v"], h=cfg["h"], tao=cfg["tao"])

        if dim == 1:
            sigma = linear_sys.cal_sigma(r, kh, coeff)
            ax.plot(kh, sigma, color=base_color, linewidth=1.8, label=name)
        elif dim == 2:
            for theta, ls in zip(theta_vals, styles):
                sigma = linear_sys.cal_sigma(r, kh, coeff, theta=theta)
                ax.plot(kh, sigma, color=base_color, linestyle=ls,
                        linewidth=1.4, label=f"θ={_angle_text(theta)}")
        else:
            for theta, ls in zip(theta_vals, styles):
                for phi_, col in zip(phi_vals, angle_colors):
                    sigma = linear_sys.cal_sigma(r, kh, coeff,
                                                 theta=theta, phi=phi_)
                    ax.plot(kh, sigma, color=col, linestyle=ls,
                            linewidth=1.2,
                            label=f"θ={_angle_text(theta)},φ={_angle_text(phi_)}")

        ax.set_facecolor("#f5f5f5")
        ax.set_xlim([0.0, PI])
        ax.set_ylim([0.90, 1.05])
        ax.set_xlabel("kh", fontsize=12)
        ax.tick_params(top=True, right=True, direction="in", length=4)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.grid(True, which="major", linestyle="--", linewidth=0.5)
        ax.set_title(f"{name}\nσ_MAE={m['sigma_mae']:.5f}", fontsize=10,
                     fontweight="bold",
                     color=METHOD_COLORS.get(name, "black"))
        if dim < 3:
            ax.legend(fontsize=7, loc="lower left", frameon=True)
        else:
            ax.legend(fontsize=5, loc="lower left", frameon=True, ncol=2)

    axes[0].set_ylabel("σ(θ,φ)", fontsize=13)
    fig.suptitle(f"{dim}D Dispersion: DOPP Novelty Comparison", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{out_name}")


# ═══════════════════════════════════════════════════════════════════
# Waveform simulation (reuse from Waveform_Simulation.py)
# ═══════════════════════════════════════════════════════════════════
def ricker_wavelet(t, f0):
    u = (PI * f0 * t) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


def second_deriv_stencil(coeff):
    M = len(coeff)
    s = np.zeros(2 * M + 1)
    c = M
    for m in range(M):
        am = float(coeff[m].flat[0]) if hasattr(coeff[m], 'flat') else float(coeff[m])
        s[c + m + 1] += am
        s[c - m - 1] += am
        s[c + m]     -= am
        s[c - m]     -= am
    return s


def simulate_1d_wave(coeff, v, h, dt, nx, nt, src_x, f0, rec_x=None):
    stencil = second_deriv_stencil(coeff)
    M = len(coeff)
    inv_h2 = 1.0 / (h * h)
    v2dt2 = (v * dt) ** 2
    p_prev = np.zeros(nx)
    p_curr = np.zeros(nx)
    p_next = np.zeros(nx)
    seismogram = np.zeros(nt) if rec_x is not None else None
    t_delay = 1.5 / f0
    for it in range(nt):
        t = it * dt
        lap = np.zeros(nx)
        for k in range(-M, M + 1):
            c = stencil[k + M] * inv_h2
            if c == 0.0:
                continue
            lap[M:nx-M] += c * p_curr[M+k:nx-M+k]
        p_next[M:nx-M] = (2.0 * p_curr[M:nx-M] - p_prev[M:nx-M]
                          + v2dt2 * lap[M:nx-M])
        p_next[src_x] += dt * dt * ricker_wavelet(t - t_delay, f0)
        if seismogram is not None:
            seismogram[it] = p_curr[rec_x]
        p_prev, p_curr, p_next = p_curr, p_next, p_prev
    return {"snapshot": p_curr.copy(), "seismogram": seismogram}


def simulate_2d_wave(coeff, v, h, dt, nx, ny, nt, src_x, src_y, f0,
                     rec_x=None, rec_y=None):
    stencil = second_deriv_stencil(coeff)
    M = len(coeff)
    inv_h2 = 1.0 / (h * h)
    v2dt2 = (v * dt) ** 2
    p_prev = np.zeros((nx, ny))
    p_curr = np.zeros((nx, ny))
    p_next = np.zeros((nx, ny))
    seismogram = np.zeros(nt) if rec_x is not None else None
    snap_times = [nt // 4, nt // 2, 3 * nt // 4, nt - 1]
    snapshots = {}
    t_delay = 1.5 / f0
    for it in range(nt):
        t = it * dt
        lap = np.zeros_like(p_curr)
        for k in range(-M, M + 1):
            c = stencil[k + M] * inv_h2
            if c == 0.0:
                continue
            lap[M:nx-M, M:ny-M] += c * p_curr[M+k:nx-M+k, M:ny-M]
            lap[M:nx-M, M:ny-M] += c * p_curr[M:nx-M, M+k:ny-M+k]
        p_next[M:nx-M, M:ny-M] = (2.0 * p_curr[M:nx-M, M:ny-M]
                                   - p_prev[M:nx-M, M:ny-M]
                                   + v2dt2 * lap[M:nx-M, M:ny-M])
        p_next[src_x, src_y] += dt * dt * ricker_wavelet(t - t_delay, f0)
        if seismogram is not None:
            seismogram[it] = p_curr[rec_x, rec_y]
        if it in snap_times:
            snapshots[it] = p_curr.copy()
        p_prev, p_curr, p_next = p_curr, p_next, p_prev
    return {"snapshot": p_curr.copy(), "seismogram": seismogram,
            "snapshots": snapshots, "snap_times": snap_times}


# ═══════════════════════════════════════════════════════════════════
# Waveform figure functions
# ═══════════════════════════════════════════════════════════════════
def plot_seismograms_novelty(all_results, methods, dt, out_path, title):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    for i, name in enumerate(methods):
        seis = all_results[name]["seismogram"]
        if seis is None:
            continue
        t = np.arange(len(seis)) * dt
        offset = -i * 0.35
        norm = np.max(np.abs(seis)) if np.max(np.abs(seis)) > 0 else 1.0
        ax.plot(t * 1000, seis / norm + offset,
                color=METHOD_COLORS.get(name, "black"),
                linewidth=1.3, label=name)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Normalised amplitude (offset)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_facecolor("#f8f8f8")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_2d_snapshots_novelty(all_results, methods, cfg, snap_idx, out_path):
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), dpi=200)
    if n == 1:
        axes = [axes]
    vmax = max(np.abs(all_results[m]["snapshots"][snap_idx]).max()
               for m in methods if snap_idx in all_results[m].get("snapshots", {}))
    vmax = vmax * 0.8 if vmax > 0 else 1.0
    for ax, name in zip(axes, methods):
        snap = all_results[name]["snapshots"].get(snap_idx)
        if snap is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            ax.set_title(name)
            continue
        ax.imshow(snap.T, cmap="seismic", vmin=-vmax, vmax=vmax,
                  aspect="equal", origin="lower",
                  extent=[0, snap.shape[0]*cfg["h"],
                          0, snap.shape[1]*cfg["h"]])
        ax.set_title(name, fontsize=10, fontweight="bold",
                     color=METHOD_COLORS.get(name, "black"))
        ax.set_xlabel("x (m)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("y (m)", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"2D Wavefield Snapshots — DOPP Novelty",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_2d_error_novelty(all_results, methods, ref, cfg, snap_idx, out_path):
    ref_snap = all_results[ref]["snapshots"].get(snap_idx)
    if ref_snap is None:
        return
    others = [m for m in methods if m != ref and m != "Taylor"]
    n = len(others)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), dpi=200)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, others):
        snap = all_results[name]["snapshots"].get(snap_idx)
        if snap is None:
            continue
        err = np.abs(snap - ref_snap)
        vmax = err.max() * 0.9 if err.max() > 0 else 1.0
        im = ax.imshow(err.T, cmap="hot", vmin=0, vmax=vmax,
                       aspect="equal", origin="lower",
                       extent=[0, snap.shape[0]*cfg["h"],
                               0, snap.shape[1]*cfg["h"]])
        ax.set_title(f"|{name} − {ref}|", fontsize=9,
                     color=METHOD_COLORS.get(name, "black"))
        ax.set_xlabel("x (m)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("y (m)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    fig.suptitle(f"Wavefield Error vs {ref}", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, nargs="+", default=[1, 2],
                        choices=[1, 2, 3])
    args = parser.parse_args()

    for dim in args.dim:
        cfg = SIM_CONFIGS[dim]
        print(f"\n{'='*60}")
        print(f"  DOPP Forward Simulation & Waveforms: {dim}D")
        print(f"{'='*60}")

        # Get coefficients
        methods, linear_sys = get_novelty_coefficients(dim, cfg)
        method_order = ["Wang", "QA baseline", "DOPP(Wang)",
                        "QDP", "Full Pipeline"]

        # --- Dispersion curves ---
        print(f"\n  Generating dispersion curves...")
        plot_dispersion_novelty(dim, cfg, methods, linear_sys,
                                f"dopp_dispersion_dim{dim}.png")

        # --- Waveform simulation ---
        if dim == 1:
            print(f"\n  Running 1D waveform simulation...")
            v, h_val, dt = cfg["v"], cfg["h"], cfg["tao"]
            nx, nt = 1000, 2000
            src_x, rec_x = nx // 3, 2 * nx // 3
            f0 = 30.0

            sim_results = {}
            for name in method_order:
                coeff = methods[name]
                print(f"    Simulating {name}...")
                sim_results[name] = simulate_1d_wave(
                    coeff, v, h_val, dt, nx, nt, src_x, f0, rec_x=rec_x)

            plot_seismograms_novelty(
                sim_results, method_order, dt,
                FIGURES_DIR / "dopp_seismogram_1d.png",
                "1D Seismogram: DOPP Novelty Comparison")

        elif dim == 2:
            print(f"\n  Running 2D waveform simulation...")
            v, h_val, dt = cfg["v"], cfg["h"], cfg["tao"]
            nx, ny, nt = 200, 200, 500
            src_x, src_y = nx // 2, ny // 2
            rec_x, rec_y = 3 * nx // 4, 3 * ny // 4
            f0 = 20.0

            sim_results = {}
            for name in method_order:
                coeff = methods[name]
                print(f"    Simulating {name}...")
                sim_results[name] = simulate_2d_wave(
                    coeff, v, h_val, dt, nx, ny, nt,
                    src_x, src_y, f0, rec_x=rec_x, rec_y=rec_y)

            snap_idx = sim_results["Wang"]["snap_times"][2]  # 3/4 time

            plot_2d_snapshots_novelty(
                sim_results, method_order, cfg, snap_idx,
                FIGURES_DIR / "dopp_snapshot_2d.png")

            plot_2d_error_novelty(
                sim_results, method_order, "Wang", cfg, snap_idx,
                FIGURES_DIR / "dopp_error_2d.png")

            plot_seismograms_novelty(
                sim_results, method_order, dt,
                FIGURES_DIR / "dopp_seismogram_2d.png",
                "2D Seismogram: DOPP Novelty Comparison")

    print(f"\n{'='*60}")
    print("All DOPP novelty figures generated!")
    print(f"{'='*60}")
