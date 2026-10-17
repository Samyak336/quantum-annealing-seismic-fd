#!/usr/bin/env python
"""
Waveform Propagation Simulation using Optimised FD Coefficients
================================================================
Generates publication-quality seismic waveform figures comparing:
  - Taylor (baseline)
  - Wang (classical optimised)
  - QA-only (quantum annealing)
  - QA ESE K=5 (ensemble sample exploitation)
  - QA CNAR (condition-number-aware regularisation)
  - QA-to-Wang (hybrid stabilisation)

Produces:
  1. 2D wavefield snapshots at chosen time steps
  2. 1D seismogram traces at receiver locations
  3. Wavefield error maps (difference from Wang reference)

Usage:
    python experiments/Waveform_Simulation.py                # full run (~15-25 min)
    python experiments/Waveform_Simulation.py --quick        # fast test (~3-5 min)
    python experiments/Waveform_Simulation.py --dim 1        # 1D seismograms only
    python experiments/Waveform_Simulation.py --dim 2        # 2D snapshots only
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
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_common import (
    PI, SolverSettings, make_linear_system,
    solve_quantum, solve_taylor, solve_wang,
)

FIGURES_DIR = ROOT.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Physical configurations (matching the rest of the project)
# ═══════════════════════════════════════════════════════════════════
SIM_CONFIGS = {
    1: {"v": 4500, "h": 10, "tao": 0.001, "freq_max": 174, "M": 8,  "epsilon": 1e-8},
    2: {"v": 4500, "h": 20, "tao": 0.001, "freq_max": 90,  "M": 10, "epsilon": 1e-10},
}

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def _silent(func, **kw):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(**kw)


def ricker_wavelet(t, f0):
    """Ricker (Mexican hat) wavelet centered at t=0."""
    u = (PI * f0 * t) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


def second_deriv_stencil(coeff):
    """
    Compute 1D second-derivative stencil from staggered-grid
    first-derivative coefficients.

    The staggered first derivative is:
      D+[f]_i = (1/h) sum_{m=0}^{M-1} a_m [f_{i+m+1} - f_{i-m}]

    The second derivative is D- D+:
      d2f/dx2|_i = (1/h^2) sum_m a_m [f_{i+m+1} + f_{i-m-1} - f_{i+m} - f_{i-m}]

    Returns array of length 2M+1 centered at index M.
    """
    M = len(coeff)
    s = np.zeros(2 * M + 1)
    c = M  # center index
    for m in range(M):
        am = float(coeff[m].flat[0]) if hasattr(coeff[m], 'flat') else float(coeff[m])
        s[c + m + 1] += am   # f_{i+m+1}
        s[c - m - 1] += am   # f_{i-m-1}
        s[c + m]     -= am   # f_{i+m}
        s[c - m]     -= am   # f_{i-m}
    return s


# ═══════════════════════════════════════════════════════════════════
# 2D Acoustic Wave Simulation
# ═══════════════════════════════════════════════════════════════════
def simulate_2d_wave(coeff, v, h, dt, nx, ny, nt, src_x, src_y, f0,
                     rec_x=None, rec_y=None):
    """
    Run 2D acoustic wave equation using staggered-grid FD coefficients.

    Parameters
    ----------
    coeff : array  — FD coefficients (length M)
    v : float      — wave velocity (m/s)
    h : float      — spatial grid spacing (m)
    dt : float     — time step (s)
    nx, ny : int   — grid dimensions
    nt : int       — number of time steps
    src_x, src_y   — source location (grid indices)
    f0 : float     — dominant frequency of Ricker source (Hz)
    rec_x, rec_y   — receiver location (grid indices), optional

    Returns
    -------
    dict with keys: 'snapshot' (2D array at final time),
                    'seismogram' (1D array at receiver),
                    'snapshots' (list of snapshots at selected times)
    """
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

    t_delay = 1.5 / f0  # shift so wavelet starts near zero

    for it in range(nt):
        t = it * dt

        # Compute Laplacian using the stencil
        lap = np.zeros_like(p_curr)
        for k in range(-M, M + 1):
            c = stencil[k + M] * inv_h2
            if c == 0.0:
                continue
            # x-direction
            lap[M:nx-M, M:ny-M] += c * p_curr[M+k:nx-M+k, M:ny-M]
            # y-direction
            lap[M:nx-M, M:ny-M] += c * p_curr[M:nx-M, M+k:ny-M+k]

        # Time stepping
        p_next[M:nx-M, M:ny-M] = (2.0 * p_curr[M:nx-M, M:ny-M]
                                   - p_prev[M:nx-M, M:ny-M]
                                   + v2dt2 * lap[M:nx-M, M:ny-M])

        # Add source
        p_next[src_x, src_y] += dt * dt * ricker_wavelet(t - t_delay, f0)

        # Record seismogram
        if seismogram is not None:
            seismogram[it] = p_curr[rec_x, rec_y]

        # Save snapshots
        if it in snap_times:
            snapshots[it] = p_curr.copy()

        # Swap
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    return {
        "snapshot": p_curr.copy(),
        "seismogram": seismogram,
        "snapshots": snapshots,
        "snap_times": snap_times,
    }


# ═══════════════════════════════════════════════════════════════════
# 1D Acoustic Wave Simulation
# ═══════════════════════════════════════════════════════════════════
def simulate_1d_wave(coeff, v, h, dt, nx, nt, src_x, f0,
                     rec_x=None):
    """Run 1D acoustic wave equation."""
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
        # Laplacian
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


# ═══════════════════════════════════════════════════════════════════
# Coefficient generation for each method
# ═══════════════════════════════════════════════════════════════════
def get_all_coefficients(dim, cfg, max_iter=20, num_sample=100):
    """Compute FD coefficients for all methods."""
    taylor = solve_taylor(cfg["M"])["coeff"]
    common = dict(dim=dim, v=cfg["v"], h=cfg["h"], tao=cfg["tao"],
                  freq_max=cfg["freq_max"], M=cfg["M"],
                  max_iter_num=max_iter, alpha=1.0, alpha_decay=0.1,
                  beta=1.0, epsilon=cfg["epsilon"], coeff_init=taylor)

    base_settings = SolverSettings(num_sample=num_sample, R=10,
                                   fixed_point=1, inner_maxiter=3)

    print("  [1/6] Taylor coefficients (analytic)")
    methods = {"Taylor": taylor}

    print("  [2/6] Wang solver...")
    wang = _silent(solve_wang, **common)
    methods["Wang"] = wang["coeff"]

    print("  [3/6] QA-only (K=1)...")
    qa = _silent(solve_quantum, **common, settings=base_settings)
    methods["QA (K=1)"] = qa["coeff"]

    print("  [4/6] QA ESE (K=5)...")
    ese_settings = SolverSettings(num_sample=num_sample, R=10,
                                  fixed_point=1, inner_maxiter=3,
                                  ensemble_k=5)
    qa_ese = _silent(solve_quantum, **common, settings=ese_settings)
    methods["QA ESE (K=5)"] = qa_ese["coeff"]

    print("  [5/6] QA CNAR...")
    # Compute CNAR alpha from condition number
    ls = make_linear_system(dim=dim, v=cfg["v"], h=cfg["h"],
                            tao=cfg["tao"], freq_max=cfg["freq_max"])
    A, b = ls.construct(cfg["M"], taylor)
    kappa = np.linalg.cond(A.T @ A)
    cnar_alpha = 0.1 * max(0, np.log10(kappa) - 6.0)
    # Use the improved threshold (4.0) so CNAR actually activates
    cnar_alpha_improved = 0.1 * max(0, np.log10(kappa) - 4.0)
    print(f"    kappa={kappa:.2e}, log10(kappa)={np.log10(kappa):.2f}, "
          f"original_alpha={cnar_alpha:.4f}, improved_alpha={cnar_alpha_improved:.4f}")
    qa_cnar = _silent(solve_quantum, **{**common,
                      "alpha": cnar_alpha_improved,
                      "alpha_decay": 0.5},
                      settings=base_settings)
    methods["QA CNAR"] = qa_cnar["coeff"]

    print("  [6/6] QA-to-Wang hybrid...")
    qa_coarse = _silent(solve_quantum, **{**common, "max_iter_num": 8},
                        settings=base_settings)
    wang_polish = _silent(solve_wang, **{**common, "max_iter_num": 12,
                          "coeff_init": qa_coarse["coeff"]})
    methods["QA→Wang"] = wang_polish["coeff"]

    return methods


# ═══════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════
METHOD_COLORS = {
    "Taylor": "#888888",
    "Wang": "forestgreen",
    "QA (K=1)": "royalblue",
    "QA ESE (K=5)": "darkorange",
    "QA CNAR": "crimson",
    "QA→Wang": "purple",
}


def plot_2d_snapshots(all_results, methods_order, cfg, snap_idx, out_path):
    """Panel of 2D wavefield snapshots across methods."""
    n = len(methods_order)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), dpi=200)
    if n == 1:
        axes = [axes]

    vmax = max(np.abs(all_results[m]["snapshots"][snap_idx]).max()
               for m in methods_order if snap_idx in all_results[m]["snapshots"])
    vmax = vmax * 0.8 if vmax > 0 else 1.0

    for ax, method in zip(axes, methods_order):
        snap = all_results[method]["snapshots"].get(snap_idx)
        if snap is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            ax.set_title(method, fontsize=10)
            continue

        im = ax.imshow(snap.T, cmap="seismic", vmin=-vmax, vmax=vmax,
                       aspect="equal", origin="lower",
                       extent=[0, snap.shape[0]*cfg["h"],
                               0, snap.shape[1]*cfg["h"]])
        ax.set_title(method, fontsize=10, fontweight="bold",
                     color=METHOD_COLORS.get(method, "black"))
        ax.set_xlabel("x (m)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("y (m)", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"2D Wavefield Snapshots (t-step {snap_idx})",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_2d_error_maps(all_results, methods_order, ref_method, cfg,
                       snap_idx, out_path):
    """Error maps: difference of each method from the reference (Wang)."""
    ref_snap = all_results[ref_method]["snapshots"].get(snap_idx)
    if ref_snap is None:
        return

    others = [m for m in methods_order if m != ref_method]
    n = len(others)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), dpi=200)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, others):
        snap = all_results[method]["snapshots"].get(snap_idx)
        if snap is None:
            continue
        err = np.abs(snap - ref_snap)
        vmax_err = err.max() * 0.9 if err.max() > 0 else 1.0

        im = ax.imshow(err.T, cmap="hot", vmin=0, vmax=vmax_err,
                       aspect="equal", origin="lower",
                       extent=[0, snap.shape[0]*cfg["h"],
                               0, snap.shape[1]*cfg["h"]])
        ax.set_title(f"|{method} − {ref_method}|", fontsize=9,
                     color=METHOD_COLORS.get(method, "black"))
        ax.set_xlabel("x (m)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("y (m)", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    fig.suptitle(f"2D Wavefield Error Maps vs {ref_method}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_seismograms(all_results, methods_order, dt, out_path, title=""):
    """Overlay seismogram traces from all methods."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    for i, method in enumerate(methods_order):
        seis = all_results[method]["seismogram"]
        if seis is None:
            continue
        t = np.arange(len(seis)) * dt
        # Offset for visibility
        offset = -i * 0.3
        norm = np.max(np.abs(seis)) if np.max(np.abs(seis)) > 0 else 1.0
        ax.plot(t * 1000, seis / norm + offset,
                color=METHOD_COLORS.get(method, "black"),
                linewidth=1.2, label=method)

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Normalised amplitude (offset)", fontsize=12)
    ax.set_title(title or "Seismogram Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_facecolor("#f8f8f8")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_seismogram_detail(all_results, methods_order, dt, out_path,
                           title="", zoom_start_ms=None, zoom_end_ms=None):
    """Seismograms stacked vertically with optional zoom window."""
    n = len(methods_order)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n), dpi=200,
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_order):
        seis = all_results[method]["seismogram"]
        if seis is None:
            continue
        t = np.arange(len(seis)) * dt * 1000  # ms
        ax.plot(t, seis, color=METHOD_COLORS.get(method, "black"),
                linewidth=1.0)
        ax.set_ylabel(method, fontsize=9, rotation=0, ha="right", va="center")
        ax.set_facecolor("#f8f8f8")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)
        if zoom_start_ms is not None and zoom_end_ms is not None:
            ax.set_xlim(zoom_start_ms, zoom_end_ms)

    axes[-1].set_xlabel("Time (ms)", fontsize=11)
    fig.suptitle(title or "Seismogram Detail", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main simulation drivers
# ═══════════════════════════════════════════════════════════════════
def run_2d_simulation(cfg, coefficients, quick=False):
    """Run 2D wave simulation for all methods and generate figures."""
    v, h, M = cfg["v"], cfg["h"], cfg["M"]
    f0 = cfg["freq_max"] * 0.4  # dominant frequency for Ricker

    nx = ny = 200 if quick else 300
    src_x, src_y = nx // 2, ny // 2
    rec_x, rec_y = nx // 2 + nx // 4, ny // 2

    # CFL condition
    coeff_abs_sum = max(
        sum(abs(float(c.flat[0]) if hasattr(c, 'flat') else float(c))
            for c in coefficients[m])
        for m in coefficients
    )
    dt_max = h / (v * np.sqrt(2) * coeff_abs_sum)
    dt = 0.8 * dt_max
    nt = int(0.25 / dt) if not quick else int(0.12 / dt)  # simulation duration

    print(f"\n  2D Simulation: {nx}x{ny} grid, {nt} steps, dt={dt*1e6:.1f} μs, "
          f"f0={f0:.0f} Hz")

    all_results = {}
    methods_order = [m for m in ["Taylor", "Wang", "QA (K=1)",
                                  "QA ESE (K=5)", "QA CNAR", "QA→Wang"]
                     if m in coefficients]

    for method in methods_order:
        print(f"  Simulating {method}...", end=" ", flush=True)
        t0 = time.time()
        result = simulate_2d_wave(
            coeff=coefficients[method], v=v, h=h, dt=dt,
            nx=nx, ny=ny, nt=nt, src_x=src_x, src_y=src_y, f0=f0,
            rec_x=rec_x, rec_y=rec_y,
        )
        print(f"done ({time.time()-t0:.1f}s)")
        all_results[method] = result

    # --- Generate figures ---
    snap_idx = all_results[methods_order[0]]["snap_times"][2]  # 3/4 time

    # Full comparison snapshots
    plot_2d_snapshots(all_results, methods_order, cfg, snap_idx,
                      FIGURES_DIR / "waveform_snapshot_comparison.png")

    # Error maps vs Wang
    plot_2d_error_maps(all_results, methods_order, "Wang", cfg, snap_idx,
                       FIGURES_DIR / "waveform_error_vs_wang.png")

    # Seismograms
    plot_seismograms(all_results, methods_order, dt,
                     FIGURES_DIR / "waveform_seismogram_overlay.png",
                     title="2D Seismogram Comparison at Receiver")
    plot_seismogram_detail(all_results, methods_order, dt,
                           FIGURES_DIR / "waveform_seismogram_detail.png",
                           title="2D Seismogram Detail")

    # Novelty-specific panels (3 methods each)
    for tag, subset in [
        ("ese",     ["Wang", "QA (K=1)", "QA ESE (K=5)"]),
        ("cnar",    ["Wang", "QA (K=1)", "QA CNAR"]),
        ("qa2wang", ["Wang", "QA (K=1)", "QA→Wang"]),
    ]:
        sub = [m for m in subset if m in all_results]
        if len(sub) >= 2:
            plot_2d_snapshots(all_results, sub, cfg, snap_idx,
                              FIGURES_DIR / f"waveform_snapshot_{tag}.png")
            plot_seismograms(all_results, sub, dt,
                             FIGURES_DIR / f"waveform_seismogram_{tag}.png",
                             title=f"Seismogram: {tag.upper()} comparison")

    return all_results


def run_1d_simulation(cfg, coefficients, quick=False):
    """Run 1D wave simulation for all methods and generate figures."""
    v, h, M = cfg["v"], cfg["h"], cfg["M"]
    f0 = cfg["freq_max"] * 0.4

    nx = 800 if not quick else 400
    src_x = nx // 2
    rec_x = nx // 2 + nx // 4

    coeff_abs_sum = max(
        sum(abs(float(c.flat[0]) if hasattr(c, 'flat') else float(c))
            for c in coefficients[m])
        for m in coefficients
    )
    dt_max = h / (v * coeff_abs_sum)
    dt = 0.8 * dt_max
    nt = int(0.3 / dt) if not quick else int(0.15 / dt)

    print(f"\n  1D Simulation: {nx} points, {nt} steps, dt={dt*1e6:.1f} μs")

    all_results = {}
    methods_order = [m for m in ["Taylor", "Wang", "QA (K=1)",
                                  "QA ESE (K=5)", "QA CNAR", "QA→Wang"]
                     if m in coefficients]

    for method in methods_order:
        print(f"  Simulating {method}...", end=" ", flush=True)
        t0 = time.time()
        result = simulate_1d_wave(
            coeff=coefficients[method], v=v, h=h, dt=dt,
            nx=nx, nt=nt, src_x=src_x, f0=f0, rec_x=rec_x,
        )
        print(f"done ({time.time()-t0:.1f}s)")
        all_results[method] = result

    plot_seismograms(all_results, methods_order, dt,
                     FIGURES_DIR / "waveform_1d_seismogram.png",
                     title="1D Seismogram Comparison")
    plot_seismogram_detail(all_results, methods_order, dt,
                           FIGURES_DIR / "waveform_1d_seismogram_detail.png",
                           title="1D Seismogram Detail")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate seismic waveform propagation figures")
    parser.add_argument("--dim", type=int, choices=[1, 2], default=None,
                        help="Run only 1D or 2D (default: both)")
    parser.add_argument("--quick", action="store_true",
                        help="Faster run with smaller grid and fewer steps")
    args = parser.parse_args()

    max_iter = 10 if args.quick else 20
    num_sample = 50 if args.quick else 100
    dims = [args.dim] if args.dim else [1, 2]

    for dim in dims:
        cfg = SIM_CONFIGS[dim]
        print(f"\n{'='*60}")
        print(f"  Generating {dim}D waveform figures")
        print(f"{'='*60}")

        print("\n  Computing FD coefficients for all methods...")
        coefficients = get_all_coefficients(dim, cfg, max_iter, num_sample)

        if dim == 2:
            run_2d_simulation(cfg, coefficients, quick=args.quick)
        else:
            run_1d_simulation(cfg, coefficients, quick=args.quick)

    print(f"\n{'='*60}")
    print(f"  All waveform figures saved to: {FIGURES_DIR}")
    print(f"{'='*60}")
